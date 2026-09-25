# -*- coding: utf-8 -*-
"""Spatial + FIRE dual-stream detector with anomaly-guided cross-attention.

Design goals
------------
1) Keep the original single-branch SpatialClassifier and FIREOfficialModel intact
   and reusable.
2) Reuse their pretrained checkpoints as branch initialization.
3) Add a stronger fusion block than simple scalar gating:
   - FIRE anomaly prior guides token interaction
   - bidirectional cross-attention exchanges local evidence
   - sample-level credibility router decides branch trustworthiness
4) Keep the default forward() compatible with the current training / eval code:
   returns final 2-class logits.
5) Expose return_aux=True for richer multi-loss training.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial import SpatialClassifier
from .fire_official import FIREOfficialModel


class _ImageNetNorm(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MLPBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(64, int(dim * mlp_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusionBlock(nn.Module):
    """Bidirectional cross-attention with FIRE anomaly-prior guidance."""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.s_norm = nn.LayerNorm(dim)
        self.f_norm = nn.LayerNorm(dim)
        self.cross_s = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.cross_f = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.s_post = nn.LayerNorm(dim)
        self.f_post = nn.LayerNorm(dim)
        self.s_mlp = MLPBlock(dim, mlp_ratio=4.0, dropout=dropout)
        self.f_mlp = MLPBlock(dim, mlp_ratio=4.0, dropout=dropout)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        s_tokens: torch.Tensor,
        f_tokens: torch.Tensor,
        fire_prior: torch.Tensor,
        use_anomaly_guidance: bool = True,
    ):
        # fire_prior: [B, N, 1], normalized to [0,1]
        s_q = self.s_norm(s_tokens)
        f_q = self.f_norm(f_tokens)
        if use_anomaly_guidance:
            gate = 1.0 + fire_prior
            inv_gate = 2.0 - gate
            f_guided = f_q * gate
            s_guided = s_q * inv_gate
        else:
            # Capacity-controlled baseline: retain exactly the same attention,
            # FFN and normalization parameters, but remove anomaly modulation.
            f_guided = f_q
            s_guided = s_q

        s_delta, _ = self.cross_s(query=s_q, key=f_guided, value=f_guided, need_weights=False)
        f_delta, _ = self.cross_f(query=f_q, key=s_guided, value=s_guided, need_weights=False)

        s_tokens = s_tokens + self.drop(s_delta)
        f_tokens = f_tokens + self.drop(f_delta)

        s_tokens = s_tokens + self.drop(self.s_mlp(self.s_post(s_tokens)))
        f_tokens = f_tokens + self.drop(self.f_mlp(self.f_post(f_tokens)))
        return s_tokens, f_tokens


class CredibilityRouter(nn.Module):
    """Sample-level reliability estimation for the two branches."""
    def __init__(self, feat_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        # spatial_feat + fire_feat + spatial_logits(2) + fire_logit(1) +
        # s_entropy(1) + f_entropy(1) + prior_mean(1) + prior_std(1)
        in_dim = feat_dim * 2 + 7
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 2),
        )

    @staticmethod
    def _binary_entropy_from_prob(prob_pos: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        p = prob_pos.clamp(min=eps, max=1.0 - eps)
        ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        return ent.unsqueeze(1)

    def forward(
        self,
        spatial_feat: torch.Tensor,
        fire_feat: torch.Tensor,
        spatial_logits: torch.Tensor,
        fire_logit: torch.Tensor,
        prior_mean: torch.Tensor,
        prior_std: torch.Tensor,
    ) -> torch.Tensor:
        s_prob = torch.softmax(spatial_logits, dim=1)[:, 1]
        f_prob = torch.sigmoid(fire_logit.view(-1))
        s_ent = self._binary_entropy_from_prob(s_prob)
        f_ent = self._binary_entropy_from_prob(f_prob)
        router_in = torch.cat(
            [
                spatial_feat,
                fire_feat,
                spatial_logits,
                fire_logit.view(-1, 1),
                s_ent,
                f_ent,
                prior_mean,
                prior_std,
            ],
            dim=1,
        )
        return torch.softmax(self.net(router_in), dim=1)


class ConventionalScalarGate(nn.Module):
    """Standard sample-wise gate used as a conventional fusion baseline."""
    def __init__(self, feat_dim: int, hidden_dim: int = 512, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, spatial_feat: torch.Tensor, fire_feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([spatial_feat, fire_feat], dim=1)))


class CapacityMatchedConcatHead(nn.Module):
    """Concatenation MLP whose parameter count is matched to the full fusion head."""
    def __init__(self, feat_dim: int, target_params: int, dropout: float = 0.0):
        super().__init__()
        # Params for Linear(2d,h) + Linear(h,d) + Linear(d,2):
        # h*(3d+1) + 3d + 2. Choose the closest positive integer h.
        denom = 3 * feat_dim + 1
        hidden_dim = max(1, int(round((int(target_params) - 3 * feat_dim - 2) / denom)))
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(feat_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, feat_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(feat_dim, 2),
        )

    def forward(self, spatial_feat: torch.Tensor, fire_feat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([spatial_feat, fire_feat], dim=1))


class SpatialFIREDualStreamModel(nn.Module):
    """Formal dual-stream model built on top of the existing Spatial + FIRE branches."""
    def __init__(
        self,
        spatial_backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = False,
        dropout: float = 0.1,
        fuse_dim: int = 512,
        num_heads: int = 8,
        router_hidden: int = 256,
        temperature: float = 0.07,
        fusion_variant: str = "full",
    ):
        super().__init__()
        if int(num_classes) != 2:
            raise ValueError("SpatialFIREDualStreamModel currently expects num_classes=2.")

        self.temperature = float(temperature)
        self.fusion_variant = str(fusion_variant).lower().strip()
        valid_variants = {
            "full",
            "average",
            "concat_linear",
            "gated",
            "concat_mlp_matched",
            "standard_crossattn",
            "router_only",
            "no_routing",
        }
        if self.fusion_variant not in valid_variants:
            raise ValueError(
                f"Unsupported fusion_variant={fusion_variant!r}; choose one of {sorted(valid_variants)}"
            )
        self.rgb_norm = _ImageNetNorm()

        self.spatial_branch = SpatialClassifier(
            backbone=spatial_backbone,
            num_classes=2,
            pretrained=bool(pretrained),
            dropout=float(dropout),
        )
        self.fire_branch = FIREOfficialModel(mode="frq", norm_layer="instance", pretrained=bool(pretrained))

        self.spatial_dim = self._infer_spatial_dim()
        self.fire_dim = 2048
        self.fuse_dim = int(fuse_dim)

        self.spatial_map_proj = nn.Conv2d(self.spatial_dim, self.fuse_dim, kernel_size=1, bias=False)
        self.fire_map_proj = nn.Conv2d(self.fire_dim, self.fuse_dim, kernel_size=1, bias=False)
        self.spatial_vec_proj = nn.Linear(self.spatial_dim, self.fuse_dim)
        self.fire_vec_proj = nn.Linear(self.fire_dim, self.fuse_dim)

        use_crossattn = self.fusion_variant in {"full", "standard_crossattn", "no_routing"}
        use_router = self.fusion_variant in {"full", "standard_crossattn", "router_only"}
        use_interaction_terms = self.fusion_variant in {"full", "standard_crossattn", "router_only", "no_routing"}

        self.fusion_block = (
            CrossAttentionFusionBlock(dim=self.fuse_dim, num_heads=int(num_heads), dropout=float(dropout))
            if use_crossattn else None
        )
        self.router = (
            CredibilityRouter(feat_dim=self.fuse_dim, hidden_dim=int(router_hidden), dropout=float(dropout))
            if use_router else None
        )
        self.interaction_mlp = (
            nn.Sequential(
                nn.Linear(self.fuse_dim * 4, self.fuse_dim),
                nn.GELU(),
                nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(self.fuse_dim, self.fuse_dim),
            )
            if use_interaction_terms else None
        )
        self.final_head = nn.Linear(self.fuse_dim, 2) if self.fusion_variant not in {"concat_linear", "concat_mlp_matched"} else None
        self.concat_head = nn.Linear(self.fuse_dim * 2, 2) if self.fusion_variant == "concat_linear" else None
        self.conventional_gate = (
            ConventionalScalarGate(self.fuse_dim, hidden_dim=self.fuse_dim, dropout=float(dropout))
            if self.fusion_variant == "gated" else None
        )
        self.matched_concat_head = None
        if self.fusion_variant == "concat_mlp_matched":
            # Count the full model's replaceable fusion components without
            # registering them. Common branch/projection parameters are equal.
            reference_modules = [
                CrossAttentionFusionBlock(self.fuse_dim, int(num_heads), float(dropout)),
                CredibilityRouter(self.fuse_dim, int(router_hidden), float(dropout)),
                nn.Sequential(
                    nn.Linear(self.fuse_dim * 4, self.fuse_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if float(dropout) > 0 else nn.Identity(),
                    nn.Linear(self.fuse_dim, self.fuse_dim),
                ),
                nn.Linear(self.fuse_dim, 2),
            ]
            target_params = sum(p.numel() for module in reference_modules for p in module.parameters())
            self.matched_concat_head = CapacityMatchedConcatHead(
                feat_dim=self.fuse_dim,
                target_params=target_params,
                dropout=float(dropout),
            )
            del reference_modules

        self.proj_spatial = nn.Linear(self.fuse_dim, self.fuse_dim)
        self.proj_fire = nn.Linear(self.fuse_dim, self.fuse_dim)

    def _infer_spatial_dim(self) -> int:
        # Prefer the explicit attribute exposed by the updated SpatialClassifier.
        if hasattr(self.spatial_branch, "feature_dim"):
            return int(getattr(self.spatial_branch, "feature_dim"))

        # Backward compatibility: some older spatial branches may expose .feat.num_features.
        feat = getattr(self.spatial_branch, "feat", None)
        if feat is not None and hasattr(feat, "num_features"):
            return int(getattr(feat, "num_features"))

        raise AttributeError(
            "Spatial branch does not expose feature_dim or .feat.num_features; cannot build dual-stream model."
        )

    @staticmethod
    def _strip_wrappers(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk.startswith("model."):
                nk = nk[len("model."):]
            out[nk] = v
        return out

    def load_pretrained_branches(self, spatial_ckpt: str = "", fire_ckpt: str = "", strict: bool = False):
        report = {}
        if spatial_ckpt:
            payload = torch.load(spatial_ckpt, map_location="cpu")
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            state = self._strip_wrappers(state)
            missing, unexpected = self.spatial_branch.load_state_dict(state, strict=bool(strict))
            report["spatial"] = {"missing": list(missing), "unexpected": list(unexpected)}
        if fire_ckpt:
            payload = torch.load(fire_ckpt, map_location="cpu")
            state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
            state = self._strip_wrappers(state)
            missing, unexpected = self.fire_branch.load_state_dict(state, strict=bool(strict))
            report["fire"] = {"missing": list(missing), "unexpected": list(unexpected)}
        return report

    def set_backbones_trainable(self, trainable: bool = True):
        for module in [self.spatial_branch, self.fire_branch]:
            for p in module.parameters():
                p.requires_grad = bool(trainable)
        if trainable:
            self.spatial_branch.train()
            self.fire_branch.train()
        else:
            self.spatial_branch.eval()
            self.fire_branch.eval()

    def _extract_spatial_features(self, x: torch.Tensor):
        xs = self.rgb_norm(x)

        # Preferred path: the updated SpatialClassifier exposes extract_features().
        if hasattr(self.spatial_branch, "extract_features"):
            out = self.spatial_branch.extract_features(xs, return_logits=True)
            fmap = out.get("feature_map", None)
            pooled = out.get("global_feature", None)
            logits = out.get("logits", None)
            if fmap is None or pooled is None or logits is None:
                raise ValueError("Spatial branch extract_features() did not return feature_map/global_feature/logits.")
            if fmap.ndim != 4:
                raise ValueError(f"Expected spatial feature map [B,C,H,W], got {tuple(fmap.shape)}")
            return fmap, pooled, logits

        # Backward compatibility: older timm-style implementation with .feat/.dropout/.head.
        feat_module = getattr(self.spatial_branch, "feat", None)
        if feat_module is None:
            raise AttributeError(
                "Spatial branch exposes neither extract_features() nor .feat; cannot obtain spatial features."
            )
        if not hasattr(feat_module, "forward_features"):
            raise AttributeError("Spatial timm backbone does not expose forward_features().")
        fmap = feat_module.forward_features(xs)
        if isinstance(fmap, (tuple, list)):
            fmap = fmap[-1]
        if fmap.ndim != 4:
            raise ValueError(f"Expected spatial feature map [B,C,H,W], got {tuple(fmap.shape)}")

        if hasattr(feat_module, "forward_head"):
            pooled = feat_module.forward_head(fmap, pre_logits=True)
        else:
            pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)

        dropout = getattr(self.spatial_branch, "dropout", None)
        head = getattr(self.spatial_branch, "head", None)
        if dropout is not None:
            pooled = dropout(pooled)
        if head is None:
            raise AttributeError("Spatial branch fallback path requires .head for logits computation.")
        logits = head(pooled)
        return fmap, pooled, logits

    @staticmethod
    def _binary_to_two_class(logit: torch.Tensor) -> torch.Tensor:
        logit = logit.view(-1)
        return torch.stack([-logit, logit], dim=1)

    def _compute_router_regularization(self, router_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        eps = 1e-6
        mean_w = router_weights.mean(dim=0)
        balance = torch.sum((mean_w - 0.5) ** 2)
        entropy = -torch.mean(torch.sum(router_weights.clamp_min(eps) * torch.log(router_weights.clamp_min(eps)), dim=1))
        return {"router_balance": balance, "router_entropy": entropy}

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        s_map, s_vec, s_logits = self._extract_spatial_features(x)
        fire_aux = self.fire_branch.extract_features(x, return_aux=True)
        f_map = fire_aux["feature_map"]
        f_vec = fire_aux["global_feature"]
        f_logit = fire_aux["logits"].view(-1)
        f_logits_2c = self._binary_to_two_class(f_logit)

        prior = fire_aux["anomaly_prior"]
        if prior.ndim != 4 or prior.shape[1] != 1:
            raise ValueError(f"FIRE anomaly_prior should be [B,1,H,W], got {tuple(prior.shape)}")

        s_map_proj = self.spatial_map_proj(s_map)
        f_map_proj = self.fire_map_proj(f_map)

        if s_map_proj.shape[-2:] != f_map_proj.shape[-2:]:
            target_hw = s_map_proj.shape[-2:]
            f_map_proj = F.interpolate(f_map_proj, size=target_hw, mode="bilinear", align_corners=False)
        else:
            target_hw = s_map_proj.shape[-2:]

        prior_small = F.interpolate(prior, size=target_hw, mode="bilinear", align_corners=False)
        prior_flat = prior_small.flatten(2).transpose(1, 2)
        prior_flat = prior_flat / (prior_flat.amax(dim=1, keepdim=True) + 1e-6)

        s_tokens = s_map_proj.flatten(2).transpose(1, 2)
        f_tokens = f_map_proj.flatten(2).transpose(1, 2)
        if self.fusion_block is not None:
            s_tokens, f_tokens = self.fusion_block(
                s_tokens,
                f_tokens,
                prior_flat,
                use_anomaly_guidance=(self.fusion_variant != "standard_crossattn"),
            )

        s_local = s_tokens.mean(dim=1)
        f_local = f_tokens.mean(dim=1)

        s_global = self.spatial_vec_proj(s_vec)
        f_global = self.fire_vec_proj(f_vec)
        s_repr = s_global + s_local
        f_repr = f_global + f_local

        prior_mean = prior_small.mean(dim=(2, 3))
        prior_std = prior_small.flatten(2).std(dim=2)
        if self.router is not None:
            router_weights = self.router(
                spatial_feat=s_repr,
                fire_feat=f_repr,
                spatial_logits=s_logits,
                fire_logit=f_logit,
                prior_mean=prior_mean,
                prior_std=prior_std.unsqueeze(1) if prior_std.ndim == 1 else prior_std,
            )
        elif self.fusion_variant == "gated":
            alpha = self.conventional_gate(s_repr, f_repr)
            router_weights = torch.cat([alpha, 1.0 - alpha], dim=1)
        else:
            router_weights = torch.full(
                (s_repr.size(0), 2), 0.5, dtype=s_repr.dtype, device=s_repr.device
            )

        if self.fusion_variant == "concat_linear":
            logits = self.concat_head(torch.cat([s_repr, f_repr], dim=1))
        elif self.fusion_variant == "concat_mlp_matched":
            logits = self.matched_concat_head(s_repr, f_repr)
        elif self.fusion_variant == "average":
            logits = self.final_head(0.5 * (s_repr + f_repr))
        elif self.fusion_variant == "gated":
            logits = self.final_head(router_weights[:, 0:1] * s_repr + router_weights[:, 1:2] * f_repr)
        else:
            ws = router_weights[:, 0:1]
            wf = router_weights[:, 1:2]
            gated = ws * s_repr + wf * f_repr
            interaction = self.interaction_mlp(
                torch.cat([s_repr, f_repr, torch.abs(s_repr - f_repr), s_repr * f_repr], dim=1)
            )
            logits = self.final_head(gated + interaction)

        if not return_aux:
            return logits

        router_reg = self._compute_router_regularization(router_weights)
        return {
            "logits": logits,
            "spatial_logits": s_logits,
            "fire_logits_2c": f_logits_2c,
            "fire_logits_raw": f_logit,
            "router_weights": router_weights,
            "router_balance": router_reg["router_balance"],
            "router_entropy": router_reg["router_entropy"],
            "spatial_proj": F.normalize(self.proj_spatial(s_repr), dim=1),
            "fire_proj": F.normalize(self.proj_fire(f_repr), dim=1),
            "anomaly_prior": prior,
            "prior_mean": prior_mean,
            "prior_std": prior_std.unsqueeze(1) if prior_std.ndim == 1 else prior_std,
        }
