"""Lightweight fusion-only model for cached Spatial/FIRE branch features.

This is used only by the capacity-controlled reviewer experiment. Branch
encoders are trained once, frozen, and evaluated once; every fusion strategy
then sees byte-identical cached branch evidence.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spatial_fire_dualstream import (
    CapacityMatchedConcatHead,
    ConventionalScalarGate,
    CredibilityRouter,
    CrossAttentionFusionBlock,
)


class CachedSpatialFIREFusion(nn.Module):
    def __init__(
        self,
        fusion_variant: str = "full",
        spatial_dim: int = 2048,
        fire_dim: int = 2048,
        fuse_dim: int = 512,
        num_heads: int = 8,
        router_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_variant = str(fusion_variant).lower().strip()
        valid = {"full", "concat_linear", "gated", "concat_mlp_matched", "standard_crossattn"}
        if self.fusion_variant not in valid:
            raise ValueError(f"Unsupported cached fusion variant: {fusion_variant}; choose from {sorted(valid)}")
        self.fuse_dim = int(fuse_dim)

        # These modules and dimensions exactly mirror the replaceable part of
        # SpatialFIREDualStreamModel; only the heavy branch encoders are absent.
        self.spatial_map_proj = nn.Conv2d(spatial_dim, fuse_dim, kernel_size=1, bias=False)
        self.fire_map_proj = nn.Conv2d(fire_dim, fuse_dim, kernel_size=1, bias=False)
        self.spatial_vec_proj = nn.Linear(spatial_dim, fuse_dim)
        self.fire_vec_proj = nn.Linear(fire_dim, fuse_dim)

        use_crossattn = self.fusion_variant in {"full", "standard_crossattn"}
        use_router = self.fusion_variant in {"full", "standard_crossattn"}
        use_interaction = self.fusion_variant in {"full", "standard_crossattn"}
        self.fusion_block = (
            CrossAttentionFusionBlock(fuse_dim, num_heads=num_heads, dropout=dropout) if use_crossattn else None
        )
        self.router = CredibilityRouter(fuse_dim, router_hidden, dropout) if use_router else None
        self.interaction_mlp = (
            nn.Sequential(
                nn.Linear(fuse_dim * 4, fuse_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(fuse_dim, fuse_dim),
            )
            if use_interaction else None
        )
        self.final_head = nn.Linear(fuse_dim, 2) if self.fusion_variant not in {"concat_linear", "concat_mlp_matched"} else None
        self.concat_head = nn.Linear(fuse_dim * 2, 2) if self.fusion_variant == "concat_linear" else None
        self.conventional_gate = ConventionalScalarGate(fuse_dim, fuse_dim, dropout) if self.fusion_variant == "gated" else None
        self.matched_concat_head = None
        if self.fusion_variant == "concat_mlp_matched":
            reference = [
                CrossAttentionFusionBlock(fuse_dim, num_heads, dropout),
                CredibilityRouter(fuse_dim, router_hidden, dropout),
                nn.Sequential(
                    nn.Linear(fuse_dim * 4, fuse_dim), nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(fuse_dim, fuse_dim),
                ),
                nn.Linear(fuse_dim, 2),
            ]
            target_params = sum(p.numel() for module in reference for p in module.parameters())
            self.matched_concat_head = CapacityMatchedConcatHead(fuse_dim, target_params, dropout)
            del reference

        self.proj_spatial = nn.Linear(fuse_dim, fuse_dim)
        self.proj_fire = nn.Linear(fuse_dim, fuse_dim)

    @staticmethod
    def _router_regularization(weights: torch.Tensor):
        eps = 1e-6
        mean_w = weights.mean(dim=0)
        balance = torch.sum((mean_w - 0.5) ** 2)
        entropy = -torch.mean(torch.sum(weights.clamp_min(eps) * torch.log(weights.clamp_min(eps)), dim=1))
        return balance, entropy

    def forward(
        self,
        spatial_map: torch.Tensor,
        spatial_vec: torch.Tensor,
        spatial_logits: torch.Tensor,
        fire_map: torch.Tensor,
        fire_vec: torch.Tensor,
        fire_logit: torch.Tensor,
        anomaly_prior: torch.Tensor,
        return_aux: bool = False,
    ):
        s_map = self.spatial_map_proj(spatial_map)
        f_map = self.fire_map_proj(fire_map)
        if f_map.shape[-2:] != s_map.shape[-2:]:
            f_map = F.interpolate(f_map, size=s_map.shape[-2:], mode="bilinear", align_corners=False)
        prior = F.interpolate(anomaly_prior, size=s_map.shape[-2:], mode="bilinear", align_corners=False)
        prior_flat = prior.flatten(2).transpose(1, 2)
        prior_flat = prior_flat / (prior_flat.amax(dim=1, keepdim=True) + 1e-6)

        s_tokens = s_map.flatten(2).transpose(1, 2)
        f_tokens = f_map.flatten(2).transpose(1, 2)
        if self.fusion_block is not None:
            s_tokens, f_tokens = self.fusion_block(
                s_tokens, f_tokens, prior_flat,
                use_anomaly_guidance=(self.fusion_variant != "standard_crossattn"),
            )
        s_repr = self.spatial_vec_proj(spatial_vec) + s_tokens.mean(dim=1)
        f_repr = self.fire_vec_proj(fire_vec) + f_tokens.mean(dim=1)

        fire_logit = fire_logit.view(-1)
        fire_logits_2c = torch.stack([-fire_logit, fire_logit], dim=1)
        prior_mean = prior.mean(dim=(2, 3))
        prior_std = prior.flatten(2).std(dim=2)
        if self.router is not None:
            weights = self.router(
                s_repr, f_repr, spatial_logits, fire_logit,
                prior_mean, prior_std.unsqueeze(1) if prior_std.ndim == 1 else prior_std,
            )
        elif self.fusion_variant == "gated":
            alpha = self.conventional_gate(s_repr, f_repr)
            weights = torch.cat([alpha, 1.0 - alpha], dim=1)
        else:
            weights = torch.full((s_repr.size(0), 2), 0.5, device=s_repr.device, dtype=s_repr.dtype)

        if self.fusion_variant == "concat_linear":
            logits = self.concat_head(torch.cat([s_repr, f_repr], dim=1))
        elif self.fusion_variant == "concat_mlp_matched":
            logits = self.matched_concat_head(s_repr, f_repr)
        elif self.fusion_variant == "gated":
            logits = self.final_head(weights[:, :1] * s_repr + weights[:, 1:] * f_repr)
        else:
            gated = weights[:, :1] * s_repr + weights[:, 1:] * f_repr
            interaction = self.interaction_mlp(
                torch.cat([s_repr, f_repr, torch.abs(s_repr - f_repr), s_repr * f_repr], dim=1)
            )
            logits = self.final_head(gated + interaction)
        if not return_aux:
            return logits

        balance, entropy = self._router_regularization(weights)
        return {
            "logits": logits,
            "spatial_logits": spatial_logits,
            "fire_logits_2c": fire_logits_2c,
            "router_weights": weights,
            "router_balance": balance,
            "router_entropy": entropy,
            "spatial_proj": F.normalize(self.proj_spatial(s_repr), dim=1),
            "fire_proj": F.normalize(self.proj_fire(f_repr), dim=1),
        }
