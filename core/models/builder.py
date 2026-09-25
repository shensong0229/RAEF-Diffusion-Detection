# -*- coding: utf-8 -*-
"""
Model builder.

Supports:
- spatial_resnet18 / spatial_resnet50 / spatial_timm:<backbone>
- freq_resnet18 / freq_resnet50            (FFT log-magnitude as input, 1-channel)
- sf_concat_resnet50 / sf_concat_resnet18  (dual-stream: spatial+freq, concat fusion)
- sf_gate_resnet50 / sf_gate_resnet18      (dual-stream: spatial+freq, gated fusion)

New (reconstruction-error branch):
- re_resnet18 / re_resnet50                (residual-only, expects 3-channel residual map)
- sr_concat_resnet18 / sr_concat_resnet50  (dual-stream: spatial+residual, concat fusion)
- sr_gate_resnet18 / sr_gate_resnet50      (dual-stream: spatial+residual, gated fusion)

Design choices
--------------
1) Keep the old S-only / F-only / SF behavior unchanged.
2) Add a new reconstruction-error branch without touching the current spatial branch.
3) For SR models, the spatial stream still consumes RAW RGB in [0,1] and applies
   ImageNet normalization internally; the residual stream consumes a precomputed
   3-channel residual map in [0,1].
4) For SR-gate, use a stronger symmetric dual channel-wise gating module.
5) For residual branches, add dedicated residual-map normalization to reduce
   cross-generator scale drift.
   channel-wise gating module:
   - branch-wise channel recalibration
   - two independent channel gates (for spatial / residual)
   - normalized fusion without assuming either branch must dominate
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import timm

from .spatial import SpatialClassifier
from .fire_official import FIREOfficialModel
from .spatial_fire_dualstream import SpatialFIREDualStreamModel


# -----------------------------
# helpers: model type checks
# -----------------------------
def is_freq_model(name: str) -> bool:
    n = (name or "").lower().strip()
    return n.startswith("freq_")


def is_dual_model(name: str) -> bool:
    n = (name or "").lower().strip()
    return n.startswith("sf_")


def is_re_model(name: str) -> bool:
    n = (name or "").lower().strip()
    return n.startswith("re_")


def is_sr_model(name: str) -> bool:
    n = (name or "").lower().strip()
    return n.startswith("sr_")


def is_residual_model(name: str) -> bool:
    return is_re_model(name) or is_sr_model(name)


# -----------------------------
# frequency transform module (inside model for legacy SF branch)
# -----------------------------
class _FFTLogMag(nn.Module):
    """
    Compute log(1+|FFT|) magnitude (shifted) from raw RGB image tensor in [0,1].
    Output: standardized 1-channel tensor (B,1,H,W).
    """
    def __init__(self, eps: float = 1e-6, clamp: float = 5.0):
        super().__init__()
        self.eps = float(eps)
        self.clamp = float(clamp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4 or x.size(1) != 3:
            raise ValueError(f"FFTLogMag expects (B,3,H,W), got {tuple(x.shape)}")

        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        fft = torch.fft.fft2(gray, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft)
        feat = torch.log1p(mag)

        mean = feat.mean(dim=(-2, -1), keepdim=True)
        std = feat.std(dim=(-2, -1), keepdim=True)
        feat = (feat - mean) / (std + self.eps)

        if self.clamp > 0:
            feat = feat.clamp(-self.clamp, self.clamp)
        return feat


class _ImageNetNorm(nn.Module):
    """Apply ImageNet normalization to raw tensor in [0,1]."""
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _ResidualMapNorm(nn.Module):
    """
    Residual-map normalization for reconstruction-error inputs.

    Goals
    -----
    1) Compress very small / very large residual values with a gentle log mapping.
    2) Standardize each sample per channel to reduce domain-specific scale drift.
    3) Clamp the result to keep the residual branch numerically stable.

    This is especially helpful when residual maps come from different generators and
    the absolute reconstruction error scale shifts a lot across domains.
    """
    def __init__(self, log_gain: float = 8.0, eps: float = 1e-6, clamp: float = 3.0):
        super().__init__()
        self.log_gain = float(log_gain)
        self.eps = float(eps)
        self.clamp = float(clamp)
        self._log_denom = math.log1p(self.log_gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        if self.log_gain > 0:
            x = torch.log1p(self.log_gain * x) / self._log_denom

        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True)
        x = (x - mean) / (std + self.eps)

        if self.clamp > 0:
            x = x.clamp(-self.clamp, self.clamp)
        return x


class _ChannelRecalibration(nn.Module):
    """
    Lightweight channel-wise recalibration for pooled feature vectors.
    Input / output shape: (B, C)
    """
    def __init__(self, dim: int, reduction: int = 16, min_hidden: int = 32):
        super().__init__()
        dim = int(dim)
        hidden = max(int(dim // max(1, reduction)), int(min_hidden))
        self.gate = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gate(x)


class _DualIndependentChannelGate(nn.Module):
    """
    Symmetric dual independent channel-wise gate (stable version).

    Given fs, fr in R^{B x C}, build an interaction descriptor:
        z = [fs, fr, |fs-fr|]
    then predict two independent channel gates gs, gr in [0,1]^C,
    and perform normalized fusion:
        fused = gs*fs + gr*fr,
        where gs, gr are normalized from their sigmoid activations.

    Stability notes
    ---------------
    1) We intentionally remove the multiplicative interaction term fs*fr, which
       is much more likely to explode under mixed precision / large feature scale.
    2) Gate computation is carried out in float32 to reduce NaN/Inf risk.
    """
    def __init__(self, dim: int, hidden_dim: int = 1024, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        dim = int(dim)
        hidden_dim = int(hidden_dim)
        self.eps = float(eps)

        self.trunk = nn.Sequential(
            nn.Linear(dim * 3, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
        )
        self.gate_s = nn.Linear(hidden_dim, dim)
        self.gate_r = nn.Linear(hidden_dim, dim)
        self.sigmoid = nn.Sigmoid()

        self.last_gate_s = None
        self.last_gate_r = None

    def forward(self, fs: torch.Tensor, fr: torch.Tensor):
        if fs.shape != fr.shape:
            raise ValueError(f"DualIndependentChannelGate expects equal shapes, got {tuple(fs.shape)} vs {tuple(fr.shape)}")

        # Run gate branch in float32 for stability.
        fs32 = fs.float()
        fr32 = fr.float()

        z = torch.cat([fs32, fr32, torch.abs(fs32 - fr32)], dim=1)
        h = self.trunk(z)

        gs_raw = self.sigmoid(self.gate_s(h))
        gr_raw = self.sigmoid(self.gate_r(h))

        denom = gs_raw + gr_raw + self.eps
        gs = gs_raw / denom
        gr = gr_raw / denom

        self.last_gate_s = gs.detach()
        self.last_gate_r = gr.detach()

        fused = gs * fs32 + gr * fr32
        return fused.to(dtype=fs.dtype), gs, gr


def _expect_pair(x, branch_name: str):
    if not isinstance(x, (tuple, list)) or len(x) != 2:
        raise ValueError(
            f"{branch_name} expects a paired input (rgb_tensor, residual_tensor), "
            f"got type={type(x)}"
        )
    return x[0], x[1]


# -----------------------------
# F-only model (legacy)
# -----------------------------
class FreqClassifier(nn.Module):
    """
    Frequency-only classifier. Expects 1-channel input already in frequency space,
    e.g., from dataset transforms (FFT log-mag standardized).
    """
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = False, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone

        self.feat = timm.create_model(backbone, pretrained=pretrained, in_chans=1, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.feat, "num_features", None)
        if feat_dim is None:
            feat_dim = 2048

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(int(feat_dim), int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.feat(x)
        f = self.dropout(f)
        return self.head(f)


# -----------------------------
# Reconstruction-error only model
# -----------------------------
class ReconErrorClassifier(nn.Module):
    """
    Residual-only classifier. Expects:
      - either a residual tensor (B,3,H,W)
      - or a paired input (rgb, residual), in which case only residual is used.
    """
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = False, dropout: float = 0.0):
        super().__init__()
        self.backbone_name = backbone
        self.r_norm = _ResidualMapNorm()
        self.feat = timm.create_model(backbone, pretrained=pretrained, in_chans=3, num_classes=0, global_pool="avg")
        feat_dim = getattr(self.feat, "num_features", 2048)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.head = nn.Linear(int(feat_dim), int(num_classes))

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            _rgb, x = _expect_pair(x, "ReconErrorClassifier")
        x = self.r_norm(x)
        f = self.feat(x)
        f = self.dropout(f)
        return self.head(f)


# -----------------------------
# Spatial + Frequency dual-stream (legacy)
# -----------------------------
class DualStreamClassifier(nn.Module):
    """
    Spatial + Frequency dual-stream.

    Inputs: raw RGB tensor in [0,1], shape (B,3,H,W)
    - spatial stream: ImageNet norm -> timm backbone -> pooled feature
    - freq stream: FFT log-mag -> timm backbone(in_chans=1) -> pooled feature
    Fusion:
      - "concat": concat features -> MLP -> logits
      - "gate":  learn alpha from concat -> fused = alpha*fs + (1-alpha)*ff -> logits
    """
    def __init__(
        self,
        spatial_backbone: str = "resnet50",
        freq_backbone: str = "resnet50",
        num_classes: int = 2,
        spatial_pretrained: bool = False,
        freq_pretrained: bool = False,
        dropout: float = 0.2,
        fusion: str = "concat",
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.fusion = fusion.lower().strip()
        if self.fusion not in ["concat", "gate"]:
            raise ValueError(f"Unsupported fusion={fusion}; choose concat/gate")

        self.norm = _ImageNetNorm()
        self.fft = _FFTLogMag()

        self.s_feat = timm.create_model(spatial_backbone, pretrained=spatial_pretrained, num_classes=0, global_pool="avg")
        self.f_feat = timm.create_model(freq_backbone, pretrained=freq_pretrained, in_chans=1, num_classes=0, global_pool="avg")

        s_dim = getattr(self.s_feat, "num_features", 2048)
        f_dim = getattr(self.f_feat, "num_features", 2048)

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        if self.fusion == "concat":
            in_dim = int(s_dim + f_dim)
            hd = int(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hd),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(hd, int(num_classes)),
            )
        else:
            in_dim = int(s_dim + f_dim)
            hd = int(hidden_dim)
            self.gate = nn.Sequential(
                nn.Linear(in_dim, hd),
                nn.ReLU(inplace=True),
                nn.Linear(hd, 1),
                nn.Sigmoid(),
            )
            self.head = nn.Linear(int(s_dim), int(num_classes))

        self.s_dim = int(s_dim)
        self.f_dim = int(f_dim)

        if self.fusion == "gate" and self.s_dim != self.f_dim:
            self.proj_f = nn.Linear(self.f_dim, self.s_dim)
        else:
            self.proj_f = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.norm(x)
        fs = self.s_feat(xs)

        xf = self.fft(x)
        ff = self.f_feat(xf)

        fs = self.dropout(fs)
        ff = self.dropout(ff)

        if self.fusion == "concat":
            z = torch.cat([fs, ff], dim=1)
            return self.mlp(z)

        if self.proj_f is not None:
            ff = self.proj_f(ff)

        z = torch.cat([fs, ff], dim=1)
        alpha = self.gate(z)
        fused = alpha * fs + (1 - alpha) * ff
        return self.head(fused)


# -----------------------------
# Spatial + Reconstruction-error dual-stream
# -----------------------------
class SpatialResidualDualClassifier(nn.Module):
    """
    Spatial + Reconstruction-error dual-stream.

    Input:
      x = (rgb_tensor, residual_tensor)
      - rgb_tensor: (B,3,H,W), raw RGB in [0,1]
      - residual_tensor: (B,3,H,W), precomputed residual map in [0,1]

    Fusion:
      - "concat": concat features -> MLP -> logits
      - "gate":   symmetric dual independent channel-wise gating
                    (no fixed dominance assumption)
    """
    def __init__(
        self,
        spatial_backbone: str = "resnet50",
        residual_backbone: str = "resnet50",
        num_classes: int = 2,
        spatial_pretrained: bool = False,
        residual_pretrained: bool = False,
        dropout: float = 0.2,
        fusion: str = "concat",
        hidden_dim: int = 1024,
    ):
        super().__init__()
        self.fusion = fusion.lower().strip()
        if self.fusion not in ["concat", "gate"]:
            raise ValueError(f"Unsupported fusion={fusion}; choose concat/gate")

        self.norm = _ImageNetNorm()
        self.r_in_norm = _ResidualMapNorm()
        self.s_feat = timm.create_model(spatial_backbone, pretrained=spatial_pretrained, num_classes=0, global_pool="avg")
        self.r_feat = timm.create_model(residual_backbone, pretrained=residual_pretrained, in_chans=3, num_classes=0, global_pool="avg")

        s_dim = getattr(self.s_feat, "num_features", 2048)
        r_dim = getattr(self.r_feat, "num_features", 2048)

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.s_dim = int(s_dim)
        self.r_dim = int(r_dim)

        if self.fusion == "concat":
            in_dim = int(self.s_dim + self.r_dim)
            hd = int(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hd),
                nn.ReLU(inplace=True),
                nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity(),
                nn.Linear(hd, int(num_classes)),
            )
            self.proj_r = None
            self.s_recalib = None
            self.r_recalib = None
            self.dual_gate = None
        else:
            if self.s_dim != self.r_dim:
                self.proj_r = nn.Linear(self.r_dim, self.s_dim)
                gate_dim = self.s_dim
            else:
                self.proj_r = None
                gate_dim = self.s_dim

            self.s_norm = nn.LayerNorm(gate_dim)
            self.r_norm = nn.LayerNorm(gate_dim)
            self.s_recalib = _ChannelRecalibration(gate_dim)
            self.r_recalib = _ChannelRecalibration(gate_dim)
            self.dual_gate = _DualIndependentChannelGate(
                dim=gate_dim,
                hidden_dim=int(hidden_dim),
                dropout=float(dropout),
            )
            self.head = nn.Linear(int(gate_dim), int(num_classes))

    def forward(self, x):
        rgb, residual = _expect_pair(x, "SpatialResidualDualClassifier")

        xs = self.norm(rgb)
        xr = self.r_in_norm(residual)
        fs = self.s_feat(xs)
        fr = self.r_feat(xr)

        fs = self.dropout(fs)
        fr = self.dropout(fr)

        if self.fusion == "concat":
            z = torch.cat([fs, fr], dim=1)
            return self.mlp(z)

        if self.proj_r is not None:
            fr = self.proj_r(fr)

        fs = self.s_norm(fs)
        fr = self.r_norm(fr)
        fs = self.s_recalib(fs)
        fr = self.r_recalib(fr)
        fused, _gs, _gr = self.dual_gate(fs, fr)
        return self.head(fused)


def build_model(name: str, **kwargs):
    """
    Build model by name.
    kwargs typically include: num_classes, pretrained, dropout
    For dual-stream you can pass:
      - spatial_pretrained / freq_pretrained / residual_pretrained
      - fusion ('concat' or 'gate')
    """
    name = (name or "").lower().strip()

    # S-only
    if name in ["spatial_resnet50", "s_only_resnet50"]:
        return SpatialClassifier(backbone="resnet50", **kwargs)

    if name in ["spatial_resnet18", "s_only_resnet18"]:
        return SpatialClassifier(backbone="resnet18", **kwargs)

    if name.startswith("spatial_timm:"):
        backbone = name.split(":", 1)[1]
        return SpatialClassifier(backbone=backbone, **kwargs)

    # F-only (legacy FFT branch)
    if name in ["freq_resnet50", "f_only_resnet50"]:
        return FreqClassifier(backbone="resnet50", **kwargs)

    if name in ["freq_resnet18", "f_only_resnet18"]:
        return FreqClassifier(backbone="resnet18", **kwargs)

    # RE-only (new reconstruction-error branch)
    if name in ["re_resnet50"]:
        return ReconErrorClassifier(backbone="resnet50", **kwargs)

    if name in ["re_resnet18"]:
        return ReconErrorClassifier(backbone="resnet18", **kwargs)

    # Dual-stream (legacy spatial + FFT)
    if name in ["sf_concat_resnet50", "sf_resnet50"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return DualStreamClassifier(
            spatial_backbone="resnet50",
            freq_backbone="resnet50",
            spatial_pretrained=pretrained,
            freq_pretrained=False,
            fusion="concat",
            **kwargs,
        )

    if name in ["sf_concat_resnet18", "sf_resnet18"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return DualStreamClassifier(
            spatial_backbone="resnet18",
            freq_backbone="resnet18",
            spatial_pretrained=pretrained,
            freq_pretrained=False,
            fusion="concat",
            hidden_dim=512,
            **kwargs,
        )

    if name in ["sf_gate_resnet50"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return DualStreamClassifier(
            spatial_backbone="resnet50",
            freq_backbone="resnet50",
            spatial_pretrained=pretrained,
            freq_pretrained=False,
            fusion="gate",
            **kwargs,
        )

    if name in ["sf_gate_resnet18"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return DualStreamClassifier(
            spatial_backbone="resnet18",
            freq_backbone="resnet18",
            spatial_pretrained=pretrained,
            freq_pretrained=False,
            fusion="gate",
            hidden_dim=512,
            **kwargs,
        )

    # Dual-stream (new spatial + residual)
    if name in ["sr_concat_resnet50"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return SpatialResidualDualClassifier(
            spatial_backbone="resnet50",
            residual_backbone="resnet50",
            spatial_pretrained=pretrained,
            residual_pretrained=False,
            fusion="concat",
            **kwargs,
        )

    if name in ["sr_concat_resnet18"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return SpatialResidualDualClassifier(
            spatial_backbone="resnet18",
            residual_backbone="resnet18",
            spatial_pretrained=pretrained,
            residual_pretrained=False,
            fusion="concat",
            hidden_dim=512,
            **kwargs,
        )

    if name in ["sr_gate_resnet50"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return SpatialResidualDualClassifier(
            spatial_backbone="resnet50",
            residual_backbone="resnet50",
            spatial_pretrained=pretrained,
            residual_pretrained=False,
            fusion="gate",
            **kwargs,
        )

    if name in ["sr_gate_resnet18"]:
        pretrained = bool(kwargs.pop("pretrained", False))
        return SpatialResidualDualClassifier(
            spatial_backbone="resnet18",
            residual_backbone="resnet18",
            spatial_pretrained=pretrained,
            residual_pretrained=False,
            fusion="gate",
            hidden_dim=512,
            **kwargs,
        )


    # Spatial + FIRE dual-stream and capacity-controlled ablation variants.
    sfire_variants = {
        "sfire_crossattn_resnet50": "full",
        "spatial_fire_dualstream_resnet50": "full",
        "sfire_ablation_average_resnet50": "average",
        "sfire_ablation_concat_linear_resnet50": "concat_linear",
        "sfire_ablation_gated_resnet50": "gated",
        "sfire_ablation_concat_mlp_matched_resnet50": "concat_mlp_matched",
        "sfire_ablation_standard_crossattn_resnet50": "standard_crossattn",
        "sfire_ablation_router_only_resnet50": "router_only",
        "sfire_ablation_no_routing_resnet50": "no_routing",
    }
    if name in sfire_variants:
        pretrained = bool(kwargs.pop("pretrained", False))
        return SpatialFIREDualStreamModel(
            spatial_backbone="resnet50",
            pretrained=pretrained,
            fusion_variant=sfire_variants[name],
            **kwargs,
        )

    # FIRE official single-branch
    if name in ["fire_resnet50"]:
        pretrained = bool(kwargs.pop("pretrained", True))
        kwargs.pop("num_classes", None)
        kwargs.pop("dropout", None)
        return FIREOfficialModel(mode="frq", norm_layer="instance", pretrained=pretrained, radiuslow=40, radiushigh=120)

    raise ValueError(f"Unknown model name: {name}")
