# -*- coding: utf-8 -*-
"""
Save the final 4-column visualization as FOUR separate image files (no titles/text).

Outputs
-------
The script saves 4 standalone PNGs into --out_dir:
    01_input.png
    02_spatial_response.png
    03_recon_diff_a3.png
    04_post_interaction_fused.png

Design choice
-------------
This script uses the final visualization scheme you confirmed:
1) Input
2) Spatial response           (the clearer pre-interaction spatial overlay)
3) |Δ_raw - Δ_filtered|       (A3 reconstruction-aware difference map, grayscale)
4) Post-interaction fused map (closer to the real internal fusion process)

Usage (PowerShell):
python .\tools\vis_four_columns_separate.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02100236\ILSVRC2012_val_00016444.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_dir ".\results\vis\sample_real_001"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Avoid stale cache/tmp paths on a missing drive like E:\
os.environ.setdefault("XDG_CACHE_HOME", r"F:\FakeImageDetect\.cache")
os.environ.setdefault("HF_HOME", r"F:\FakeImageDetect\.cache\hf")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", r"F:\FakeImageDetect\.cache\hf\hub")
os.environ.setdefault("HF_HUB_CACHE", r"F:\FakeImageDetect\.cache\hf\hub")
os.environ.setdefault("TRANSFORMERS_CACHE", r"F:\FakeImageDetect\.cache\hf\transformers")
os.environ.setdefault("TORCH_HOME", r"F:\FakeImageDetect\.cache\torch")
os.environ.setdefault("TIMM_HOME", r"F:\FakeImageDetect\.cache\timm")
os.environ.setdefault("MPLCONFIGDIR", r"F:\FakeImageDetect\.cache\matplotlib")
os.environ.setdefault("TEMP", r"F:\FakeImageDetect\.tmp")
os.environ.setdefault("TMP", r"F:\FakeImageDetect\.tmp")
os.environ.setdefault("FIRE_VAE_DIR", r"F:\FakeImageDetect\pretrained\sd15_vae")

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import build_model
from core.dataset.transforms import build_transforms_for_model


def strip_wrappers(state_dict):
    out = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out


def load_model(ckpt_path: str, model_name: str, device: str):
    model = build_model(model_name, num_classes=2, pretrained=False, dropout=0.0).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    state = strip_wrappers(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def robust_normalize(x: torch.Tensor, q_low: float = 0.02, q_high: float = 0.995) -> np.ndarray:
    x = x.detach().float().cpu()
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    x = (x - lo) / max(float(hi - lo), 1e-8)
    x = x.clamp(0.0, 1.0)
    return x.numpy()


def to_uint8_rgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).round().astype(np.uint8)


def apply_cmap(x01: np.ndarray, cmap_name: str = "jet") -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    y = cmap(np.clip(x01, 0.0, 1.0))[..., :3]
    return y


def make_overlay(base_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    heat_rgb = apply_cmap(heatmap_01, cmap_name="jet")
    out = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(out, 0.0, 1.0)


def make_grayscale_rgb(x01: np.ndarray) -> np.ndarray:
    gray = np.clip(x01, 0.0, 1.0)
    return np.repeat(gray[..., None], 3, axis=2)


def extract_pre_spatial_map(model, x: torch.Tensor) -> torch.Tensor:
    # clearer pre-interaction spatial response, as you preferred
    if hasattr(model, "_extract_spatial_features"):
        fmap, _pooled, _logits = model._extract_spatial_features(x)
    elif hasattr(model, "extract_features"):
        out = model.extract_features(x, return_logits=True)
        fmap = out["feature_map"]
    else:
        raise AttributeError("Model does not expose spatial feature extraction interface.")

    if fmap.ndim != 4:
        raise ValueError(f"Expected 4D spatial feature map, got {tuple(fmap.shape)}")

    resp = fmap.abs().mean(dim=1, keepdim=True)
    resp = F.interpolate(resp, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return resp[0, 0]


def extract_recon_diff_a3_map(model, x: torch.Tensor) -> torch.Tensor:
    # A3: |Δ_raw - Δ_filtered|
    if not hasattr(model, "fire_branch"):
        raise ValueError("This script expects the final dual-stream model with a FIRE-based branch.")
    fire_aux = model.fire_branch.extract_features(x, return_aux=True)
    raw_delta = fire_aux["raw_reconstructions_delta"]
    filtered_delta = fire_aux["filtered_reconstructions_delta"]
    diff = torch.abs(raw_delta - filtered_delta).mean(dim=1, keepdim=True)
    diff = F.interpolate(diff, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return diff[0, 0]


def token_map_to_2d(tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    # tokens: [B, N, C] -> [B, C, H, W]
    b, n, c = tokens.shape
    h, w = hw
    if n != h * w:
        raise ValueError(f"Token count mismatch: N={n}, HxW={h*w}")
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def reduce_and_upsample_map(feat_map: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    m = feat_map.abs().mean(dim=1, keepdim=True)
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    return m[0, 0]


def extract_post_interaction_fused_map(model, x: torch.Tensor) -> torch.Tensor:
    # closer to the real internal fusion process
    s_map, s_vec, s_logits = model._extract_spatial_features(x)
    fire_aux = model.fire_branch.extract_features(x, return_aux=True)

    f_map = fire_aux["feature_map"]
    f_vec = fire_aux["global_feature"]
    f_logit = fire_aux["logits"].view(-1)
    prior = fire_aux["anomaly_prior"]

    s_map_proj = model.spatial_map_proj(s_map)
    f_map_proj = model.fire_map_proj(f_map)

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

    s_tokens_post, f_tokens_post = model.fusion_block(s_tokens, f_tokens, prior_flat)

    s_map_post = token_map_to_2d(s_tokens_post, target_hw)
    f_map_post = token_map_to_2d(f_tokens_post, target_hw)

    out_hw = x.shape[-2:]
    M_s_post = reduce_and_upsample_map(s_map_post, out_hw)
    M_f_post = reduce_and_upsample_map(f_map_post, out_hw)

    s_local = s_tokens_post.mean(dim=1)
    f_local = f_tokens_post.mean(dim=1)

    s_global = model.spatial_vec_proj(s_vec)
    f_global = model.fire_vec_proj(f_vec)
    s_repr = s_global + s_local
    f_repr = f_global + f_local

    prior_mean = prior_small.mean(dim=(2, 3))
    prior_std = prior_small.flatten(2).std(dim=2)
    prior_std = prior_std.unsqueeze(1) if prior_std.ndim == 1 else prior_std

    router_weights = model.router(
        spatial_feat=s_repr,
        fire_feat=f_repr,
        spatial_logits=s_logits,
        fire_logit=f_logit,
        prior_mean=prior_mean,
        prior_std=prior_std,
    )
    ws = float(router_weights[0, 0].item())
    wr = float(router_weights[0, 1].item())

    M_fused_post = ws * M_s_post + wr * M_f_post
    return M_fused_post


def save_rgb(arr01: np.ndarray, path: Path):
    img = Image.fromarray(to_uint8_rgb(arr01))
    img.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="sfire_crossattn_resnet50")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model_name, device=device)

    pil = Image.open(image_path).convert("RGB")
    tf = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    x = tf(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        pre_spatial_map = extract_pre_spatial_map(model, x)
        recon_diff_map = extract_recon_diff_a3_map(model, x)
        fused_post_map = extract_post_interaction_fused_map(model, x)

    input_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()
    input_rgb = np.clip(input_rgb, 0.0, 1.0)

    pre_spatial_np = robust_normalize(pre_spatial_map)
    recon_diff_np = robust_normalize(recon_diff_map)
    fused_post_np = robust_normalize(fused_post_map)

    spatial_overlay = make_overlay(input_rgb, pre_spatial_np, alpha=0.42)
    recon_gray_rgb = make_grayscale_rgb(recon_diff_np)
    fused_overlay = make_overlay(input_rgb, fused_post_np, alpha=0.42)

    save_rgb(input_rgb, out_dir / "01_input.png")
    save_rgb(spatial_overlay, out_dir / "02_spatial_response.png")
    save_rgb(recon_gray_rgb, out_dir / "03_recon_diff_a3.png")
    save_rgb(fused_overlay, out_dir / "04_post_interaction_fused.png")

    print(f"[OK] saved 4 images to: {out_dir}")
    print("      01_input.png")
    print("      02_spatial_response.png")
    print("      03_recon_diff_a3.png")
    print("      04_post_interaction_fused.png")


if __name__ == "__main__":
    main()
