# -*- coding: utf-8 -*-
"""
Visualize a spatialized approximation of the REAL internal fusion process
for the final dual-stream model.

What it shows
-------------
This script visualizes the stage:
    token interaction  ->  token maps  ->  pooling later

Concretely:
1) extract spatial / FIRE feature maps
2) project both maps to the shared fuse_dim
3) apply anomaly-guided cross-attention (the real fusion_block in code)
4) reshape post-interaction tokens back to 2D maps
5) upsample the post-interaction maps to input resolution
6) build:
   - post-interaction spatial token map
   - post-interaction reconstruction token map
   - router-guided fused token map

Usage (PowerShell):
python .\tools\vis_token_fusion_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02100236\ILSVRC2012_val_00016444.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_path ".\results\vis\token_fusion_one.png"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

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
import matplotlib.pyplot as plt

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


def make_overlay(base_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.42) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap_01, 0.0, 1.0))[..., :3]
    overlay = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def token_map_to_2d(tokens: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    # tokens: [B, N, C] -> [B, C, H, W]
    b, n, c = tokens.shape
    h, w = hw
    if n != h * w:
        raise ValueError(f"Token count mismatch: N={n}, HxW={h*w}")
    return tokens.transpose(1, 2).reshape(b, c, h, w)


def reduce_and_upsample_map(feat_map: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    # feat_map: [B,C,H,W] -> [H_out, W_out]
    m = feat_map.abs().mean(dim=1, keepdim=True)
    m = F.interpolate(m, size=out_hw, mode="bilinear", align_corners=False)
    return m[0, 0]


def extract_token_fusion_maps(model, x: torch.Tensor):
    # ---- same real internal path as model.forward(), up to token interaction ----
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

    # ---- real anomaly-guided cross-attention block ----
    s_tokens_post, f_tokens_post = model.fusion_block(s_tokens, f_tokens, prior_flat)

    # spatialize post-interaction token maps
    s_map_post = token_map_to_2d(s_tokens_post, target_hw)
    f_map_post = token_map_to_2d(f_tokens_post, target_hw)

    # IMPORTANT: upsample from token-map resolution (e.g. 8x8) to input resolution (e.g. 256x256)
    out_hw = x.shape[-2:]
    M_s_post = reduce_and_upsample_map(s_map_post, out_hw)
    M_f_post = reduce_and_upsample_map(f_map_post, out_hw)

    # ---- continue the real path to get router weights ----
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

    logits = model.final_head(
        ws * s_repr + wr * f_repr +
        model.interaction_mlp(torch.cat([s_repr, f_repr, torch.abs(s_repr - f_repr), s_repr * f_repr], dim=1))
    )
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(torch.argmax(logits, dim=1)[0].item())

    M_fused_post = ws * M_s_post + wr * M_f_post

    return {
        "M_s_post": robust_normalize(M_s_post),
        "M_f_post": robust_normalize(M_f_post),
        "M_fused_post": robust_normalize(M_fused_post),
        "ws": ws,
        "wr": wr,
        "prob_fake": prob_fake,
        "pred": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="sfire_crossattn_resnet50")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model_name, device=device)

    pil = Image.open(image_path).convert("RGB")
    tf = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    x = tf(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = extract_token_fusion_maps(model, x)

    input_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()
    input_rgb = np.clip(input_rgb, 0.0, 1.0)

    spatial_overlay = make_overlay(input_rgb, out["M_s_post"], alpha=0.42)
    recon_overlay = make_overlay(input_rgb, out["M_f_post"], alpha=0.42)
    fused_overlay = make_overlay(input_rgb, out["M_fused_post"], alpha=0.42)

    fig = plt.figure(figsize=(16, 4))
    axs = [plt.subplot(1, 4, i + 1) for i in range(4)]

    axs[0].imshow(input_rgb)
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(spatial_overlay)
    axs[1].set_title("Post-interaction spatial map")
    axs[1].axis("off")

    axs[2].imshow(recon_overlay)
    axs[2].set_title("Post-interaction recon map")
    axs[2].axis("off")

    axs[3].imshow(fused_overlay)
    axs[3].set_title(
        f"Post-interaction fused map\n$w_s$={out['ws']:.2f}, $w_r$={out['wr']:.2f}\n"
        f"p(fake)={out['prob_fake']:.4f}, pred={'fake' if out['pred'] == 1 else 'real'}"
    )
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved to: {out_path}")


if __name__ == "__main__":
    main()
