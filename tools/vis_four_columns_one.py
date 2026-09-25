# -*- coding: utf-8 -*-
"""
Build the final 4-column visualization for one image.

Columns
-------
1) Input
2) Spatial response
3) |Δ_raw - Δ_filtered|
4) Routing-guided fused response

Usage (PowerShell):
python .\tools\vis_four_columns_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02100236\ILSVRC2012_val_00016444.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_path ".\results\vis\four_columns_one.png"
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
from matplotlib import cm

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
    cmap = cm.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap_01, 0.0, 1.0))[..., :3]
    overlay = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def extract_spatial_map(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_extract_spatial_features"):
        fmap, _pooled, _logits = model._extract_spatial_features(x)
    elif hasattr(model, "extract_features"):
        out = model.extract_features(x, return_logits=True)
        fmap = out["feature_map"]
    else:
        raise AttributeError("Model does not expose spatial feature extraction interface.")
    resp = fmap.abs().mean(dim=1, keepdim=True)
    resp = F.interpolate(resp, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return resp[0, 0]


def extract_recon_map(model, x: torch.Tensor) -> torch.Tensor:
    fire_aux = model.fire_branch.extract_features(x, return_aux=True)
    raw_delta = fire_aux["raw_reconstructions_delta"]
    filtered_delta = fire_aux["filtered_reconstructions_delta"]
    diff = torch.abs(raw_delta - filtered_delta).mean(dim=1, keepdim=True)
    diff = F.interpolate(diff, size=x.shape[-2:], mode="bilinear", align_corners=False)
    return diff[0, 0]


def extract_router_weights(model, x: torch.Tensor):
    aux = model(x, return_aux=True)
    router = aux["router_weights"]
    ws = float(router[0, 0].item())
    wr = float(router[0, 1].item())
    return ws, wr


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
        spatial_map = extract_spatial_map(model, x)
        recon_map = extract_recon_map(model, x)
        ws, wr = extract_router_weights(model, x)

    fused_map = ws * spatial_map + wr * recon_map

    spatial_np = robust_normalize(spatial_map)
    recon_np = robust_normalize(recon_map)
    fused_np = robust_normalize(fused_map)

    input_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()
    input_rgb = np.clip(input_rgb, 0.0, 1.0)
    spatial_overlay = make_overlay(input_rgb, spatial_np, alpha=0.42)
    fused_overlay = make_overlay(input_rgb, fused_np, alpha=0.42)

    fig = plt.figure(figsize=(16, 4))
    axs = [plt.subplot(1, 4, i + 1) for i in range(4)]

    axs[0].imshow(input_rgb)
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(spatial_overlay)
    axs[1].set_title("Spatial response")
    axs[1].axis("off")

    axs[2].imshow(recon_np, cmap="gray", vmin=0.0, vmax=1.0)
    axs[2].set_title(r"|Δ_raw - Δ_filtered|")
    axs[2].axis("off")

    axs[3].imshow(fused_overlay)
    axs[3].set_title(f"Routing-guided fused response\n$w_s$={ws:.2f}, $w_r$={wr:.2f}")
    axs[3].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved to: {out_path}")


if __name__ == "__main__":
    main()
