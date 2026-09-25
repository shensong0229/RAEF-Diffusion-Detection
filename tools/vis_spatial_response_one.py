# -*- coding: utf-8 -*-
"""
Visualize the spatial-branch response for one image.

Usage (PowerShell):
python .\tools\vis_spatial_response_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02100236\ILSVRC2012_val_00016444.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_path ".\results\vis\spatial_response_one.png"

Notes
-----
1) Default usage targets the final Ours model (sfire_crossattn_resnet50).
2) The response map here is a spatial activation map from the spatial branch
   feature map, computed as mean(abs(feature_map), dim=channel), then normalized.
3) The visualization is produced on the model input view after resize/crop so the
   heatmap aligns with what the model actually sees.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def make_overlay(base_rgb: np.ndarray, heatmap_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = cm.get_cmap("jet")
    heat_rgb = cmap(np.clip(heatmap_01, 0.0, 1.0))[..., :3]
    overlay = (1.0 - alpha) * base_rgb + alpha * heat_rgb
    return np.clip(overlay, 0.0, 1.0)


def extract_spatial_response(model, x: torch.Tensor):
    if hasattr(model, "_extract_spatial_features"):
        fmap, _pooled, logits = model._extract_spatial_features(x)
    elif hasattr(model, "extract_features"):
        out = model.extract_features(x, return_logits=True)
        fmap = out["feature_map"]
        logits = out["logits"]
    else:
        raise AttributeError("Model does not expose spatial feature extraction interface.")

    if fmap.ndim != 4:
        raise ValueError(f"Expected 4D feature map, got shape={tuple(fmap.shape)}")

    resp = fmap.abs().mean(dim=1, keepdim=True)
    resp = F.interpolate(resp, size=x.shape[-2:], mode="bilinear", align_corners=False)
    resp = resp[0, 0]

    resp = resp - resp.min()
    denom = resp.max().clamp_min(1e-8)
    resp = resp / denom

    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(torch.argmax(logits, dim=1)[0].item())
    return resp.detach().cpu().numpy(), prob_fake, pred


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
        resp, prob_fake, pred = extract_spatial_response(model, x)

    input_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()
    input_rgb = np.clip(input_rgb, 0.0, 1.0)
    overlay = make_overlay(input_rgb, resp, alpha=0.45)

    fig = plt.figure(figsize=(12, 4))
    axs = [plt.subplot(1, 3, i + 1) for i in range(3)]

    axs[0].imshow(input_rgb)
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(resp, cmap="jet")
    axs[1].set_title("Spatial response")
    axs[1].axis("off")

    axs[2].imshow(overlay)
    axs[2].set_title(f"Overlay | pred={'fake' if pred == 1 else 'real'} | p(fake)={prob_fake:.4f}")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved to: {out_path}")


if __name__ == "__main__":
    main()
