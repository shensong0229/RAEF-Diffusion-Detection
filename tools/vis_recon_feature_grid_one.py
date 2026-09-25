# -*- coding: utf-8 -*-
"""
Visualize reconstruction-aware feature tensors for one image.

This script targets the FIRE-based reconstruction branch inside the final
sfire_crossattn_resnet50 model. It extracts fire_aux["feature_map"] and creates
three user-friendly outputs:

1) mean-feature heatmap overlay on the input image
2) top-channel feature maps overview
3) stacked grid icon similar to the schematic tensor block used in the paper

Outputs
-------
- 01_input.png
- 02_recon_feature_overlay.png
- 03_recon_feature_channels.png
- 04_recon_feature_grid_stack.png
- recon_feature_summary.png

Usage (PowerShell)
------------------
cd F:\FakeImageDetect
$env:PYTHONPATH = (Get-Location).Path

python .\tools\vis_recon_feature_grid_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02487347\ILSVRC2012_val_00022040.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_dir ".\results\vis\recon_feature_one"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

# avoid stale cache/tmp paths on a missing drive like E:\
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
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

import torch
import torch.nn.functional as F

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


def normalize_map(x: torch.Tensor, q_low: float = 0.02, q_high: float = 0.995) -> np.ndarray:
    x = x.detach().float().cpu()
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    denom = max(float(hi - lo), 1e-8)
    x = ((x - lo) / denom).clamp(0.0, 1.0)
    return x.numpy()


def make_overlay(base_rgb: np.ndarray, map_01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = plt.get_cmap("viridis")(map_01)[..., :3]
    overlay = (1.0 - alpha) * base_rgb + alpha * heat
    return np.clip(overlay, 0.0, 1.0)


def extract_recon_feature_map(model, x: torch.Tensor) -> Tuple[torch.Tensor, float, int]:
    if not hasattr(model, "fire_branch"):
        raise ValueError("This script expects the final dual-stream model with a FIRE-based branch.")

    fire_aux = model.fire_branch.extract_features(x, return_aux=True)
    feature_map = fire_aux["feature_map"]  # [B, C, H, W]

    logits = model(x)
    prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
    pred = int(torch.argmax(logits, dim=1)[0].item())
    return feature_map[0], prob_fake, pred


def select_top_channels(feat_map: torch.Tensor, topk: int = 3) -> List[int]:
    # rank channels by mean absolute activation
    score = feat_map.abs().mean(dim=(1, 2))
    topk = max(1, min(int(topk), int(score.numel())))
    idx = torch.topk(score, k=topk, largest=True).indices.tolist()
    return [int(i) for i in idx]


def save_image(path: Path, arr: np.ndarray, cmap: str | None = None, title: str | None = None, dpi: int = 220):
    fig = plt.figure(figsize=(4, 4))
    ax = plt.gca()
    if cmap is None:
        ax.imshow(arr)
    else:
        ax.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    if title:
        ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def draw_grid_stack_icon(path: Path, grid_maps: List[np.ndarray], dpi: int = 240):
    # Create a schematic stacked tensor icon using real feature grids.
    fig = plt.figure(figsize=(4.2, 3.4))
    ax = plt.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    offsets = [(1.2, 1.0), (2.2, 1.8), (3.2, 2.6)]
    grid_w = 5.4
    grid_h = 5.4

    # back-to-front for better depth feeling
    for i, (g, (ox, oy)) in enumerate(zip(grid_maps, offsets)):
        ax.imshow(g, extent=[ox, ox + grid_w, oy, oy + grid_h], origin="lower", cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
        # grid lines
        gh, gw = g.shape
        for c in range(gw + 1):
            x = ox + grid_w * c / gw
            ax.plot([x, x], [oy, oy + grid_h], color="black", lw=0.8, alpha=0.35)
        for r in range(gh + 1):
            y = oy + grid_h * r / gh
            ax.plot([ox, ox + grid_w], [y, y], color="black", lw=0.8, alpha=0.35)
        ax.add_patch(Rectangle((ox, oy), grid_w, grid_h, fill=False, edgecolor="black", linewidth=1.2, alpha=0.45))

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)


def save_channel_overview(path: Path, channels: List[np.ndarray], channel_ids: List[int], dpi: int = 220):
    n = len(channels)
    fig = plt.figure(figsize=(4 * n, 4))
    for i, (m, cid) in enumerate(zip(channels, channel_ids), start=1):
        ax = plt.subplot(1, n, i)
        ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"ch {cid}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def save_summary(path: Path, input_rgb: np.ndarray, overlay: np.ndarray, channels: List[np.ndarray], channel_ids: List[int], grid_icon: np.ndarray | None = None, dpi: int = 220):
    n = len(channels)
    fig = plt.figure(figsize=(4 * (n + 2), 4.2))

    ax = plt.subplot(1, n + 2, 1)
    ax.imshow(input_rgb)
    ax.set_title("Input")
    ax.axis("off")

    ax = plt.subplot(1, n + 2, 2)
    ax.imshow(overlay)
    ax.set_title("Recon feature overlay")
    ax.axis("off")

    for i, (m, cid) in enumerate(zip(channels, channel_ids), start=3):
        ax = plt.subplot(1, n + 2, i)
        ax.imshow(m, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"Top ch {cid}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="sfire_crossattn_resnet50")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--grid_size", type=int, default=8)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    stages = tqdm(total=5, desc="vis recon feature", ncols=100, ascii=True)

    model = load_model(args.ckpt, args.model_name, device=device)
    stages.update(1)

    pil = Image.open(image_path).convert("RGB")
    tf = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    x = tf(pil).unsqueeze(0).to(device)
    input_rgb = x[0].detach().cpu().permute(1, 2, 0).numpy()
    input_rgb = np.clip(input_rgb, 0.0, 1.0)
    stages.update(1)

    with torch.inference_mode():
        feat_map, prob_fake, pred = extract_recon_feature_map(model, x)
    stages.update(1)

    mean_map = feat_map.abs().mean(dim=0, keepdim=False)
    mean_map = F.interpolate(mean_map[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    mean_map_01 = normalize_map(mean_map)
    overlay = make_overlay(input_rgb, mean_map_01, alpha=0.45)

    top_ids = select_top_channels(feat_map, topk=args.topk)
    channel_maps = []
    grid_maps = []
    for cid in top_ids:
        ch = feat_map[cid]
        ch_up = F.interpolate(ch[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        channel_maps.append(normalize_map(ch_up))

        g = F.interpolate(ch[None, None], size=(args.grid_size, args.grid_size), mode="bilinear", align_corners=False)[0, 0]
        grid_maps.append(normalize_map(g))
    stages.update(1)

    save_image(out_dir / "01_input.png", input_rgb, cmap=None, title="Input")
    save_image(
        out_dir / "02_recon_feature_overlay.png",
        overlay,
        cmap=None,
        title=f"Recon feature overlay | pred={'fake' if pred == 1 else 'real'} | p(fake)={prob_fake:.4f}",
    )
    save_channel_overview(out_dir / "03_recon_feature_channels.png", channel_maps, top_ids)
    draw_grid_stack_icon(out_dir / "04_recon_feature_grid_stack.png", grid_maps)
    save_summary(out_dir / "recon_feature_summary.png", input_rgb, overlay, channel_maps, top_ids)
    stages.update(1)
    stages.close()

    print(f"[OK] saved to: {out_dir}")
    print(f"[INFO] pred={'fake' if pred == 1 else 'real'} | p(fake)={prob_fake:.6f}")
    print(f"[INFO] top channels: {top_ids}")


if __name__ == "__main__":
    main()
