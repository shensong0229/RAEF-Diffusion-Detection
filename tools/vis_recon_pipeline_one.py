# -*- coding: utf-8 -*-
"""
Visualize the full reconstruction-aware branch pipeline for one image.

This script is designed for the final Ours model path (sfire_crossattn_resnet50),
and makes the intermediate reconstruction branch states explicit for a single input:

    x
    -> x_f (middle filtered image)
    -> x_hat (reconstruction of x)
    -> x_hat_f (reconstruction of x_f)
    -> Delta_raw = |x_hat - x|
    -> Delta_filtered = |x_hat_f - x|
    -> |Delta_raw - Delta_filtered|

Outputs
-------
1) A single overview figure containing the whole pipeline.
2) Separate panel images for each stage.

Usage (PowerShell)
------------------
cd F:\FakeImageDetect
$env:PYTHONPATH = (Get-Location).Path

python .\tools\vis_recon_pipeline_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02487347\ILSVRC2012_val_00022040.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --out_dir ".\results\vis\recon_pipeline_one"
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict

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
from tqdm import tqdm

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import build_model
from core.dataset.transforms import build_transforms_for_model


def strip_wrappers(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def _to_rgb01(x: torch.Tensor, robust: bool = False) -> np.ndarray:
    """Convert a [3,H,W] tensor to displayable [H,W,3] in [0,1]."""
    x = x.detach().float().cpu()
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(x.shape)}")

    if robust:
        flat = x.flatten()
        lo = torch.quantile(flat, 0.01)
        hi = torch.quantile(flat, 0.99)
        x = (x - lo) / max(float(hi - lo), 1e-8)
    x = x.clamp(0.0, 1.0)
    return x.permute(1, 2, 0).numpy()


def _to_map01(x: torch.Tensor, q_low: float = 0.02, q_high: float = 0.995) -> np.ndarray:
    """Convert a [H,W] nonnegative map to displayable [0,1]."""
    x = x.detach().float().cpu()
    if x.ndim != 2:
        raise ValueError(f"Expected [H,W], got {tuple(x.shape)}")
    flat = x.flatten()
    lo = torch.quantile(flat, q_low)
    hi = torch.quantile(flat, q_high)
    x = (x - lo) / max(float(hi - lo), 1e-8)
    x = x.clamp(0.0, 1.0)
    return x.numpy()


def _mean_abs_map(x3: torch.Tensor) -> torch.Tensor:
    """[3,H,W] -> [H,W] by channel mean of absolute values."""
    if x3.ndim != 3 or x3.shape[0] != 3:
        raise ValueError(f"Expected [3,H,W], got {tuple(x3.shape)}")
    return x3.abs().mean(dim=0)


def make_overlay(base_rgb: np.ndarray, map01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    gray3 = np.repeat(map01[..., None], 3, axis=2)
    overlay = (1.0 - alpha) * base_rgb + alpha * gray3
    return np.clip(overlay, 0.0, 1.0)


def extract_pipeline(model, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "fire_branch"):
        raise ValueError("This script expects the final dual-stream model with a FIRE-based branch.")

    fire = model.fire_branch

    with torch.inference_mode():
        middle_filtered_image, _mask_mid_frq, _mask_mid_filterd, _ = fire.fft_filter_module(x)
        reconstructions_x = fire._reconstruct(x)
        reconstructions_middle_filtered = fire._reconstruct(middle_filtered_image)

        raw_delta = torch.abs(reconstructions_x - x)
        filtered_delta = torch.abs(reconstructions_middle_filtered - x)
        diff_map = torch.abs(raw_delta - filtered_delta).mean(dim=1, keepdim=True)

        logits = model(x)
        prob_fake = torch.softmax(logits, dim=1)[0, 1].item()
        pred = int(torch.argmax(logits, dim=1)[0].item())

    return {
        "input": x[0],
        "filtered": middle_filtered_image[0],
        "recon_x": reconstructions_x[0],
        "recon_filtered": reconstructions_middle_filtered[0],
        "delta_raw": raw_delta[0],
        "delta_filtered": filtered_delta[0],
        "diff_map": diff_map[0, 0],
        "prob_fake": prob_fake,
        "pred": pred,
    }


def save_panel_rgb(x: torch.Tensor, out_path: Path, robust: bool = False):
    arr = _to_rgb01(x, robust=robust)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_panel_map(x: torch.Tensor, out_path: Path, cmap: str = "gray"):
    arr = _to_map01(x)
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap=cmap, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_overview(pipeline: Dict[str, torch.Tensor], out_path: Path):
    input_rgb = _to_rgb01(pipeline["input"], robust=False)
    filtered_rgb = _to_rgb01(pipeline["filtered"], robust=True)
    recon_x_rgb = _to_rgb01(pipeline["recon_x"], robust=True)
    recon_filtered_rgb = _to_rgb01(pipeline["recon_filtered"], robust=True)

    delta_raw_map = _to_map01(_mean_abs_map(pipeline["delta_raw"]))
    delta_filtered_map = _to_map01(_mean_abs_map(pipeline["delta_filtered"]))
    diff_map = _to_map01(pipeline["diff_map"])
    overlay = make_overlay(input_rgb, diff_map, alpha=0.45)

    pred_name = "fake" if int(pipeline["pred"]) == 1 else "real"
    prob_fake = float(pipeline["prob_fake"])

    fig = plt.figure(figsize=(18, 9))
    axs = [plt.subplot(2, 4, i + 1) for i in range(8)]

    axs[0].imshow(input_rgb)
    axs[0].set_title("1. Input x")
    axs[0].axis("off")

    axs[1].imshow(filtered_rgb)
    axs[1].set_title("2. Filtered image x_f")
    axs[1].axis("off")

    axs[2].imshow(recon_x_rgb)
    axs[2].set_title(r"3. Reconstruction $\hat{x}$")
    axs[2].axis("off")

    axs[3].imshow(recon_filtered_rgb)
    axs[3].set_title(r"4. Reconstruction $\hat{x}_f$")
    axs[3].axis("off")

    axs[4].imshow(delta_raw_map, cmap="gray", vmin=0.0, vmax=1.0)
    axs[4].set_title(r"5. $\Delta_{raw}=|\hat{x}-x|$")
    axs[4].axis("off")

    axs[5].imshow(delta_filtered_map, cmap="gray", vmin=0.0, vmax=1.0)
    axs[5].set_title(r"6. $\Delta_{filtered}=|\hat{x}_f-x|$")
    axs[5].axis("off")

    axs[6].imshow(diff_map, cmap="gray", vmin=0.0, vmax=1.0)
    axs[6].set_title(r"7. $|\Delta_{raw}-\Delta_{filtered}|$")
    axs[6].axis("off")

    axs[7].imshow(overlay)
    axs[7].set_title(f"8. Overlay | pred={pred_name} | p(fake)={prob_fake:.4f}")
    axs[7].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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

    pbar = tqdm(total=4, desc="vis recon pipeline", ncols=110, ascii=True, dynamic_ncols=True)

    pbar.set_postfix(stage="load_model")
    model = load_model(args.ckpt, args.model_name, device=device)
    pbar.update(1)

    pbar.set_postfix(stage="load_image")
    pil = Image.open(image_path).convert("RGB")
    tf = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    x = tf(pil).unsqueeze(0).to(device)
    pbar.update(1)

    pbar.set_postfix(stage="extract_pipeline")
    pipeline = extract_pipeline(model, x)
    pbar.update(1)

    pbar.set_postfix(stage="save_outputs")
    save_panel_rgb(pipeline["input"], out_dir / "01_input_x.png", robust=False)
    save_panel_rgb(pipeline["filtered"], out_dir / "02_filtered_xf.png", robust=True)
    save_panel_rgb(pipeline["recon_x"], out_dir / "03_reconstruction_xhat.png", robust=True)
    save_panel_rgb(pipeline["recon_filtered"], out_dir / "04_reconstruction_xhat_f.png", robust=True)
    save_panel_map(_mean_abs_map(pipeline["delta_raw"]), out_dir / "05_delta_raw.png", cmap="gray")
    save_panel_map(_mean_abs_map(pipeline["delta_filtered"]), out_dir / "06_delta_filtered.png", cmap="gray")
    save_panel_map(pipeline["diff_map"], out_dir / "07_diff_map.png", cmap="gray")

    input_rgb = _to_rgb01(pipeline["input"], robust=False)
    diff_map = _to_map01(pipeline["diff_map"])
    overlay = make_overlay(input_rgb, diff_map, alpha=0.45)
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_dir / "08_diff_overlay.png", dpi=220, bbox_inches="tight", pad_inches=0)
    plt.close()

    save_overview(pipeline, out_dir / "recon_pipeline_overview.png")
    pbar.update(1)
    pbar.close()

    print(f"[OK] saved overview to: {out_dir / 'recon_pipeline_overview.png'}")
    print(f"[OK] saved separate panels under: {out_dir}")


if __name__ == "__main__":
    main()
