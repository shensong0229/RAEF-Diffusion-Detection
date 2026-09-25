# -*- coding: utf-8 -*-
"""
Visualize routing weights & prediction card for one image.

Usage (PowerShell):
python .\tools\vis_router_card_one.py `
  --image_path "F:\FakeImageDetect\DiffusionForensics\images\train\imagenet\real\n02100236\ILSVRC2012_val_00016444.JPEG" `
  --ckpt ".\checkpoints\run_sfire_best_test\best_by_auc.pth" `
  --model_name sfire_crossattn_resnet50 `
  --image_size 256 `
  --gt_label real `
  --out_path ".\results\vis\router_card_one.png"

Notes
-----
1) This script is designed for the final Ours model: sfire_crossattn_resnet50.
2) It creates a paper-friendly panel image for the 4th column of your visualization:
   - GT / Pred / p(fake)
   - router weights for Spatial / Reconstruction-aware branches
   - spatial score / reconstruction-aware score / final score
3) The output is a standalone PNG tile, so you can directly mosaic it with the
   other three columns.
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


def binary_fake_prob_from_two_class_logits(logits: torch.Tensor) -> float:
    prob = torch.softmax(logits, dim=1)[0, 1].item()
    return float(prob)


def parse_gt_label(gt: str) -> int:
    gt = str(gt).strip().lower()
    if gt in {"0", "real"}:
        return 0
    if gt in {"1", "fake"}:
        return 1
    raise ValueError(f"Unsupported gt_label={gt!r}; use real/fake or 0/1.")


def gather_outputs(model, x: torch.Tensor):
    aux = model(x, return_aux=True)
    if not isinstance(aux, dict):
        raise ValueError("Expected dict output from model(x, return_aux=True).")

    logits = aux["logits"]                    # [1,2]
    spatial_logits = aux["spatial_logits"]    # [1,2]
    fire_logits_2c = aux["fire_logits_2c"]    # [1,2]
    router_weights = aux["router_weights"]    # [1,2]

    p_final = binary_fake_prob_from_two_class_logits(logits)
    p_spatial = binary_fake_prob_from_two_class_logits(spatial_logits)
    p_recon = binary_fake_prob_from_two_class_logits(fire_logits_2c)

    pred = int(torch.argmax(logits, dim=1)[0].item())
    ws = float(router_weights[0, 0].item())
    wr = float(router_weights[0, 1].item())

    return {
        "pred": pred,
        "p_final": p_final,
        "p_spatial": p_spatial,
        "p_recon": p_recon,
        "w_spatial": ws,
        "w_recon": wr,
    }


def draw_card(info: dict, gt_label: int, out_path: Path, title: str = "Routing weights & prediction"):
    pred_label = info["pred"]
    p_final = info["p_final"]
    p_spatial = info["p_spatial"]
    p_recon = info["p_recon"]
    w_spatial = info["w_spatial"]
    w_recon = info["w_recon"]

    gt_name = "Real" if gt_label == 0 else "Fake"
    pred_name = "Real" if pred_label == 0 else "Fake"
    is_correct = (gt_label == pred_label)

    fig = plt.figure(figsize=(4.8, 4.8))
    ax = plt.gca()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background panel
    rect = plt.Rectangle((0.03, 0.03), 0.94, 0.94, facecolor="white", edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)

    # Title
    ax.text(0.5, 0.92, title, ha="center", va="center", fontsize=14, fontweight="bold")

    # GT / Pred / Final score
    ax.text(0.10, 0.80, f"GT:   {gt_name}", ha="left", va="center", fontsize=13)
    ax.text(0.10, 0.72, f"Pred: {pred_name}", ha="left", va="center", fontsize=13)
    ax.text(0.10, 0.64, f"p(fake): {p_final:.4f}", ha="left", va="center", fontsize=13)

    badge_text = "Correct" if is_correct else "Wrong"
    badge_color = "#2ca02c" if is_correct else "#d62728"
    badge = plt.Rectangle((0.67, 0.69), 0.20, 0.12, facecolor=badge_color, edgecolor="black", linewidth=1.0)
    ax.add_patch(badge)
    ax.text(0.77, 0.75, badge_text, ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    # Router weights section
    ax.text(0.10, 0.54, "Router weights", ha="left", va="center", fontsize=13, fontweight="bold")

    # Spatial bar
    ax.text(0.10, 0.46, "Spatial", ha="left", va="center", fontsize=12)
    ax.add_patch(plt.Rectangle((0.33, 0.43), 0.48, 0.05, facecolor="#e8eef7", edgecolor="black", linewidth=0.8))
    ax.add_patch(plt.Rectangle((0.33, 0.43), 0.48 * w_spatial, 0.05, facecolor="#4c78a8", edgecolor="none"))
    ax.text(0.84, 0.46, f"{w_spatial:.2f}", ha="left", va="center", fontsize=12)

    # Recon bar
    ax.text(0.10, 0.36, "Recon.", ha="left", va="center", fontsize=12)
    ax.add_patch(plt.Rectangle((0.33, 0.33), 0.48, 0.05, facecolor="#f8eadc", edgecolor="black", linewidth=0.8))
    ax.add_patch(plt.Rectangle((0.33, 0.33), 0.48 * w_recon, 0.05, facecolor="#f28e2b", edgecolor="none"))
    ax.text(0.84, 0.36, f"{w_recon:.2f}", ha="left", va="center", fontsize=12)

    # Scores section
    ax.text(0.10, 0.22, "Branch scores", ha="left", va="center", fontsize=13, fontweight="bold")
    ax.text(0.10, 0.15, f"Spatial score: {p_spatial:.4f}", ha="left", va="center", fontsize=12)
    ax.text(0.10, 0.10, f"Recon. score:  {p_recon:.4f}", ha="left", va="center", fontsize=12)
    ax.text(0.10, 0.05, f"Final score:    {p_final:.4f}", ha="left", va="center", fontsize=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="sfire_crossattn_resnet50")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--gt_label", required=True, help="real/fake or 0/1")
    ap.add_argument("--out_path", required=True)
    args = ap.parse_args()

    image_path = Path(args.image_path).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    out_path = Path(args.out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_label = parse_gt_label(args.gt_label)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, args.model_name, device=device)

    pil = Image.open(image_path).convert("RGB")
    tf = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    x = tf(pil).unsqueeze(0).to(device)

    with torch.inference_mode():
        info = gather_outputs(model, x)

    draw_card(info, gt_label=gt_label, out_path=out_path)
    print(f"[OK] saved to: {out_path}")


if __name__ == "__main__":
    main()
