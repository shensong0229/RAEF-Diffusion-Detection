# -*- coding: utf-8 -*-
"""
Export real visualization panels for the reconstruction-aware branch.

This script exports square image panels for paper figures:
  01_input.png
  02_middle_freq_image.png
  03_middle_filtered_image.png
  04_reconstruction_x.png
  05_reconstruction_middle_filtered.png
  06_raw_delta.png
  07_filtered_delta.png
  08_anomaly_prior.png
  09_feature_map_preview.png
  10_contact_sheet.png

Usage example on Windows PowerShell:

python .\\tools\\export_reconstruction_branch_panels.py `
  --ckpt "F:\\FakeImageDetect\\checkpoints\\run_sfire_best_test\\best_by_auc.pth" `
  --image "F:\\FakeImageDetect\\export_test_by_domain_unique_real\\glide\\nature\\sdv4__train__nature__n01514668_26005.JPEG" `
  --out_dir "F:\\FakeImageDetect\\paper_assets\\fig_reconstruction_branch_sample" `
  --image_size 256 `
  --model_name sfire_crossattn_resnet50 `
  --device auto
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms


def add_project_root_to_sys_path(project_root: Path) -> None:
    project_root = project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def fix_env_paths(project_root: Path) -> None:
    """
    Avoid stale cache/temp paths and make FIRE VAE local by default.
    """
    cache_base = project_root / ".cache"
    tmp_base = project_root / ".tmp"
    cache_base.mkdir(parents=True, exist_ok=True)
    tmp_base.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(cache_base / "hf"))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_base / "hf" / "hub"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_base / "hf" / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_base / "hf" / "transformers"))
    os.environ.setdefault("TORCH_HOME", str(cache_base / "torch"))
    os.environ.setdefault("TIMM_HOME", str(cache_base / "timm"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_base))
    os.environ.setdefault("TEMP", str(tmp_base))
    os.environ.setdefault("TMP", str(tmp_base))
    os.environ.setdefault("TMPDIR", str(tmp_base))

    vae_dir = project_root / "pretrained" / "sd15_vae"
    if vae_dir.is_dir():
        os.environ.setdefault("FIRE_VAE_DIR", str(vae_dir))


def square_center_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = int((w - side) / 2)
    top = int((h - side) / 2)
    return img.crop((left, top, left + side, top + side))


def load_image_as_tensor(image_path: Path, image_size: int) -> Tuple[torch.Tensor, Image.Image]:
    img = Image.open(image_path).convert("RGB")
    # Match sfire eval transform: raw RGB tensor in [0,1], resize+center crop.
    tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )
    x = tf(img).unsqueeze(0)
    vis_img = transforms.ToPILImage()(x[0].clamp(0, 1))
    return x, vis_img


def strip_wrappers(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out


def build_and_load_model(model_name: str, ckpt: Path, device: str):
    from core.models import build_model

    model = build_model(model_name, num_classes=2, pretrained=False, dropout=0.0)
    payload = torch.load(str(ckpt), map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    state = strip_wrappers(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"[INFO] loaded checkpoint: {ckpt}")
    print(f"[INFO] missing keys: {len(missing)}")
    print(f"[INFO] unexpected keys: {len(unexpected)}")
    if missing:
        print("[WARN] first missing keys:", list(missing)[:10])
    if unexpected:
        print("[WARN] first unexpected keys:", list(unexpected)[:10])

    model.to(device)
    model.eval()
    return model


def tensor_rgb_to_pil(x: torch.Tensor, size: int = 512) -> Image.Image:
    """
    x: [3,H,W] or [1,3,H,W], expected roughly [0,1] or VAE output range.
    """
    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().cpu()
    x = x.clamp(0, 1)
    img = transforms.ToPILImage()(x)
    img = square_center_crop(img).resize((size, size), Image.Resampling.BICUBIC)
    return img


def tensor_map_to_pil(
    x: torch.Tensor,
    size: int = 512,
    cmap: str = "magma",
    normalize: bool = True,
) -> Image.Image:
    """
    x: [C,H,W] or [1,C,H,W]. If C>1, channel mean is used.
    """
    import matplotlib.cm as cm

    if x.ndim == 4:
        x = x[0]
    x = x.detach().float().cpu()
    if x.ndim == 3:
        x = x.mean(dim=0)
    elif x.ndim != 2:
        raise ValueError(f"Expected 2D/3D/4D tensor map, got shape={tuple(x.shape)}")

    arr = x.numpy()
    if normalize:
        arr = arr - float(np.nanmin(arr))
        denom = float(np.nanmax(arr)) + 1e-8
        arr = arr / denom
    arr = np.clip(arr, 0, 1)

    rgba = cm.get_cmap(cmap)(arr)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    img = square_center_crop(img).resize((size, size), Image.Resampling.BICUBIC)
    return img


def feature_map_preview(feature_map: torch.Tensor, size: int = 512) -> Image.Image:
    """
    Create a simple heatmap preview from a high-dimensional feature map.
    """
    if feature_map.ndim == 4:
        fmap = feature_map[0]
    else:
        fmap = feature_map
    # Average absolute activation over channels.
    fmap = fmap.detach().float().abs().mean(dim=0)
    return tensor_map_to_pil(fmap, size=size, cmap="viridis", normalize=True)


def save_panel(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def labeled_tile(img: Image.Image, title: str, tile_size: int = 260) -> Image.Image:
    """
    Make a labeled square tile for a contact sheet.
    """
    img = img.resize((tile_size, tile_size), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (tile_size, tile_size + 44), "white")
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((8, tile_size + 12), title, fill=(0, 0, 0), font=font)
    return canvas


def make_contact_sheet(panels: Dict[str, Image.Image], out_path: Path) -> None:
    order = [
        ("input", "Input I"),
        ("middle_filtered", "Middle-filtered"),
        ("reconstruction", "Recon. I_r"),
        ("raw_delta", "Raw delta"),
        ("filtered_delta", "Filtered delta"),
        ("anomaly_prior", "Anomaly prior A"),
        ("feature_map", "Feature map F_r"),
    ]
    tiles = []
    for key, title in order:
        if key in panels:
            tiles.append(labeled_tile(panels[key], title))

    if not tiles:
        return

    cols = 4
    rows = int(np.ceil(len(tiles) / cols))
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, rows * th), "white")
    for i, tile in enumerate(tiles):
        x = (i % cols) * tw
        y = (i // cols) * th
        sheet.paste(tile, (x, y))
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--image", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--panel_size", type=int, default=512)
    parser.add_argument("--model_name", default="sfire_crossattn_resnet50")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    add_project_root_to_sys_path(project_root)
    fix_env_paths(project_root)

    ckpt = Path(args.ckpt)
    image_path = Path(args.image)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] image        = {image_path}")
    print(f"[INFO] out_dir      = {out_dir}")
    print(f"[INFO] device       = {device}")
    print(f"[INFO] FIRE_VAE_DIR = {os.environ.get('FIRE_VAE_DIR', '')}")

    x, input_vis = load_image_as_tensor(image_path, image_size=args.image_size)
    x = x.to(device)

    model = build_and_load_model(args.model_name, ckpt, device=device)

    with torch.inference_mode():
        # Use the exact FIRE branch already loaded inside Ours/SFIRE.
        if not hasattr(model, "fire_branch"):
            raise AttributeError(
                f"Model {args.model_name} has no fire_branch. "
                "Use model_name=sfire_crossattn_resnet50 for Ours."
            )
        aux = model.fire_branch.extract_features(x, return_aux=True)

    panels: Dict[str, Image.Image] = {}
    panels["input"] = input_vis.resize((args.panel_size, args.panel_size), Image.Resampling.BICUBIC)

    if "middle_freq_image" in aux:
        panels["middle_freq"] = tensor_rgb_to_pil(aux["middle_freq_image"], size=args.panel_size)
    if "middle_filtered_image" in aux:
        panels["middle_filtered"] = tensor_rgb_to_pil(aux["middle_filtered_image"], size=args.panel_size)
    if "reconstructions_x" in aux:
        panels["reconstruction"] = tensor_rgb_to_pil(aux["reconstructions_x"], size=args.panel_size)
    if "reconstructions_middle_filtered" in aux:
        panels["reconstruction_middle_filtered"] = tensor_rgb_to_pil(
            aux["reconstructions_middle_filtered"],
            size=args.panel_size,
        )
    if "raw_reconstructions_delta" in aux:
        panels["raw_delta"] = tensor_map_to_pil(
            aux["raw_reconstructions_delta"],
            size=args.panel_size,
            cmap="magma",
            normalize=True,
        )
    if "filtered_reconstructions_delta" in aux:
        panels["filtered_delta"] = tensor_map_to_pil(
            aux["filtered_reconstructions_delta"],
            size=args.panel_size,
            cmap="magma",
            normalize=True,
        )
    if "anomaly_prior" in aux:
        panels["anomaly_prior"] = tensor_map_to_pil(
            aux["anomaly_prior"],
            size=args.panel_size,
            cmap="gray",
            normalize=True,
        )
    if "feature_map" in aux:
        panels["feature_map"] = feature_map_preview(aux["feature_map"], size=args.panel_size)

    name_map = {
        "input": "01_input_I.png",
        "middle_freq": "02_middle_frequency_image.png",
        "middle_filtered": "03_middle_filtered_image.png",
        "reconstruction": "04_reconstructed_image_Ir.png",
        "reconstruction_middle_filtered": "05_reconstructed_middle_filtered.png",
        "raw_delta": "06_raw_reconstruction_delta_Draw.png",
        "filtered_delta": "07_filtered_reconstruction_delta_Dfiltered.png",
        "anomaly_prior": "08_anomaly_prior_A.png",
        "feature_map": "09_reconstruction_feature_preview_Fr.png",
    }

    for key, img in panels.items():
        if key in name_map:
            save_panel(img, out_dir / name_map[key])
            print(f"[OK] saved {name_map[key]}")

    make_contact_sheet(panels, out_dir / "10_contact_sheet.png")
    print(f"[OK] saved contact sheet: {out_dir / '10_contact_sheet.png'}")

    # Save a short markdown note for figure drawing.
    note = f"""# Reconstruction-aware branch panels

Source image:
{image_path}

Checkpoint:
{ckpt}

Suggested usage in draw.io:
- Input Image I: 01_input_I.png
- Reconstructed Image Ir: 04_reconstructed_image_Ir.png
- Difference Map D: 06_raw_reconstruction_delta_Draw.png
- Normalized / filtered Error Map: 07_filtered_reconstruction_delta_Dfiltered.png
- Anomaly Prior A: 08_anomaly_prior_A.png
- Reconstruction-aware Feature Fr: 09_reconstruction_feature_preview_Fr.png

Code-level note:
- raw_delta = |reconstructions_x - x|
- filtered_delta = |reconstructions_middle_filtered - x|
- anomaly_prior = mean(abs(raw_delta - filtered_delta), channel)
"""
    (out_dir / "README_panels.md").write_text(note, encoding="utf-8")
    print(f"[OK] wrote README: {out_dir / 'README_panels.md'}")


if __name__ == "__main__":
    main()