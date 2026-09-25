# -*- coding: utf-8 -*-
"""Build an Introduction motivation figure from real model outputs.

The script scans held-out fake samples and finds two complementary cases:
1. the spatial branch is wrong, the reconstruction branch and fusion are correct;
2. the reconstruction branch is wrong, the spatial branch and fusion are correct.

It never modifies training data, checkpoints, or existing result files. Outputs are
written to a dedicated directory as PNG, PDF, CSV, and JSON files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
TMP_ROOT = PROJECT_ROOT / ".tmp"
for directory in (CACHE_ROOT, TMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

# Keep all model/cache activity inside this project and use the local VAE.
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "hf"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(CACHE_ROOT / "hf" / "hub"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "hf" / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_ROOT / "hf" / "transformers"))
os.environ.setdefault("TORCH_HOME", str(CACHE_ROOT / "torch"))
os.environ.setdefault("TIMM_HOME", str(CACHE_ROOT / "timm"))
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
os.environ.setdefault("TEMP", str(TMP_ROOT))
os.environ.setdefault("TMP", str(TMP_ROOT))
os.environ.setdefault("FIRE_VAE_DIR", str(PROJECT_ROOT / "pretrained" / "sd15_vae"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset.transforms import build_transforms_for_model
from core.models import build_model


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One or more held-out test CSV files.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "paper_assets" / "fig_intro_motivation"))
    parser.add_argument("--model_name", default="sfire_crossattn_resnet50")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_scan", type=int, default=2000)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--min_router_gap", type=float, default=0.02)
    parser.add_argument(
        "--require_aligned_router",
        action="store_true",
        help="Optionally require the router to favor the branch that is correct.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reuse_selection",
        action="store_true",
        help="Reuse selected cases from out_dir/selection_summary.json and only redraw the figure.",
    )
    parser.add_argument(
        "--scan_full",
        action="store_true",
        help="Continue to max_scan after both case types are found. By default the scan stops early.",
    )
    return parser.parse_args()


def resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = base / path
    return path.resolve()


class FakeOnlyDataset(Dataset):
    def __init__(self, csv_paths: list[str], model_name: str, image_size: int):
        frames = []
        for csv_value in csv_paths:
            csv_path = resolve_path(csv_value)
            frame = pd.read_csv(csv_path)
            path_column = "path" if "path" in frame.columns else "image"
            if path_column not in frame.columns or "label" not in frame.columns:
                raise ValueError(f"CSV must contain path/image and label columns: {csv_path}")
            frame = frame.loc[frame["label"].astype(int) == 1].copy()
            frame["_csv"] = str(csv_path)
            frame["_path_column"] = path_column
            frames.append(frame)
        if not frames:
            raise ValueError("No CSV inputs were provided.")
        self.frame = pd.concat(frames, ignore_index=True)
        self.transform = build_transforms_for_model(model_name, image_size=image_size, is_train=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = resolve_path(row[row["_path_column"]])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return {
            "image": tensor,
            "path": str(image_path),
            "label": 1,
            "domain": str(row["domain"]) if "domain" in self.frame.columns else Path(row["_csv"]).stem,
        }


def strip_wrappers(state_dict: dict) -> dict:
    clean = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        clean[new_key] = value
    return clean


def load_model(checkpoint: Path, model_name: str, device: torch.device):
    model = build_model(model_name, num_classes=2, pretrained=False, dropout=0.0)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(strip_wrappers(state), strict=True)
    model.to(device).eval()
    return model


def fake_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] >= 2:
        return torch.softmax(logits, dim=1)[:, 1]
    return torch.sigmoid(logits.reshape(-1))


def candidate_score(item: dict, case_type: str) -> float:
    if case_type == "spatial_failure":
        wrong_margin = 0.5 - item["p_spatial"]
        correct_margin = item["p_recon"] - 0.5
        router_margin = item["w_recon"] - item["w_spatial"]
    else:
        wrong_margin = 0.5 - item["p_recon"]
        correct_margin = item["p_spatial"] - 0.5
        router_margin = item["w_spatial"] - item["w_recon"]
    final_margin = item["p_final"] - 0.5
    return float(wrong_margin + correct_margin + final_margin + max(router_margin, 0.0))


def scan_candidates(
    model,
    loader,
    device,
    max_scan: int,
    threshold: float,
    min_router_gap: float,
    require_aligned_router: bool,
    scan_full: bool,
):
    candidates = {"spatial_failure": [], "reconstruction_failure": []}
    counts = {
        "fake_samples_scanned": 0,
        "spatial_failure_fusion_correct": 0,
        "reconstruction_failure_fusion_correct": 0,
        "spatial_failure_with_aligned_router": 0,
        "reconstruction_failure_with_aligned_router": 0,
    }

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Scanning held-out fake samples"):
            remaining = max_scan - counts["fake_samples_scanned"]
            if remaining <= 0:
                break
            images = batch["image"][:remaining].to(device, non_blocking=True)
            outputs = model(images, return_aux=True)
            p_spatial = fake_probability(outputs["spatial_logits"]).detach().cpu()
            p_recon = fake_probability(outputs["fire_logits_2c"]).detach().cpu()
            p_final = fake_probability(outputs["logits"]).detach().cpu()
            weights = outputs["router_weights"].detach().cpu()

            for index in range(images.shape[0]):
                item = {
                    "path": batch["path"][index],
                    "domain": batch["domain"][index],
                    "label": 1,
                    "p_spatial": float(p_spatial[index]),
                    "p_recon": float(p_recon[index]),
                    "p_final": float(p_final[index]),
                    "w_spatial": float(weights[index, 0]),
                    "w_recon": float(weights[index, 1]),
                }
                spatial_pred = int(item["p_spatial"] >= threshold)
                recon_pred = int(item["p_recon"] >= threshold)
                final_pred = int(item["p_final"] >= threshold)
                item.update(
                    spatial_pred=spatial_pred,
                    recon_pred=recon_pred,
                    final_pred=final_pred,
                )

                if spatial_pred == 0 and recon_pred == 1 and final_pred == 1:
                    counts["spatial_failure_fusion_correct"] += 1
                    aligned = item["w_recon"] - item["w_spatial"] >= min_router_gap
                    if aligned:
                        counts["spatial_failure_with_aligned_router"] += 1
                    if aligned or not require_aligned_router:
                        item["selection_score"] = candidate_score(item, "spatial_failure")
                        candidates["spatial_failure"].append(item)

                if recon_pred == 0 and spatial_pred == 1 and final_pred == 1:
                    counts["reconstruction_failure_fusion_correct"] += 1
                    aligned = item["w_spatial"] - item["w_recon"] >= min_router_gap
                    if aligned:
                        counts["reconstruction_failure_with_aligned_router"] += 1
                    if aligned or not require_aligned_router:
                        item["selection_score"] = candidate_score(item, "reconstruction_failure")
                        candidates["reconstruction_failure"].append(item)

            counts["fake_samples_scanned"] += images.shape[0]
            if not scan_full and all(candidates[key] for key in candidates):
                break

    selected = {}
    for key, values in candidates.items():
        selected[key] = max(values, key=lambda value: value["selection_score"]) if values else None
    return selected, counts


def normalize_map(tensor: torch.Tensor, low: float = 0.02, high: float = 0.995) -> np.ndarray:
    tensor = tensor.detach().float().cpu()
    flat = tensor.flatten()
    lo = torch.quantile(flat, low)
    hi = torch.quantile(flat, high)
    normalized = (tensor - lo) / max(float(hi - lo), 1e-8)
    return normalized.clamp(0.0, 1.0).numpy()


def overlay(base: np.ndarray, heatmap: np.ndarray, cmap_name: str, alpha: float = 0.43) -> np.ndarray:
    colored = matplotlib.colormaps.get_cmap(cmap_name)(np.clip(heatmap, 0.0, 1.0))[..., :3]
    return np.clip((1.0 - alpha) * base + alpha * colored, 0.0, 1.0)


def extract_visuals(model, transform, item: dict, device: torch.device) -> dict:
    with Image.open(item["path"]) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model(tensor, return_aux=True)
        spatial_map, _, _ = model._extract_spatial_features(tensor)
        spatial_map = spatial_map.abs().mean(dim=1, keepdim=True)
        spatial_map = F.interpolate(spatial_map, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        anomaly = outputs["anomaly_prior"]
        anomaly = F.interpolate(anomaly, size=tensor.shape[-2:], mode="bilinear", align_corners=False)

    rgb = tensor[0].detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)
    spatial = normalize_map(spatial_map[0, 0])
    anomaly_map = normalize_map(anomaly[0, 0])
    return {
        "input": rgb,
        "spatial_overlay": overlay(rgb, spatial, "viridis"),
        "anomaly_overlay": overlay(rgb, anomaly_map, "magma"),
    }


def style_axis(axis):
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.5, 2.5)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", labelsize=8)
    axis.grid(axis="x", color="#DDDDDD", linewidth=0.6, alpha=0.8)
    axis.axvline(0.5, color="#666666", linestyle="--", linewidth=0.9)


def draw_figure(selected: dict, visuals: dict, out_dir: Path, threshold: float):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    rows = [
        ("spatial_failure", "(a) Spatial-branch failure"),
        ("reconstruction_failure", "(b) Reconstruction-branch failure"),
    ]
    fig = plt.figure(figsize=(11.8, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 5, width_ratios=[1.0, 1.0, 1.0, 1.22, 1.22])

    for row_index, (key, row_title) in enumerate(rows):
        item = selected[key]
        visual = visuals[key]
        image_titles = ["Input", "Spatial response", "Anomaly prior"]
        image_keys = ["input", "spatial_overlay", "anomaly_overlay"]
        for column_index, (image_title, image_key) in enumerate(zip(image_titles, image_keys)):
            axis = fig.add_subplot(grid[row_index, column_index])
            axis.imshow(visual[image_key])
            axis.axis("off")
            if row_index == 0:
                axis.set_title(image_title, fontweight="bold", pad=5)
            if column_index == 0:
                axis.text(
                    -0.08,
                    0.5,
                    f"{row_title}\nDomain: {item['domain']}\nGT: Fake",
                    transform=axis.transAxes,
                    ha="right",
                    va="center",
                    fontsize=8.5,
                )

        score_axis = fig.add_subplot(grid[row_index, 3])
        score_labels = ["Spatial", "Recon.", "Fusion"]
        score_values = [item["p_spatial"], item["p_recon"], item["p_final"]]
        colors = ["#4C78A8", "#F28E2B", "#2E7D32"]
        score_axis.barh([2, 1, 0], score_values, color=colors, height=0.54)
        score_axis.set_yticks([2, 1, 0], score_labels)
        score_axis.set_xticks([0.0, 0.5, 1.0])
        style_axis(score_axis)
        for y_value, value in zip([2, 1, 0], score_values):
            score_axis.text(min(value + 0.025, 0.94), y_value, f"{value:.3f}", va="center", fontsize=8)
        if row_index == 0:
            score_axis.set_title("Fake-class probability", fontweight="bold", pad=5)
        score_axis.set_xlabel(f"Decision threshold = {threshold:.2f}", fontsize=7.5, labelpad=3)

        outcome_axis = fig.add_subplot(grid[row_index, 4])
        outcome_axis.set_xlim(0.0, 1.0)
        outcome_axis.set_ylim(0.0, 1.0)
        outcome_axis.axis("off")
        if row_index == 0:
            outcome_axis.set_title("Complementary recovery", fontweight="bold", pad=5)
        if key == "spatial_failure":
            cue_text = "Spatial criterion\nmisses the fake"
            recovery_text = "Reconstruction evidence\nrecovers the decision"
        else:
            cue_text = "Reconstruction criterion\nmisses the fake"
            recovery_text = "Spatial evidence\nrecovers the decision"
        outcome_axis.text(
            0.5,
            0.76,
            cue_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#B3261E",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#FDECEA", "edgecolor": "#D98C86"},
        )
        outcome_axis.annotate(
            "",
            xy=(0.5, 0.43),
            xytext=(0.5, 0.59),
            arrowprops={"arrowstyle": "->", "color": "#666666", "linewidth": 1.2},
        )
        outcome_axis.text(
            0.5,
            0.34,
            recovery_text,
            ha="center",
            va="center",
            fontsize=8.5,
            color="#1B5E20",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EAF4EA", "edgecolor": "#86B889"},
        )
        outcome_axis.text(
            0.5,
            0.08,
            "Fusion: Fake (correct)",
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#2E7D32",
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "fig1_intro_motivation.png"
    pdf_path = out_dir / "fig1_intro_motivation.pdf"
    fig.savefig(png_path, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.out_dir)
    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    requested_device = args.device.lower()
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(args.device)

    model = load_model(checkpoint, args.model_name, device)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "selection_summary.json"
    if args.reuse_selection:
        if not summary_path.is_file():
            raise FileNotFoundError(f"Selection summary not found: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary["selected"]
        counts = summary["counts"]
    else:
        dataset = FakeOnlyDataset(args.csv, args.model_name, args.image_size)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )
        selected, counts = scan_candidates(
            model,
            loader,
            device,
            min(args.max_scan, len(dataset)),
            args.threshold,
            args.min_router_gap,
            args.require_aligned_router,
            args.scan_full,
        )
        summary = {
            "checkpoint": str(checkpoint),
            "csv_files": [str(resolve_path(value)) for value in args.csv],
            "threshold": args.threshold,
            "minimum_router_gap": args.min_router_gap,
            "require_aligned_router": args.require_aligned_router,
            "selection_rule": (
                "Fusion correct and exactly one branch wrong. "
                + ("Routing must also favor the correct branch. " if args.require_aligned_router else "")
                + "Unless --scan_full is used, scanning stops after the first deterministic batch order "
                "that contains both case types."
            ),
            "counts": counts,
            "selected": selected,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [key for key, value in selected.items() if value is None]
    if missing:
        raise RuntimeError(
            "No candidate was found for: " + ", ".join(missing) + ". "
            "Try additional held-out domains or increase --max_scan."
        )

    pd.DataFrame(selected.values()).to_csv(out_dir / "selected_cases.csv", index=False)
    transform = build_transforms_for_model(args.model_name, image_size=args.image_size, is_train=False)
    visuals = {
        key: extract_visuals(model, transform, item, device) for key, item in selected.items()
    }
    png_path, pdf_path = draw_figure(selected, visuals, out_dir, args.threshold)
    print(json.dumps({"png": str(png_path), "pdf": str(pdf_path), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
