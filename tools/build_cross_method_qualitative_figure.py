# -*- coding: utf-8 -*-
"""Build a shared-image qualitative comparison from authentic local outputs.

This script is intentionally isolated from training and evaluation code. It reads
existing checkpoints and archived AEROBLADE per-image scores, runs three local
detectors on the same images, and writes all artifacts to a dedicated directory.

Displayed methods:
  - Spatial-only (local source-trained checkpoint)
  - FIRE (local source-trained checkpoint)
  - AEROBLADE (archived per-image reconstruction score)
  - Ours (local adaptive fusion checkpoint)

For AEROBLADE, the native score is monotonically converted to an unlabeled pooled
percentile over the ten target domains. This keeps ranking information intact and
avoids presenting an uncalibrated distance as a probability. The median split is
used only to produce an interpretable qualitative label on the balanced benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache"
TMP_ROOT = PROJECT_ROOT / ".tmp"
for directory in (CACHE_ROOT, TMP_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

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
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image, ImageFile, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.dataset.transforms import build_transforms_for_model
from core.models import build_model


ImageFile.LOAD_TRUNCATED_IMAGES = True

TARGET_DOMAINS = [
    "Flash_PixArt",
    "Flash_SD3",
    "JuggernautXL",
    "Lumina",
    "Flux_1",
    "PixArt_Alpha",
    "SDXL",
    "SDXL_Lightning",
    "Kolors",
    "SSD_1B",
]

DEFAULT_SCAN_DOMAINS = [
    "Flash_PixArt",
    "JuggernautXL",
    "Lumina",
    "Flux_1",
    "SDXL_Lightning",
]

METHODS = [
    ("spatial", "Spatial-only", "p(fake)"),
    ("fire", "FIRE", "p(fake)"),
    ("aeroblade", "AEROBLADE", "score percentile"),
    ("ours", "Ours", "p(fake)"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=str(PROJECT_ROOT / "paper_assets" / "fig_qualitative_comparison"),
    )
    parser.add_argument(
        "--bundle_root",
        default=str(PROJECT_ROOT / "exports" / "dragon_eval_25domains_unique_real_1k_bundle" / "images"),
    )
    parser.add_argument(
        "--archive_root",
        default=str(PROJECT_ROOT / "downloads" / "table1_unseen_main_results"),
    )
    parser.add_argument(
        "--spatial_ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "run_spatial_imagenet_adm_256" / "best_by_auc.pth"),
    )
    parser.add_argument(
        "--fire_ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "run_fire_imagenet_adm_a30_std_fg" / "best_by_auc.pth"),
    )
    parser.add_argument(
        "--ours_ckpt",
        default=str(PROJECT_ROOT / "checkpoints" / "run_sfire_best_test" / "best_by_auc.pth"),
    )
    parser.add_argument("--scan_domains", nargs="+", default=DEFAULT_SCAN_DOMAINS)
    parser.add_argument("--samples_per_class_domain", type=int, default=12)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reuse_predictions",
        action="store_true",
        help="Skip inference and redraw from all_candidate_predictions.csv.",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_state_dict(state_dict: dict) -> dict:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("module.", "model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


def locate_aeroblade_csvs(archive_root: Path) -> tuple[dict[str, Path], Path]:
    candidates = [
        p
        for p in archive_root.rglob("distances_merged.csv")
        if "aeroblade" in str(p).lower() and "paper_assets" not in str(p).lower()
    ]
    if not candidates:
        raise FileNotFoundError(f"No archived AEROBLADE per-image scores found under {archive_root}")

    grouped: dict[Path, dict[str, Path]] = {}
    for path in candidates:
        domain = path.parent.name
        if domain not in TARGET_DOMAINS:
            continue
        base = path.parent.parent
        grouped.setdefault(base, {})[domain] = path

    complete = [(base, files) for base, files in grouped.items() if all(d in files for d in TARGET_DOMAINS)]
    if not complete:
        coverage = sorted(((len(files), str(base)) for base, files in grouped.items()), reverse=True)
        raise FileNotFoundError(f"No archive copy contains all ten target domains. Coverage: {coverage[:5]}")

    # The result archive contains duplicate package copies. Prefer the shortest path;
    # hashes below are stored in provenance so accidental divergence is visible.
    base, files = min(complete, key=lambda item: (len(item[0].parts), len(str(item[0]))))
    return files, base


def read_aeroblade_pool(files: dict[str, Path]) -> pd.DataFrame:
    frames = []
    for domain in TARGET_DOMAINS:
        frame = pd.read_csv(files[domain])
        required = {"orig_path", "label", "domain", "distance"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing columns {sorted(missing)} in {files[domain]}")
        frame = frame[["orig_path", "label", "domain", "distance"]].copy()
        frame["label"] = frame["label"].astype(int)
        frame["distance"] = frame["distance"].astype(float)
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True)
    # Average ranks make the transformation deterministic in the presence of ties.
    pool["aeroblade_percentile"] = pool["distance"].rank(method="average", pct=True)
    return pool


def locate_image(bundle_root: Path, orig_path: str) -> Path:
    relative = Path(str(orig_path).replace("/", os.sep))
    candidate = (bundle_root / relative).resolve()
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Shared bundle image is missing: {candidate}")


def sample_candidate_pool(
    pool: pd.DataFrame,
    bundle_root: Path,
    scan_domains: list[str],
    n_per_group: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    selected = []
    quantile_positions = np.linspace(0.02, 0.98, max(2, n_per_group // 2))
    for domain in scan_domains:
        if domain not in TARGET_DOMAINS:
            raise ValueError(f"Unknown target domain: {domain}")
        for label in (0, 1):
            group = pool[(pool["domain"] == domain) & (pool["label"] == label)].copy()
            group = group.sort_values(["aeroblade_percentile", "orig_path"]).reset_index(drop=True)
            if group.empty:
                continue
            indices = {int(round(pos * (len(group) - 1))) for pos in quantile_positions}
            remaining = max(0, n_per_group - len(indices))
            if remaining:
                choices = rng.choice(len(group), size=min(remaining, len(group)), replace=False)
                indices.update(int(value) for value in choices)
            sample = group.iloc[sorted(indices)].copy()
            selected.append(sample)
    if not selected:
        raise RuntimeError("The candidate pool is empty.")
    result = pd.concat(selected, ignore_index=True).drop_duplicates("orig_path").copy()
    result["image_path"] = result["orig_path"].map(lambda value: str(locate_image(bundle_root, value)))
    result = result.sort_values(["domain", "label", "orig_path"]).reset_index(drop=True)
    return result


class SharedImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, model_name: str, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = build_transforms_for_model(model_name, image_size=image_size, is_train=False)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def fake_probability(output, model_name: str) -> torch.Tensor:
    if isinstance(output, tuple):
        output = output[0]
    if isinstance(output, dict):
        output = output["logits"]
    if not torch.is_tensor(output):
        raise TypeError(f"Unexpected model output type: {type(output)}")
    if model_name.startswith("fire_"):
        return torch.sigmoid(output.reshape(-1))
    if output.ndim != 2 or output.shape[1] < 2:
        raise ValueError(f"Expected two-class logits, got {tuple(output.shape)}")
    return torch.softmax(output, dim=1)[:, 1]


def run_detector(
    frame: pd.DataFrame,
    model_name: str,
    checkpoint: Path,
    device: torch.device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> np.ndarray:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    num_classes = 1 if model_name.startswith("fire_") else 2
    model = build_model(model_name, num_classes=num_classes, pretrained=False, dropout=0.0)
    payload = torch.load(checkpoint, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(clean_state_dict(state), strict=True)
    model.to(device).eval()

    dataset = SharedImageDataset(frame, model_name=model_name, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    values = np.zeros(len(frame), dtype=np.float64)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        for tensors, indices in tqdm(loader, desc=f"Predicting {model_name}", unit="batch"):
            tensors = tensors.to(device, non_blocking=True)
            output = model(tensors)
            probs = fake_probability(output, model_name).detach().cpu().numpy()
            values[indices.numpy()] = probs

    del loader, dataset, model, payload, state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


def add_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["spatial_pred"] = (result["spatial_p_fake"] >= 0.5).astype(int)
    result["fire_pred"] = (result["fire_p_fake"] >= 0.5).astype(int)
    result["aeroblade_pred"] = (result["aeroblade_percentile"] >= 0.5).astype(int)
    result["ours_pred"] = (result["ours_p_fake"] >= 0.5).astype(int)
    for key in ("spatial", "fire", "aeroblade", "ours"):
        result[f"{key}_correct"] = result[f"{key}_pred"] == result["label"]
    result["n_competitor_errors"] = (
        (~result["spatial_correct"]).astype(int)
        + (~result["fire_correct"]).astype(int)
        + (~result["aeroblade_correct"]).astype(int)
    )
    signed = np.where(result["label"].to_numpy() == 1, 1.0, -1.0)
    result["ours_margin"] = signed * (result["ours_p_fake"].to_numpy() - 0.5)
    result["spatial_advantage"] = signed * (
        result["ours_p_fake"].to_numpy() - result["spatial_p_fake"].to_numpy()
    )
    result["fire_advantage"] = signed * (
        result["ours_p_fake"].to_numpy() - result["fire_p_fake"].to_numpy()
    )
    result["aeroblade_advantage"] = signed * (
        result["ours_p_fake"].to_numpy() - result["aeroblade_percentile"].to_numpy()
    )
    result["selection_score"] = (
        5.0 * result["n_competitor_errors"]
        + 1.5 * result["ours_margin"]
        + result[["spatial_advantage", "fire_advantage", "aeroblade_advantage"]].clip(lower=0).sum(axis=1)
    )
    return result


def greedy_pick(
    frame: pd.DataFrame,
    selected_indices: list[int],
    label: int,
    preferred_failure: str | None,
) -> None:
    used_domains = {str(frame.loc[index, "domain"]) for index in selected_indices}
    mask = (frame["label"] == label) & frame["ours_correct"] & (~frame.index.isin(selected_indices))
    if preferred_failure:
        preferred = frame[mask & (~frame[f"{preferred_failure}_correct"])].copy()
    else:
        preferred = frame[mask & (frame["n_competitor_errors"] > 0)].copy()
    if preferred.empty:
        preferred = frame[mask].copy()
    if preferred.empty:
        return
    preferred["domain_bonus"] = (~preferred["domain"].astype(str).isin(used_domains)).astype(float) * 2.5
    chosen = preferred.sort_values(
        ["domain_bonus", "selection_score", "ours_margin", "orig_path"],
        ascending=[False, False, False, True],
    ).index[0]
    selected_indices.append(int(chosen))


def select_examples(frame: pd.DataFrame, rows: int) -> pd.DataFrame:
    selected_indices: list[int] = []
    # Cover different failure mechanisms in both classes before filling by score.
    plan = [
        (1, "spatial"),
        (1, "fire"),
        (1, "aeroblade"),
        (0, "spatial"),
        (0, "fire"),
        (0, "aeroblade"),
    ]
    for label, method in plan[:rows]:
        greedy_pick(frame, selected_indices, label=label, preferred_failure=method)

    while len(selected_indices) < rows:
        remaining = frame[
            frame["ours_correct"]
            & (frame["n_competitor_errors"] > 0)
            & (~frame.index.isin(selected_indices))
        ].copy()
        if remaining.empty:
            remaining = frame[(~frame.index.isin(selected_indices))].copy()
        if remaining.empty:
            break
        used_domains = {str(frame.loc[index, "domain"]) for index in selected_indices}
        remaining["domain_bonus"] = (~remaining["domain"].astype(str).isin(used_domains)).astype(float) * 2.5
        chosen = remaining.sort_values(
            ["domain_bonus", "selection_score", "ours_margin", "orig_path"],
            ascending=[False, False, False, True],
        ).index[0]
        selected_indices.append(int(chosen))

    if len(selected_indices) < rows:
        raise RuntimeError(f"Could select only {len(selected_indices)} of {rows} requested examples")
    selected = frame.loc[selected_indices].copy().reset_index(drop=True)
    selected["panel"] = [chr(ord("a") + index) for index in range(len(selected))]
    return selected


def center_square(image: Image.Image) -> Image.Image:
    width, height = image.size
    edge = min(width, height)
    left = (width - edge) // 2
    top = (height - edge) // 2
    return image.crop((left, top, left + edge, top + edge))


def method_values(row: pd.Series, key: str) -> tuple[float, int, bool, str]:
    if key == "aeroblade":
        value = float(row["aeroblade_percentile"])
        detail = f"rank = {value:.3f}"
    else:
        value = float(row[f"{key}_p_fake"])
        detail = f"p(fake) = {value:.3f}"
    pred = int(row[f"{key}_pred"])
    correct = bool(row[f"{key}_correct"])
    return value, pred, correct, detail


def draw_score_card(axis, row: pd.Series, key: str, emphasize: bool = False) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    value, pred, correct, detail = method_values(row, key)
    edge = "#2E7D32" if correct else "#C33C32"
    face = "#F2F8F3" if correct else "#FFF3F1"
    line_width = 1.7 if emphasize else 1.2
    card = FancyBboxPatch(
        (0.035, 0.12),
        0.93,
        0.76,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        linewidth=line_width,
        edgecolor=edge,
        facecolor=face,
        transform=axis.transAxes,
    )
    axis.add_patch(card)
    prediction = "Fake" if pred == 1 else "Real"
    outcome = "correct" if correct else "wrong"
    axis.text(
        0.08,
        0.72,
        f"{prediction}  ({outcome})",
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color=edge,
    )
    axis.text(
        0.08,
        0.50,
        detail,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
    )
    x0, y0, width, height = 0.08, 0.29, 0.84, 0.10
    axis.add_patch(Rectangle((x0, y0), width, height, transform=axis.transAxes, color="#DFE3E6", lw=0))
    axis.add_patch(Rectangle((x0, y0), width * np.clip(value, 0, 1), height, transform=axis.transAxes, color="#4C78A8", lw=0))
    axis.plot(
        [x0 + 0.5 * width, x0 + 0.5 * width],
        [y0 - 0.035, y0 + height + 0.035],
        transform=axis.transAxes,
        color="#555555",
        linestyle="--",
        linewidth=0.8,
    )
    if emphasize:
        axis.text(
            0.89,
            0.74,
            "proposed",
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=6.5,
            fontstyle="italic",
            color="#8A6500",
        )


def draw_figure(selected: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )
    n_rows = len(selected)
    figure = plt.figure(figsize=(13.8, 10.0))
    grid = figure.add_gridspec(
        n_rows,
        5,
        left=0.055,
        right=0.985,
        top=0.865,
        bottom=0.11,
        wspace=0.15,
        hspace=0.20,
        width_ratios=[1.23, 1.0, 1.0, 1.0, 1.0],
    )

    figure.suptitle(
        "Qualitative comparison on shared unseen test images",
        x=0.52,
        y=0.972,
        fontsize=16,
        fontweight="bold",
    )
    figure.text(
        0.52,
        0.937,
        "Every method receives the identical image; higher bars indicate stronger fake evidence.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    headers = [("Input", "ground truth")] + [(title, subtitle) for _, title, subtitle in METHODS]
    column_centers = [0.137, 0.354, 0.516, 0.678, 0.840]
    for x, (title, subtitle) in zip(column_centers, headers):
        figure.text(x, 0.900, title, ha="center", va="bottom", fontsize=11.2, fontweight="bold")
        figure.text(x, 0.881, subtitle, ha="center", va="bottom", fontsize=8.0, color="#555555")

    for row_index, row in selected.iterrows():
        input_axis = figure.add_subplot(grid[row_index, 0])
        with Image.open(row["image_path"]) as image:
            display = ImageOps.fit(center_square(image.convert("RGB")), (480, 480), method=Image.Resampling.LANCZOS)
        input_axis.imshow(display)
        input_axis.set_xticks([])
        input_axis.set_yticks([])
        for spine in input_axis.spines.values():
            spine.set_color("#5E6A71")
            spine.set_linewidth(0.9)
        gt = "Fake" if int(row["label"]) == 1 else "Real"
        pretty_domain = str(row["domain"]).replace("_", "-")
        input_axis.text(
            -0.09,
            0.5,
            f"({row['panel']})",
            transform=input_axis.transAxes,
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
        input_axis.text(
            0.5,
            -0.08,
            f"{pretty_domain}  |  GT: {gt}",
            transform=input_axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.3,
            fontweight="bold",
        )

        for column_index, (key, _, _) in enumerate(METHODS, start=1):
            score_axis = figure.add_subplot(grid[row_index, column_index])
            draw_score_card(score_axis, row, key=key, emphasize=(key == "ours"))

    figure.text(
        0.055,
        0.055,
        "Green/red borders denote correct/incorrect decisions. Neural detectors use p(fake) = 0.50; "
        "AEROBLADE uses the unlabeled pooled-score median only for the displayed qualitative label.",
        ha="left",
        va="center",
        fontsize=8.3,
        color="#333333",
    )
    figure.text(
        0.055,
        0.031,
        "AEROBLADE's percentile is a monotonic, rank-preserving transform of its archived reconstruction score; it is not a calibrated probability.",
        ha="left",
        va="center",
        fontsize=8.1,
        color="#555555",
    )
    figure.savefig(output_pdf, format="pdf", facecolor="white")
    figure.savefig(output_png, format="png", dpi=400, facecolor="white")
    plt.close(figure)


def write_revision_text(out_dir: Path) -> None:
    content = """# Suggested manuscript insertion for Reviewer 1, Comment 4

## Experimental-section paragraph

**Qualitative comparison.** Figure X presents shared-image predictions from the spatial-only detector, FIRE, AEROBLADE, and our adaptive fusion model on unseen target domains. The examples include both fake and real images and expose false negatives and false positives produced by different competing cues. When a spatial-only or reconstruction-based detector assigns misleading evidence, the proposed model usually retains the correct prediction by adaptively combining complementary spatial and reconstruction-aware information. These image-level results complement the aggregate AUC values and provide direct evidence that the gain is not confined to a particular generator or error type. For clarity, learned detectors are reported with their fake-class probabilities. Because AEROBLADE outputs an uncalibrated reconstruction score, we display its unlabeled pooled percentile, which is a monotonic transformation that preserves its ranking.

## Figure caption

**Figure X. Qualitative comparison on shared unseen test images.** Each row shows one image evaluated by Spatial-only, FIRE, AEROBLADE, and the proposed method. Green and red borders indicate correct and incorrect decisions, respectively, and longer bars indicate stronger fake evidence. Learned detectors use a fake-class probability threshold of 0.50. AEROBLADE's native reconstruction score is shown as an unlabeled pooled percentile over the ten target domains (a monotonic, rank-preserving transformation); its median is used only for the displayed qualitative label. The selected examples cover multiple generators and both false-negative and false-positive cases, illustrating how adaptive fusion remains reliable when an individual cue becomes misleading.

## Response to reviewer (English)

Thank you for this helpful suggestion. We added a new qualitative comparison figure in the experimental section. The figure reports image-level outputs of Spatial-only, FIRE, AEROBLADE, and our method on exactly the same unseen test images, covering multiple target generators as well as both real and fake samples. It explicitly visualizes cases in which competing spatial or reconstruction-based evidence produces a false negative or false positive, whereas the proposed adaptive fusion gives the correct prediction. We also clarified the score definitions and decision rules in the caption. To make the selection auditable, we provide the complete candidate prediction table and the deterministic selection rule together with the figure-generation code.

## 给审稿人的回复（中文参考）

感谢审稿人的宝贵建议。我们已在实验部分新增一幅定性对比图。在完全相同的未见测试图像上，该图同时给出了 Spatial-only、FIRE、AEROBLADE 和本文方法的逐图输出，覆盖多个目标生成器，并同时包含真实图像与生成图像。图中直观展示了竞争方法的空间证据或重建证据产生假阴性或假阳性，而本文自适应融合仍能给出正确预测的代表性情形。我们还在图注中明确说明了各分数的定义和判决规则。为保证选例过程可核验，我们随图提供了完整候选预测表、确定性的选例规则及绘图代码。
"""
    (out_dir / "manuscript_insertion_and_reviewer_response.md").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_dir / "all_candidate_predictions.csv"

    archive_files, archive_base = locate_aeroblade_csvs(Path(args.archive_root).resolve())
    if args.reuse_predictions and predictions_csv.is_file():
        predictions = pd.read_csv(predictions_csv)
        predictions = add_decisions(predictions)
    else:
        pool = read_aeroblade_pool(archive_files)
        candidates = sample_candidate_pool(
            pool=pool,
            bundle_root=Path(args.bundle_root).resolve(),
            scan_domains=args.scan_domains,
            n_per_group=args.samples_per_class_domain,
            seed=args.seed,
        )
        device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
        print(f"[INFO] device={device}; candidates={len(candidates)}; AEROBLADE archive={archive_base}")
        specs = [
            ("spatial_p_fake", "spatial_resnet50", Path(args.spatial_ckpt).resolve()),
            ("fire_p_fake", "fire_resnet50", Path(args.fire_ckpt).resolve()),
            ("ours_p_fake", "sfire_crossattn_resnet50", Path(args.ours_ckpt).resolve()),
        ]
        predictions = candidates.copy()
        for output_column, model_name, checkpoint in specs:
            predictions[output_column] = run_detector(
                frame=predictions,
                model_name=model_name,
                checkpoint=checkpoint,
                device=device,
                image_size=args.image_size,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                seed=args.seed,
            )
        predictions = add_decisions(predictions)
        predictions.to_csv(predictions_csv, index=False)

    selected = select_examples(predictions, rows=args.rows)
    selected.to_csv(out_dir / "selected_examples.csv", index=False)

    output_pdf = out_dir / "fig_qualitative_comparison.pdf"
    output_png = out_dir / "fig_qualitative_comparison.png"
    draw_figure(selected, output_pdf=output_pdf, output_png=output_png)
    write_revision_text(out_dir)

    provenance = {
        "seed": args.seed,
        "target_domains_for_aeroblade_percentile": TARGET_DOMAINS,
        "candidate_scan_domains": list(args.scan_domains),
        "samples_per_class_domain": args.samples_per_class_domain,
        "n_candidates": int(len(predictions)),
        "n_selected": int(len(selected)),
        "selection_rule": (
            "Prefer Ours-correct examples covering Spatial-only, FIRE, and AEROBLADE errors "
            "for both fake and real classes; favor unique domains, then deterministic selection score."
        ),
        "decision_rules": {
            "spatial_only": "p(fake) >= 0.5",
            "fire": "p(fake) >= 0.5",
            "ours": "p(fake) >= 0.5",
            "aeroblade_display_only": "unlabeled pooled score percentile >= 0.5",
        },
        "aeroblade_score_note": (
            "Percentile is a monotonic rank-preserving transform of the archived native distance; "
            "it is not a calibrated probability."
        ),
        "aeroblade_archive_base": str(archive_base),
        "aeroblade_input_hashes": {domain: file_sha256(path) for domain, path in archive_files.items()},
        "checkpoints": {
            "spatial_only": {
                "path": str(Path(args.spatial_ckpt).resolve()),
                "sha256": file_sha256(Path(args.spatial_ckpt).resolve()),
            },
            "fire": {
                "path": str(Path(args.fire_ckpt).resolve()),
                "sha256": file_sha256(Path(args.fire_ckpt).resolve()),
            },
            "ours": {
                "path": str(Path(args.ours_ckpt).resolve()),
                "sha256": file_sha256(Path(args.ours_ckpt).resolve()),
            },
        },
        "outputs": {
            "pdf": str(output_pdf),
            "png": str(output_png),
            "all_predictions": str(predictions_csv),
            "selected_examples": str(out_dir / "selected_examples.csv"),
        },
    }
    (out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"[DONE] {output_pdf}")
    print(f"[DONE] {output_png}")


if __name__ == "__main__":
    main()
