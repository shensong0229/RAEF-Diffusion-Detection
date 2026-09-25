"""Evaluate the official SPAI checkpoint on the manuscript's ten test domains.

This adapter does not modify SPAI.  It only converts the project's existing
CSV schema to SPAI's schema, runs the official model, and writes per-image
scores plus per-domain metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


DOMAIN_FILES = [
    "Flash_PixArt.csv",
    "Flash_SD3.csv",
    "JuggernautXL.csv",
    "Lumina.csv",
    "Flux_1.csv",
    "PixArt_Alpha.csv",
    "SDXL.csv",
    "SDXL_Lightning.csv",
    "Kolors.csv",
    "SSD_1B.csv",
]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(content, encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    # Some Windows editors hold a replace/delete lock while still permitting
    # writes to the existing file.  Falling back here keeps inference alive.
    path.write_text(content, encoding="utf-8")
    tmp.unlink(missing_ok=True)


def prepare_spai_csvs(source_dir: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    for name in DOMAIN_FILES:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        frame = pd.read_csv(src)
        required = {"path", "label", "domain", "split"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{src} is missing columns: {sorted(missing)}")
        out = pd.DataFrame(
            {
                "image": frame["path"].astype(str),
                "class": frame["label"].astype(int),
                "split": "test",
                "domain": frame["domain"].astype(str),
                "original_path": frame["path"].astype(str),
            }
        )
        dst = output_dir / name
        out.to_csv(dst, index=False, quoting=csv.QUOTE_MINIMAL)
        converted.append(dst)
    return converted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--spai-root", type=Path,
                   default=Path(__file__).resolve().parents[2] / "external_baselines" / "spai")
    p.add_argument("--feature-batch", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--smoke", action="store_true",
                   help="Run only two real and two fake images per domain.")
    p.add_argument("--resume", action="store_true",
                   help="Skip domains whose prediction CSV already contains every sample.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    spai_root = args.spai_root.resolve()
    output_name = "spai_smoke" if args.smoke else "spai_official"
    output_root = project_root / "results" / "reviewer3_baselines" / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    log_path = output_root / "run.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    logger = logging.getLogger("spai_eval")

    # Import the untouched official implementation after the caller has put
    # the repository and its isolated vendor directory on PYTHONPATH.
    from spai.config import get_config
    from spai.data import build_loader_test
    from spai.models import build_cls_model
    from spai.utils import load_pretrained

    source_csv_dir = (
        project_root / "exports" / "dragon_eval_25domains_unique_real_1k_bundle" / "csvs"
    )
    image_root = (
        project_root / "exports" / "dragon_eval_25domains_unique_real_1k_bundle" / "images"
    )
    converted_dir = output_root / "input_csvs"
    csv_paths = prepare_spai_csvs(source_csv_dir, converted_dir)

    if args.smoke:
        smoke_dir = output_root / "smoke_csvs"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        smoke_paths: list[Path] = []
        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            frame = pd.concat(
                [frame[frame["class"] == 0].head(2), frame[frame["class"] == 1].head(2)],
                ignore_index=True,
            )
            smoke_path = smoke_dir / csv_path.name
            frame.to_csv(smoke_path, index=False)
            smoke_paths.append(smoke_path)
        csv_paths = smoke_paths

    expected_per_domain = 4 if args.smoke else 2000
    expected_per_class = 2 if args.smoke else 1000
    expected_total = expected_per_domain * len(csv_paths)

    for csv_path in csv_paths:
        frame = pd.read_csv(csv_path)
        if (len(frame) != expected_per_domain or
                frame["class"].value_counts().to_dict() != {0: expected_per_class,
                                                              1: expected_per_class}):
            raise RuntimeError(f"Unexpected test composition in {csv_path}")
        # The bundle was created by the project's verified export tool.  A
        # representative check avoids 20,000 slow Windows metadata probes
        # before inference while still catching a wrong root directory.
        probe_paths = pd.concat([frame.head(2), frame.tail(2)])["image"].tolist()
        missing = [p for p in probe_paths if not (image_root / p).is_file()]
        if missing:
            raise FileNotFoundError(f"{csv_path}: {len(missing)} missing images; first={missing[0]}")

    opts = [
        ("MODEL.PATCH_VIT.MINIMUM_PATCHES", "4"),
        ("MODEL.FEATURE_EXTRACTION_BATCH", str(args.feature_batch)),
        ("DATA.NUM_WORKERS", str(args.workers)),
        ("DATA.TEST_PREFETCH_FACTOR", "1"),
    ]
    cfg = get_config(
        {
            "cfg": str(spai_root / "configs" / "spai.yaml"),
            "batch_size": 1,
            "test_csv": [str(p) for p in csv_paths],
            "test_csv_root": [str(image_root)],
            "pretrained": str(spai_root / "weights" / "spai.pth"),
            "output": str(output_root),
            "tag": "ten_domains",
            "opts": opts,
        }
    )

    start_time = time.time()
    atomic_json(
        progress_path,
        {
            "status": "loading_model",
            "method": "SPAI official checkpoint",
            "domains_total": len(csv_paths),
            "domains_complete": 0,
            "images_total": expected_total,
            "images_complete": 0,
            "feature_batch": args.feature_batch,
        },
    )

    names, datasets, _loaders = build_loader_test(cfg, logger, split="test")
    # The official builder always passes prefetch_factor.  PyTorch rejects
    # prefetch_factor when num_workers=0, so construct equivalent in-process
    # loaders here.  This avoids repeatedly spawning slow Windows workers and
    # does not alter the model, transforms, samples, or predictions.
    from spai.data.data_finetune import image_enlisting_collate_fn
    loaders = [
        torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=image_enlisting_collate_fn,
        )
        for dataset in datasets
    ]
    model = build_cls_model(cfg).cuda()
    load_pretrained(cfg, model, logger, checkpoint_path=spai_root / "weights" / "spai.pth")
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    rows: list[dict] = []
    completed_images = 0
    for domain_index, (name, dataset, loader, input_csv) in enumerate(
        zip(names, datasets, loaders, csv_paths), start=1
    ):
        predictions_path = output_root / f"{name}_predictions.csv"
        if args.resume and predictions_path.exists():
            existing = pd.read_csv(predictions_path)
            if len(existing) == len(dataset) and "spai_score" in existing.columns:
                y_true = existing["class"].to_numpy(dtype=int)
                scores = existing["spai_score"].to_numpy(dtype=float)
                row = {
                    "domain": name,
                    "images": len(existing),
                    "auc": float(roc_auc_score(y_true, scores)),
                    "ap": float(average_precision_score(y_true, scores)),
                    "accuracy": float(accuracy_score(y_true, scores >= 0.5)),
                }
                rows.append(row)
                completed_images += len(existing)
                logger.info("Resume: %s already complete", name)
                continue

        logger.info("Starting domain %d/%d: %s", domain_index, len(loaders), name)
        score_by_index: dict[int, float] = {}
        domain_start = time.time()
        with torch.inference_mode():
            for batch_index, (images, _target, dataset_idx) in enumerate(loader, start=1):
                if isinstance(images, list):
                    images = [image.cuda(non_blocking=True).squeeze(dim=1) for image in images]
                    logits = model(images, cfg.MODEL.FEATURE_EXTRACTION_BATCH)
                else:
                    images = images.cuda(non_blocking=True).squeeze(dim=1)
                    logits = model(images)
                scores = torch.sigmoid(logits).reshape(-1).detach().cpu().tolist()
                indices = dataset_idx.reshape(-1).detach().cpu().tolist()
                score_by_index.update({int(i): float(s) for i, s in zip(indices, scores)})

                if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
                    current_complete = len(score_by_index)
                    elapsed = time.time() - start_time
                    total_complete = completed_images + current_complete
                    rate = total_complete / elapsed if elapsed > 0 else 0.0
                    eta = (expected_total - total_complete) / rate if rate > 0 else None
                    atomic_json(
                        progress_path,
                        {
                            "status": "running",
                            "method": "SPAI official checkpoint",
                            "current_domain": name,
                            "domain_index": domain_index,
                            "domains_total": len(loaders),
                            "domains_complete": domain_index - 1,
                            "domain_images_complete": current_complete,
                            "domain_images_total": len(dataset),
                            "images_complete": total_complete,
                            "images_total": expected_total,
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": None if eta is None else round(eta, 1),
                        },
                    )
                    logger.info(
                        "%s | %d/%d images | overall %d/%d",
                        name, current_complete, len(dataset), total_complete, expected_total,
                    )

        if len(score_by_index) != len(dataset):
            raise RuntimeError(f"{name}: got {len(score_by_index)} scores for {len(dataset)} images")
        frame = pd.read_csv(input_csv)
        frame["spai_score"] = [score_by_index[i] for i in range(len(frame))]
        frame.to_csv(predictions_path, index=False)
        y_true = frame["class"].to_numpy(dtype=int)
        scores_np = frame["spai_score"].to_numpy(dtype=float)
        row = {
            "domain": name,
            "images": len(frame),
            "auc": float(roc_auc_score(y_true, scores_np)),
            "ap": float(average_precision_score(y_true, scores_np)),
            "accuracy": float(accuracy_score(y_true, scores_np >= 0.5)),
            "seconds": round(time.time() - domain_start, 2),
        }
        rows.append(row)
        completed_images += len(frame)
        pd.DataFrame(rows).to_csv(output_root / "metrics_by_domain.csv", index=False)
        logger.info("Completed %s | AUC %.4f | AP %.4f | ACC %.4f", name,
                    row["auc"], row["ap"], row["accuracy"])

    metrics = pd.DataFrame(rows)
    macro = {
        "domain": "Macro average",
        "images": int(metrics["images"].sum()),
        "auc": float(metrics["auc"].mean()),
        "ap": float(metrics["ap"].mean()),
        "accuracy": float(metrics["accuracy"].mean()),
        "seconds": float(metrics.get("seconds", pd.Series(dtype=float)).sum()),
    }
    metrics = pd.concat([metrics, pd.DataFrame([macro])], ignore_index=True)
    metrics.to_csv(output_root / "metrics_by_domain.csv", index=False)
    summary = {
        "status": "complete",
        "method": "SPAI",
        "implementation": "official code and official checkpoint",
        "checkpoint": str(spai_root / "weights" / "spai.pth"),
        "protocol": ("smoke test, 2 real + 2 fake per domain" if args.smoke else
                     "fixed 10-domain test, 1000 real + 1000 fake per domain"),
        "macro_auc": macro["auc"],
        "macro_ap": macro["ap"],
        "macro_accuracy": macro["accuracy"],
        "elapsed_seconds": round(time.time() - start_time, 1),
    }
    atomic_json(output_root / "summary.json", summary)
    atomic_json(
        progress_path,
        {
            **summary,
            "domains_complete": 10,
            "domains_total": 10,
            "images_complete": expected_total,
            "images_total": expected_total,
        },
    )
    logger.info("All SPAI domains complete | macro AUC %.4f", macro["auc"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        output_name = "spai_smoke" if "--smoke" in sys.argv else "spai_official"
        root = Path(__file__).resolve().parents[2] / "results" / "reviewer3_baselines" / output_name
        atomic_json(root / "progress.json", {"status": "failed", "error": repr(exc)})
        raise
