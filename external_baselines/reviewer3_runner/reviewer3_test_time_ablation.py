"""Reviewer 3 test-time interventions on one trained full-model checkpoint.

This is NOT retraining. Every image shares exactly one spatial/FIRE feature
extraction, and all three predictions use the same trained model weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("FIRE_VAE_DIR", str(PROJECT / "pretrained" / "sd15_vae"))

from core.dataset.csv_dataset import CSVIndexDataset
from core.dataset.transforms import build_transforms_for_model
from core.models.builder import build_model
from core.utils.metrics import compute_binary_metrics


DOMAINS = (
    "Flash_PixArt", "Flash_SD3", "Flux_1", "JuggernautXL", "Kolors",
    "Lumina", "PixArt_Alpha", "SDXL_Lightning", "SDXL", "SSD_1B",
)
MODES = ("full", "no_anomaly_guidance", "no_reliability_routing")
FIELDS = ("domain", "path", "label", "prob_full", "prob_no_anomaly_guidance",
          "prob_no_reliability_routing")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_existing(csv_path: Path, expected_paths: list[str], domain: str) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected columns in {csv_path}")
        rows = list(reader)
    if len(rows) > len(expected_paths):
        raise ValueError(f"More prediction rows than test images: {csv_path}")
    for index, row in enumerate(rows):
        if row["domain"] != domain or row["path"] != expected_paths[index]:
            raise ValueError(f"Prediction path/order mismatch at row {index}: {csv_path}")
    return rows


def build_model_from_checkpoint(checkpoint: Path, device: str):
    model = build_model("sfire_crossattn_resnet50", num_classes=2,
                        pretrained=False, dropout=0.1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if device == "cuda":
        model.to(memory_format=torch.channels_last)
    return model


def three_predictions(model, images: torch.Tensor) -> dict[str, np.ndarray]:
    """Re-use the same branch outputs; intervene only in fusion computation."""
    original_spatial = model._extract_spatial_features
    original_fire = model.fire_branch.extract_features
    cache = {}

    def spatial_once(x):
        if "spatial" not in cache:
            cache["spatial"] = original_spatial(x)
        return cache["spatial"]

    def fire_once(x, return_aux=False):
        if "fire" not in cache:
            cache["fire"] = original_fire(x, return_aux=True)
        return cache["fire"]

    model._extract_spatial_features = spatial_once
    model.fire_branch.extract_features = fire_once
    try:
        model.fusion_variant = "full"
        full = model(images, return_aux=True)["logits"]

        # The regular cross-attention uses identical learned parameters but
        # does not inject the anomaly map into attention. Its router still
        # receives prior summary statistics, so the label is deliberately
        # "no anomaly guidance", not "no anomaly prior anywhere".
        model.fusion_variant = "standard_crossattn"
        without_guidance = model(images, return_aux=True)["logits"]

        # Keep the trained router and all model parameters in place, but
        # replace its per-image output by equal 0.5/0.5 branch weights.
        model.fusion_variant = "full"
        hook = model.router.register_forward_hook(
            lambda _module, _inputs, output: torch.full_like(output, 0.5)
        )
        try:
            without_routing = model(images, return_aux=True)["logits"]
        finally:
            hook.remove()
    finally:
        model.fusion_variant = "full"
        model._extract_spatial_features = original_spatial
        model.fire_branch.extract_features = original_fire

    outputs = {
        "full": full,
        "no_anomaly_guidance": without_guidance,
        "no_reliability_routing": without_routing,
    }
    probabilities = {}
    for mode, logits in outputs.items():
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Non-finite logits in {mode}")
        probabilities[mode] = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
    return probabilities


def summarize_domain(domain: str, rows: list[dict]) -> list[dict]:
    truth = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    result = []
    for mode in MODES:
        scores = np.asarray([float(row[f"prob_{mode}"]) for row in rows], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Non-finite saved probabilities in {domain}/{mode}")
        metric = compute_binary_metrics(truth, scores, thr=0.5)
        result.append({"domain": domain, "mode": mode, "n": len(rows), **metric})
    return result


def write_metrics(out_dir: Path, records: list[dict]) -> None:
    with (out_dir / "metrics_by_domain.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        columns = ("domain", "mode", "n", "auc", "ap", "acc", "f1")
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    summary = {mode: {} for mode in MODES}
    for mode in MODES:
        selected = [r for r in records if r["mode"] == mode]
        if selected:
            summary[mode] = {
                "domains_complete": len(selected),
                "macro_auc": float(np.mean([r["auc"] for r in selected])),
                "macro_ap": float(np.mean([r["ap"] for r in selected])),
                "macro_acc": float(np.mean([r["acc"] for r in selected])),
                "macro_f1": float(np.mean([r["f1"] for r in selected])),
            }
    write_json(out_dir / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints" / "sfire_best" / "best_by_auc.pth")
    bundle = PROJECT / "exports" / "dragon_eval_25domains_unique_real_1k_bundle"
    parser.add_argument("--csv-dir", type=Path, default=bundle / "csvs")
    parser.add_argument("--data-root", type=Path, default=bundle / "images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "reviewer3_test_time_ablation")
    # The FIRE frequency image uses a batch-wide min/max denominator, so
    # batch 1 keeps each image prediction independent of its paired sample.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke test only; omit for full evaluation.")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    parser.add_argument(
        "--checkpoint-protocol", required=True,
        help="Training source of the supplied checkpoint, e.g. SDV5-source; saved in provenance.",
    )
    args = parser.parse_args()

    if args.batch_size < 1 or args.workers < 0:
        raise ValueError("batch-size must be positive and workers nonnegative")
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    if not domains or any(domain not in DOMAINS for domain in domains):
        raise ValueError(f"Use domains from the fixed ten-domain set: {DOMAINS}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.csv_dir.is_dir():
        raise NotADirectoryError(args.csv_dir)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = device == "cuda"
    model = build_model_from_checkpoint(args.checkpoint, device)
    provenance = {
        "analysis": "test-time interventions on one fixed full-model checkpoint; no retraining",
        "checkpoint_protocol": args.checkpoint_protocol,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "csv_dir": str(args.csv_dir.resolve()),
        "data_root": str(args.data_root.resolve()),
        "domains": domains,
        "modes": {
            "full": "unchanged trained full model",
            "no_anomaly_guidance": "disable anomaly-map injection in cross-attention only; router still sees prior summaries",
            "no_reliability_routing": "replace router output by 0.5/0.5 equal branch weights",
        },
        "batch_size": args.batch_size,
        "workers": args.workers,
        "max_images": args.max_images,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "precision": "FP32 fusion; FIRE VAE retains checkpoint implementation dtype",
    }
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "checkpoint_protocol", "csv_dir", "data_root", "domains", "max_images"):
            if previous.get(key) != provenance[key]:
                raise ValueError(f"Resume provenance changed for {key}")
    else:
        write_json(provenance_path, provenance)

    metrics = []
    started = time.time()
    for domain in domains:
        source_csv = args.csv_dir / f"{domain}.csv"
        if not source_csv.is_file():
            raise FileNotFoundError(source_csv)
        dataset = CSVIndexDataset(str(source_csv), str(args.data_root), split="test",
                                  transform=build_transforms_for_model("sfire_crossattn_resnet50", 256, False),
                                  missing_policy="strict")
        limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
        if not args.max_images and limit != 2000:
            raise ValueError(f"Expected 2000 images for {domain}, found {limit}")
        paths = dataset.df["path"].astype(str).tolist()[:limit]
        labels = dataset.df["label"].astype(int).tolist()[:limit]
        prediction_path = out / f"{domain}_predictions.csv"
        existing = read_existing(prediction_path, paths, domain)
        for index, row in enumerate(existing):
            if int(row["label"]) != labels[index]:
                raise ValueError(f"Saved label mismatch for {domain} row {index}")
        offset = len(existing)
        print(f"[DOMAIN] {domain} {offset}/{limit}", flush=True)
        if offset < limit:
            subset = Subset(dataset, range(offset, limit))
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, pin_memory=device == "cuda")
            with prediction_path.open("a", encoding="utf-8-sig" if offset == 0 else "utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if offset == 0:
                    writer.writeheader()
                seen = offset
                with torch.inference_mode():
                    for images, labels_batch, _ in loader:
                        images = images.to(device, non_blocking=True)
                        if device == "cuda":
                            images = images.contiguous(memory_format=torch.channels_last)
                        try:
                            predictions = three_predictions(model, images)
                        except Exception as exc:
                            current_paths = paths[seen:seen + int(labels_batch.numel())]
                            raise RuntimeError(
                                f"Inference failed at {domain} rows {seen}:{seen + len(current_paths)} "
                                f"paths={current_paths}"
                            ) from exc
                        count = int(labels_batch.numel())
                        for j in range(count):
                            row = {"domain": domain, "path": paths[seen + j],
                                   "label": int(labels_batch[j])}
                            for mode in MODES:
                                row[f"prob_{mode}"] = format(float(predictions[mode][j]), ".10g")
                            writer.writerow(row)
                        seen += count
                        handle.flush()
                        if seen % 20 == 0 or seen == limit:
                            write_json(out / "progress.json", {
                                "status": "running", "domain": domain, "domain_seen": seen,
                                "domain_total": limit, "domains_complete": len(metrics) // len(MODES),
                                "domains_total": len(domains), "elapsed_seconds": round(time.time() - started, 1),
                            })
                            print(f"[PROGRESS] {domain} {seen}/{limit}", flush=True)
        rows = read_existing(prediction_path, paths, domain)
        if len(rows) != limit:
            raise RuntimeError(f"Incomplete prediction file for {domain}: {len(rows)}/{limit}")
        metrics.extend(summarize_domain(domain, rows))
        write_metrics(out, metrics)
        print(f"[COMPLETE] {domain}", flush=True)
    write_json(out / "progress.json", {
        "status": "complete", "domains_complete": len(domains),
        "domains_total": len(domains), "elapsed_seconds": round(time.time() - started, 1),
    })
    print("[COMPLETE] all domains", flush=True)


if __name__ == "__main__":
    main()
