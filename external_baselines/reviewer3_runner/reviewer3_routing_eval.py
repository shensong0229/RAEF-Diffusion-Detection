"""Export per-image routing evidence from a trained full-model checkpoint.

This is inference-only analysis; the checkpoint's training source is recorded.
Resumes from verified CSV prefixes after interruption. Batch size is fixed at
one because the FIRE frequency preprocessing uses batch-wide extrema.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from reviewer3_test_time_ablation import (
    DOMAINS, PROJECT, CSVIndexDataset, build_model_from_checkpoint,
    build_transforms_for_model, sha256, write_json,
)


FIELDS = (
    "domain", "path", "label", "prob_full", "prob_spatial", "prob_fire",
    "weight_spatial", "weight_fire", "entropy_spatial", "entropy_fire",
    "prior_mean", "prior_std",
)


def binary_entropy(probability: torch.Tensor) -> torch.Tensor:
    p = probability.clamp(1e-7, 1 - 1e-7)
    return -p * torch.log(p) - (1 - p) * torch.log1p(-p)


def read_prefix(path: Path, expected_paths: list[str], labels: list[int], domain: str) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"Unexpected CSV columns in {path}")
        count = 0
        for row in reader:
            if count >= len(expected_paths) or row["domain"] != domain:
                raise ValueError(f"Out-of-range or wrong-domain row {count} in {path}")
            if row["path"] != expected_paths[count] or int(row["label"]) != labels[count]:
                raise ValueError(f"Path/label mismatch at row {count} in {path}")
            values = np.asarray([float(row[k]) for k in FIELDS[3:]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite result at row {count} in {path}")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    bundle = PROJECT / "exports" / "dragon_eval_25domains_unique_real_1k_bundle"
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints" / "sfire_best" / "best_by_auc.pth")
    parser.add_argument("--csv-dir", type=Path, default=bundle / "csvs")
    parser.add_argument("--data-root", type=Path, default=bundle / "images")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "reviewer3_routing_evaluation")
    parser.add_argument(
        "--checkpoint-protocol", required=True,
        help="Training source of the supplied checkpoint, e.g. SDV5-source; saved in provenance.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-images", type=int, default=0, help="Smoke test only; zero means all 20,000 images")
    parser.add_argument("--domains", default=",".join(DOMAINS))
    args = parser.parse_args()
    if args.workers < 0 or args.max_images < 0:
        raise ValueError("workers and max-images must be nonnegative")
    domains = [item.strip() for item in args.domains.split(",") if item.strip()]
    if not domains or len(set(domains)) != len(domains) or any(domain not in DOMAINS for domain in domains):
        raise ValueError("Use unique domains from the fixed ten-domain set")
    if not args.checkpoint.is_file() or not args.csv_dir.is_dir():
        raise FileNotFoundError("Checkpoint or index directory is missing")

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Full VAE inference requires the intended CUDA environment")
    provenance = {
        "analysis": "inference-only sample-wise routing analysis of one fixed checkpoint",
        "checkpoint_protocol": args.checkpoint_protocol,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "csv_dir": str(args.csv_dir.resolve()),
        "data_root": str(args.data_root.resolve()),
        "domains": domains,
        "max_images": args.max_images,
        "batch_size": 1,
        "workers": args.workers,
        "threshold": 0.5,
        "gpu": torch.cuda.get_device_name(0),
    }
    provenance_path = out / "provenance.json"
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        for key in ("checkpoint_sha256", "checkpoint_protocol", "csv_dir", "data_root", "domains", "max_images", "batch_size"):
            if previous.get(key) != provenance[key]:
                raise ValueError(f"Resume provenance changed for {key}")
    else:
        write_json(provenance_path, provenance)

    torch.backends.cudnn.benchmark = True
    model = build_model_from_checkpoint(args.checkpoint, device)
    started = time.time()
    complete = 0
    for domain in domains:
        source_csv = args.csv_dir / f"{domain}.csv"
        dataset = CSVIndexDataset(
            str(source_csv), str(args.data_root), split="test",
            transform=build_transforms_for_model("sfire_crossattn_resnet50", 256, False),
            missing_policy="strict",
        )
        limit = min(len(dataset), args.max_images) if args.max_images else len(dataset)
        if not args.max_images and limit != 2000:
            raise ValueError(f"Expected 2,000 images in {domain}; found {limit}")
        paths = dataset.df["path"].astype(str).tolist()[:limit]
        labels = dataset.df["label"].astype(int).tolist()[:limit]
        prediction_path = out / f"{domain}_predictions.csv"
        offset = read_prefix(prediction_path, paths, labels, domain)
        print(f"[DOMAIN] {domain} {offset}/{limit}", flush=True)
        if offset < limit:
            loader = DataLoader(
                Subset(dataset, range(offset, limit)), batch_size=1, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            with prediction_path.open("a", encoding="utf-8-sig" if offset == 0 else "utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                if offset == 0:
                    writer.writeheader()
                seen = offset
                with torch.inference_mode():
                    for images, labels_batch, _ in loader:
                        images = images.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
                        try:
                            aux = model(images, return_aux=True)
                            pf = torch.softmax(aux["logits"].float(), dim=1)[:, 1]
                            ps = torch.softmax(aux["spatial_logits"].float(), dim=1)[:, 1]
                            pr = torch.softmax(aux["fire_logits_2c"].float(), dim=1)[:, 1]
                            weights = aux["router_weights"].float()
                            values = (
                                pf, ps, pr, weights[:, 0], weights[:, 1],
                                binary_entropy(ps), binary_entropy(pr),
                                aux["prior_mean"].float().flatten(),
                                aux["prior_std"].float().flatten(),
                            )
                            scalars = [float(value.item()) for value in values]
                            if not np.isfinite(scalars).all():
                                raise FloatingPointError("Nonfinite model output")
                        except Exception as exc:
                            raise RuntimeError(f"Inference failed at {domain} row {seen}: {paths[seen]}") from exc
                        row = {"domain": domain, "path": paths[seen], "label": int(labels_batch[0])}
                        row.update({key: format(value, ".10g") for key, value in zip(FIELDS[3:], scalars)})
                        writer.writerow(row)
                        handle.flush()
                        seen += 1
                        if seen % 20 == 0 or seen == limit:
                            write_json(out / "progress.json", {
                                "status": "running", "domain": domain, "domain_seen": seen,
                                "domain_total": limit, "domains_complete": complete,
                                "domains_total": len(domains), "elapsed_seconds": round(time.time() - started, 1),
                            })
                            print(f"[PROGRESS] {domain} {seen}/{limit}", flush=True)
        if read_prefix(prediction_path, paths, labels, domain) != limit:
            raise RuntimeError(f"Incomplete predictions for {domain}")
        complete += 1
        print(f"[COMPLETE] {domain}", flush=True)
    write_json(out / "progress.json", {
        "status": "complete", "domains_complete": complete,
        "domains_total": len(domains), "elapsed_seconds": round(time.time() - started, 1),
    })
    print("[COMPLETE] all domains", flush=True)


if __name__ == "__main__":
    main()
