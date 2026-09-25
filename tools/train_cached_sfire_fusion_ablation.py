"""Train/evaluate capacity-controlled fusion heads on frozen cached evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.models.sfire_cached_fusion import CachedSpatialFIREFusion
from core.utils.metrics import compute_binary_metrics
from core.utils.seed import set_seed


VARIANTS = {
    "concat_linear": "Concat + linear projection",
    "gated": "Conventional gated fusion",
    "concat_mlp_matched": "Capacity-matched concat MLP",
    "standard_crossattn": "Standard cross-attention",
    "full": "Ours: evidence-guided interaction + routing",
}
EXPECTED_PARAMS = {
    "concat_linear": 4_722_690,
    "gated": 5_246_979,
    "concat_mlp_matched": 12_602_377,
    "standard_crossattn": 12_602_884,
    "full": 12_602_884,
}
FLOAT_FIELDS = [
    "spatial_map", "spatial_vec", "spatial_logits", "fire_map",
    "fire_vec", "fire_logit", "anomaly_prior",
]


class FeatureCache:
    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        manifest_path = self.directory / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Incomplete cache: {manifest_path}")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("status") != "complete":
            raise ValueError(f"Cache is not marked complete: {manifest_path}")
        self.arrays = {name: np.load(self.directory / f"{name}.npy", mmap_mode="r") for name in FLOAT_FIELDS}
        self.labels = np.load(self.directory / "label.npy", mmap_mode="r")
        self.size = len(self.labels)
        if any(len(array) != self.size for array in self.arrays.values()):
            raise ValueError(f"Cache arrays have inconsistent lengths: {self.directory}")

    def batch(self, start: int, end: int, device: str):
        # The cache is stored as float16 to save disk space, but all fusion
        # strategies train and evaluate in float32 for identical, stable math.
        # copy=True avoids non-writable memmap warnings.
        values = [
            torch.from_numpy(np.array(self.arrays[name][start:end], copy=True)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
            for name in FLOAT_FIELDS
        ]
        labels = torch.from_numpy(np.array(self.labels[start:end], dtype=np.int64, copy=True)).to(device)
        return (*values, labels)


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    if z1.size(0) <= 1:
        return z1.new_zeros(())
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def loss_for_batch(model, features, labels):
    out = model(*features, return_aux=True)
    main = F.cross_entropy(out["logits"], labels)
    # Branch logits are frozen evidence. The two constants are retained in the
    # reported objective, while only shared projections/fusion receive gradients.
    aux_s = F.cross_entropy(out["spatial_logits"], labels)
    aux_f = F.cross_entropy(out["fire_logits_2c"], labels)
    ctr = contrastive_loss(out["spatial_proj"], out["fire_proj"], temperature=0.07)
    router = 0.02 * out["router_balance"] - 0.005 * out["router_entropy"]
    total = main + 0.3 * aux_s + 0.3 * aux_f + 0.05 * ctr + router
    return total, out


@torch.no_grad()
def evaluate(model, cache: FeatureCache, batch_size: int, device: str):
    model.eval()
    labels, probs = [], []
    for start in range(0, cache.size, batch_size):
        batch = cache.batch(start, min(cache.size, start + batch_size), device)
        features, y = batch[:-1], batch[-1]
        logits = model(*features)
        labels.append(y.cpu().numpy())
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
    return compute_binary_metrics(np.concatenate(labels), np.concatenate(probs), thr=0.5)


def block_order(size: int, batch_size: int, seed: int):
    blocks = list(range(math.ceil(size / batch_size)))
    random.Random(seed).shuffle(blocks)
    return blocks


def train_one(args, variant: str, seed: int, train_cache: FeatureCache, val_cache: FeatureCache):
    run_dir = args.out_dir / f"{variant}_seed{seed}"
    complete_path = run_dir / "training_complete.json"
    if complete_path.is_file() and (run_dir / "best_by_auc.pth").is_file():
        print(f"[SKIP] complete fusion run: {run_dir}", flush=True)
        return json.loads(complete_path.read_text(encoding="utf-8"))
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("CUDA is required for this experiment.")
    model = CachedSpatialFIREFusion(variant, dropout=args.dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if param_count != EXPECTED_PARAMS[variant]:
        raise ValueError(f"Parameter mismatch for {variant}: {param_count} != {EXPECTED_PARAMS[variant]}")
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_dir.mkdir(parents=True, exist_ok=True)
    best_auc = -1.0
    stale = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        progress = (epoch - 1) / max(1, args.epochs - 1)
        lr = args.min_lr + 0.5 * (args.lr - args.min_lr) * (1.0 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = lr
        for block in block_order(train_cache.size, args.batch_size, seed + 1009 * epoch):
            start = block * args.batch_size
            end = min(train_cache.size, start + args.batch_size)
            batch = train_cache.batch(start, end, device)
            features, labels = batch[:-1], batch[-1]
            optimizer.zero_grad(set_to_none=True)
            loss, _out = loss_for_batch(model, features, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss: variant={variant}, seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            count = end - start
            epoch_loss += float(loss.detach().cpu()) * count
            seen += count

        val = evaluate(model, val_cache, args.eval_batch_size, device)
        record = {"epoch": epoch, "loss": epoch_loss / max(1, seen), "lr": lr, **val}
        history.append(record)
        print(f"[{variant} seed={seed}] epoch={epoch}/{args.epochs} loss={record['loss']:.4f} val_auc={val['auc']:.4f}", flush=True)
        if val["auc"] > best_auc:
            best_auc = float(val["auc"])
            stale = 0
            torch.save({
                "model": model.state_dict(), "variant": variant, "seed": seed,
                "fusion_trainable_params": param_count, "best_val": val,
                "fusion_compute_precision": "float32",
                "train_cache_manifest": train_cache.manifest,
                "val_cache_manifest": val_cache.manifest,
            }, run_dir / "best_by_auc.pth")
        else:
            stale += 1
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if args.patience > 0 and stale >= args.patience:
            break

    complete = {
        "status": "complete", "variant": variant, "label": VARIANTS[variant], "seed": seed,
        "fusion_trainable_params": param_count, "best_val_auc": best_auc,
        "fusion_compute_precision": "float32",
        "epochs_completed": len(history), "elapsed_seconds": time.time() - started,
    }
    (run_dir / "training_complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    return complete


def evaluate_tests(args, variants: list[str], seeds: list[int]):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_dirs = sorted(path.parent for path in args.test_cache_root.glob("*/manifest.json"))
    if not cache_dirs:
        raise FileNotFoundError(f"No complete test caches under {args.test_cache_root}")
    rows = []
    for variant in variants:
        for seed in seeds:
            checkpoint = args.out_dir / f"{variant}_seed{seed}" / "best_by_auc.pth"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            model = CachedSpatialFIREFusion(variant, dropout=args.dropout).to(device)
            payload = torch.load(checkpoint, map_location="cpu")
            model.load_state_dict(payload["model"], strict=True)
            for cache_dir in cache_dirs:
                cache = FeatureCache(cache_dir)
                metrics = evaluate(model, cache, args.eval_batch_size, device)
                rows.append({"variant": variant, "seed": seed, "domain": cache_dir.name, **metrics})
                print(f"[EVAL] {variant} seed={seed} {cache_dir.name}: auc={metrics['auc']:.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    with (args.out_dir / "test_domain_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def summarize(args, variants: list[str], seeds: list[int]):
    result_path = args.out_dir / "test_domain_results.csv"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    with result_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary = []
    for variant in variants:
        seed_values = []
        for seed in seeds:
            values = [float(row["auc"]) for row in rows if row["variant"] == variant and int(row["seed"]) == seed]
            if values:
                seed_values.append(float(np.mean(values)))
        summary.append({
            "variant": variant,
            "method": VARIANTS[variant],
            "fusion_trainable_params": EXPECTED_PARAMS[variant],
            "completed_seeds": len(seed_values),
            "mean_auc_percent": float(np.mean(seed_values) * 100.0) if seed_values else "",
            "std_auc_percent": float(np.std(seed_values, ddof=1) * 100.0) if len(seed_values) > 1 else "",
            "seed_average_auc_percent": json.dumps([value * 100.0 for value in seed_values]),
        })
    with (args.out_dir / "paper_capacity_controlled_table.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "summarize"])
    parser.add_argument("--train_cache", default="")
    parser.add_argument("--val_cache", default="")
    parser.add_argument("--test_cache_root", default="")
    parser.add_argument("--out_dir", default=str(ROOT / "paper_assets" / "reviewer1_comment5_fast" / "fusion_runs"))
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()
    args.out_dir = Path(args.out_dir).resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if any(variant not in VARIANTS for variant in variants):
        raise ValueError(f"Unknown variants: {variants}")

    if args.mode == "train":
        train_cache = FeatureCache(Path(args.train_cache))
        val_cache = FeatureCache(Path(args.val_cache))
        results = [train_one(args, variant, seed, train_cache, val_cache) for variant in variants for seed in seeds]
        print(json.dumps(results, ensure_ascii=False, indent=2))
    elif args.mode == "eval":
        args.test_cache_root = Path(args.test_cache_root).resolve()
        evaluate_tests(args, variants, seeds)
    else:
        summarize(args, variants, seeds)


if __name__ == "__main__":
    main()
