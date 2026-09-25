"""Cache frozen Spatial/FIRE branch evidence for fast fusion-only ablations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset.csv_dataset import CSVIndexDataset
from core.dataset.transforms import build_transforms_for_model
from core.models import build_model


FIELDS = {
    "spatial_map": ((2048, 8, 8), np.float16),
    "spatial_vec": ((2048,), np.float16),
    "spatial_logits": ((2,), np.float16),
    "fire_map": ((2048, 8, 8), np.float16),
    "fire_vec": ((2048,), np.float16),
    "fire_logit": ((1,), np.float16),
    "anomaly_prior": ((1, 8, 8), np.float16),
    "label": ((), np.int8),
}


class SeededTransformDataset(Dataset):
    """Apply the manuscript training transform deterministically per sample.

    A fixed transform seed makes the cached evidence exactly reproducible after
    an interrupted run and guarantees that every fusion variant sees identical
    augmented inputs.
    """

    def __init__(self, dataset: Dataset, transform, seed: int):
        self.dataset = dataset
        self.transform = transform
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label, domain = self.dataset[index]
        sample_seed = (self.seed + 1_000_003 * int(index)) % (2**31 - 1)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(sample_seed)
            np.random.seed(sample_seed % (2**32 - 1))
            torch.manual_seed(sample_seed)
            image = self.transform(image)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
        return image, label, domain


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(model, path: Path):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise ValueError(f"Strict checkpoint load failed for {path}: missing={missing}, unexpected={unexpected}")


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def flush_arrays(arrays: dict[str, np.memmap]) -> None:
    for array in arrays.values():
        array.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_csv", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--spatial_ckpt", required=True)
    parser.add_argument("--fire_ckpt", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--transform_mode", choices=["eval", "train_fixed"], default="eval")
    parser.add_argument("--transform_seed", type=int, default=42)
    parser.add_argument("--checkpoint_every_batches", type=int, default=100)
    parser.add_argument("--cooldown_ms", type=float, default=0.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for branch feature caching.")
    index_csv = Path(args.index_csv).resolve()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    spatial_ckpt = Path(args.spatial_ckpt).resolve()
    fire_ckpt = Path(args.fire_ckpt).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("FIRE_VAE_DIR", str(ROOT / "pretrained" / "sd15_vae"))
    use_train_transform = args.transform_mode == "train_fixed"
    transform = build_transforms_for_model(
        "sfire_crossattn_resnet50", image_size=args.image_size, is_train=use_train_transform
    )
    if use_train_transform:
        base_dataset = CSVIndexDataset(
            str(index_csv), str(data_root), split=args.split, transform=None, missing_policy="strict"
        )
        dataset = SeededTransformDataset(base_dataset, transform, args.transform_seed)
        metadata_df = base_dataset.df
    else:
        dataset = CSVIndexDataset(
            str(index_csv), str(data_root), split=args.split, transform=transform, missing_policy="strict"
        )
        metadata_df = dataset.df
    n = len(dataset)
    field_specs = {
        name: {"shape": [n, *shape], "dtype": str(np.dtype(dtype))}
        for name, (shape, dtype) in FIELDS.items()
    }
    signature = {
        "index_csv": str(index_csv),
        "index_sha256": sha256(index_csv),
        "data_root": str(data_root),
        "split": args.split,
        "samples": n,
        "image_size": args.image_size,
        "transform_mode": args.transform_mode,
        "transform_seed": args.transform_seed if use_train_transform else None,
        "spatial_ckpt": str(spatial_ckpt),
        "spatial_ckpt_sha256": sha256(spatial_ckpt),
        "fire_ckpt": str(fire_ckpt),
        "fire_ckpt_sha256": sha256(fire_ckpt),
        "fields": field_specs,
    }
    progress_path = out_dir / "progress.json"
    cache_files = [out_dir / f"{name}.npy" for name in FIELDS]
    existing_entries = list(out_dir.iterdir())
    if existing_entries:
        if not progress_path.is_file():
            raise FileExistsError(
                f"Non-empty cache directory has no resumable progress file: {out_dir}. "
                "Archive it before restarting."
            )
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("signature") != signature:
            raise ValueError(f"Cache resume signature mismatch: {out_dir}")
        cursor = int(progress.get("cursor", 0))
        if not 0 <= cursor <= n:
            raise ValueError(f"Invalid resume cursor {cursor} for {n} samples")
        if any(not path.is_file() for path in cache_files):
            raise FileNotFoundError(f"One or more resumable cache arrays are missing: {out_dir}")
        arrays = {name: open_memmap(out_dir / f"{name}.npy", mode="r+") for name in FIELDS}
        for name, array in arrays.items():
            expected_shape = tuple(field_specs[name]["shape"])
            expected_dtype = np.dtype(field_specs[name]["dtype"])
            if array.shape != expected_shape or array.dtype != expected_dtype:
                raise ValueError(
                    f"Invalid resumable array {name}: shape={array.shape}, dtype={array.dtype}; "
                    f"expected shape={expected_shape}, dtype={expected_dtype}"
                )
        print(f"[RESUME] {args.split} cache continues at sample {cursor}/{n}", flush=True)
    else:
        arrays = {
            name: open_memmap(out_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=(n, *shape))
            for name, (shape, dtype) in FIELDS.items()
        }
        cursor = 0
        write_json_atomic(progress_path, {"status": "running", "cursor": 0, "signature": signature})

    remaining_dataset = Subset(dataset, range(cursor, n))
    loader = DataLoader(
        remaining_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    spatial = build_model("spatial_resnet50", num_classes=2, pretrained=False, dropout=0.2)
    fire = build_model("fire_resnet50", num_classes=1, pretrained=False, dropout=0.0)
    load_checkpoint(spatial, spatial_ckpt)
    load_checkpoint(fire, fire_ckpt)
    spatial.cuda().eval()
    fire.cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    checkpoint_every = max(1, int(args.checkpoint_every_batches))
    batch_count = 0
    with torch.inference_mode():
        for x, y, _domain in tqdm(loader, desc=f"cache {args.split} from {cursor}", unit="batch"):
            x = x.cuda(non_blocking=True)
            # FIRE samples a latent reconstruction. Tie that sample to the
            # absolute cache cursor so a resumed run is byte-reproducible.
            if use_train_transform:
                torch.manual_seed(args.transform_seed + cursor)
                torch.cuda.manual_seed_all(args.transform_seed + cursor)
            s = spatial.extract_features((x - mean) / std, return_logits=True)
            f = fire.extract_features(x, return_aux=False)
            prior = F.interpolate(f["anomaly_prior"], size=s["feature_map"].shape[-2:], mode="bilinear", align_corners=False)
            batch = x.size(0)
            end = cursor + batch
            values = {
                "spatial_map": s["feature_map"],
                "spatial_vec": s["global_feature"],
                "spatial_logits": s["logits"],
                "fire_map": f["feature_map"],
                "fire_vec": f["global_feature"],
                "fire_logit": f["logits"].view(-1, 1),
                "anomaly_prior": prior,
            }
            for name, value in values.items():
                actual = tuple(value.shape[1:])
                expected = FIELDS[name][0]
                if actual != expected:
                    raise ValueError(f"Unexpected {name} shape {actual}; expected {expected}")
                arrays[name][cursor:end] = value.detach().cpu().numpy().astype(FIELDS[name][1], copy=False)
            arrays["label"][cursor:end] = y.numpy().astype(np.int8, copy=False)
            cursor = end
            batch_count += 1
            if batch_count % checkpoint_every == 0:
                flush_arrays(arrays)
                write_json_atomic(
                    progress_path,
                    {"status": "running", "cursor": cursor, "signature": signature},
                )
            if args.cooldown_ms > 0:
                time.sleep(float(args.cooldown_ms) / 1000.0)
    if cursor != n:
        raise RuntimeError(f"Cached {cursor} rows, expected {n}")
    flush_arrays(arrays)
    write_json_atomic(progress_path, {"status": "complete", "cursor": cursor, "signature": signature})

    meta_rows = metadata_df[["path", "label", "domain", "split"]].to_dict("records")
    with (out_dir / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "domain", "split"])
        writer.writeheader()
        writer.writerows(meta_rows)
    manifest = {
        "status": "complete",
        **signature,
        "transform": (
            "fixed seeded manuscript training augmentation"
            if use_train_transform else "dual-stream deterministic evaluation transform"
        ),
        "storage_note": "float16 cache is shared byte-identically by every fusion strategy and seed",
        "resume_note": f"progress flushed every {checkpoint_every} batches; cooldown={args.cooldown_ms} ms",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
