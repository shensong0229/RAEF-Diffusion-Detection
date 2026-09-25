"""Build the deterministic SDV5 single-source train/validation protocol.

The manuscript protocol uses 100,000 real + 100,000 SDV5 images for training
and 10,000 + 10,000 for source validation.  The local SDV5 collection has
8,000 images per class in its existing validation split, so this builder adds
2,000 per class from the local training pool and then samples 100,000 per
class from the remaining files.  Files are never moved or modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def scan_nonempty_files(data_root: Path, top_level: str = "sdv5") -> tuple[set[str], set[str]]:
    """Return non-empty and zero-byte paths using cached scandir metadata."""
    scan_root = data_root / top_level
    usable: set[str] = set()
    zero_byte: set[str] = set()
    stack = [scan_root]
    while stack:
        directory = stack.pop()
        if not directory.is_dir():
            continue
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    rel = Path(entry.path).relative_to(data_root).as_posix()
                    if entry.stat(follow_symlinks=False).st_size > 0:
                        usable.add(rel)
                    else:
                        zero_byte.add(rel)
    return usable, zero_byte


def decode_error(path: Path) -> str | None:
    try:
        with Image.open(path) as image:
            image.load()
            image.convert("RGB")
        return None
    except Exception as exc:
        return f"{path}\t{type(exc).__name__}\t{exc}"


def verify_decodable(data_root: Path, rows: list[dict], workers: int) -> list[str]:
    paths = [data_root / str(row["path"]) for row in rows]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = pool.map(decode_error, paths, chunksize=64)
        return [result for result in results if result]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_csv", default="")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--out_dir", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_per_class", type=int, default=100000)
    parser.add_argument("--val_per_class", type=int, default=10000)
    parser.add_argument("--verify_all", action="store_true")
    parser.add_argument("--verify_decode", action="store_true")
    parser.add_argument("--decode_workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source_csv = Path(args.source_csv or root / "indexes" / "domain_csvs" / "sdv5.csv").resolve()
    data_root = Path(args.data_root or root / "data").resolve()
    out_dir = Path(args.out_dir or root / "indexes" / "paper_sdv5_single_source").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "trainval_sdv5_100k_10k.csv"
    out_summary = out_dir / "summary.json"
    error_path = out_dir / "decode_errors.txt"

    known_decode_bad: set[str] = set()
    if error_path.is_file():
        for line in error_path.read_text(encoding="utf-8", errors="replace").splitlines():
            raw_path = line.split("\t", 1)[0].strip()
            if not raw_path:
                continue
            try:
                known_decode_bad.add(Path(raw_path).resolve().relative_to(data_root).as_posix())
            except ValueError:
                continue

    source = read_rows(source_csv)
    usable_paths, zero_byte_paths = scan_nonempty_files(data_root, top_level="sdv5")
    source_paths = {str(row["path"]).replace("\\", "/") for row in source}
    unavailable_paths = source_paths - usable_paths
    source = [
        row for row in source
        if str(row["path"]).replace("\\", "/") in usable_paths
        and str(row["path"]).replace("\\", "/") not in known_decode_bad
    ]
    by_split_label: dict[tuple[str, int], list[dict]] = {}
    for row in source:
        split = str(row["split"]).strip().lower()
        label = int(row["label"])
        row["label"] = label
        row["split"] = split
        by_split_label.setdefault((split, label), []).append(row)

    selected = []
    selection_summary = {}
    for label in (0, 1):
        rng = random.Random(args.seed + 1009 * label)
        train_pool = sorted(by_split_label.get(("train", label), []), key=lambda x: x["path"])
        existing_val = sorted(by_split_label.get(("val", label), []), key=lambda x: x["path"])
        if len(existing_val) > args.val_per_class:
            rng.shuffle(existing_val)
            existing_val = existing_val[: args.val_per_class]
        val_needed = args.val_per_class - len(existing_val)
        if val_needed < 0:
            raise ValueError("negative validation supplement")
        rng.shuffle(train_pool)
        if len(train_pool) < val_needed + args.train_per_class:
            raise ValueError(
                f"label={label}: need {val_needed + args.train_per_class} train-pool images, "
                f"found {len(train_pool)}"
            )
        val_extra = train_pool[:val_needed]
        final_train = train_pool[val_needed : val_needed + args.train_per_class]
        final_val = existing_val + val_extra

        for row in final_train:
            selected.append({"path": row["path"], "label": label, "domain": "sdv5", "split": "train"})
        for row in final_val:
            selected.append({"path": row["path"], "label": label, "domain": "sdv5", "split": "val"})
        selection_summary[str(label)] = {
            "source_train_pool": len(train_pool),
            "source_existing_val": len(by_split_label.get(("val", label), [])),
            "validation_supplement_from_train": val_needed,
            "selected_train": len(final_train),
            "selected_val": len(final_val),
        }

    random.Random(args.seed).shuffle(selected)
    paths = [row["path"] for row in selected]
    if len(paths) != len(set(paths)):
        duplicates = len(paths) - len(set(paths))
        raise ValueError(f"selected index contains {duplicates} duplicate paths")

    check_rows = selected if args.verify_all else selected[:1000]
    missing = [str(data_root / row["path"]) for row in check_rows if str(row["path"]).replace("\\", "/") not in usable_paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} sampled files are missing; first entries: {missing[:5]}")

    decode_errors = verify_decodable(data_root, selected, args.decode_workers) if args.verify_decode else []
    if decode_errors:
        previous = error_path.read_text(encoding="utf-8", errors="replace").splitlines() if error_path.is_file() else []
        combined = list(dict.fromkeys(previous + decode_errors))
        error_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
        raise ValueError(
            f"{len(decode_errors)} selected images failed full decoding; see {error_path}. "
            "Remove or re-extract those source files, then rebuild the fixed index."
        )

    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "domain", "split"])
        writer.writeheader()
        writer.writerows(selected)

    counts = Counter((row["split"], int(row["label"])) for row in selected)
    expected = {
        ("train", 0): args.train_per_class,
        ("train", 1): args.train_per_class,
        ("val", 0): args.val_per_class,
        ("val", 1): args.val_per_class,
    }
    if dict(counts) != expected:
        raise AssertionError(f"unexpected split counts: {dict(counts)} != {expected}")

    summary = {
        "protocol": "ImageNet real + SDV5 fake single-source training",
        "seed": args.seed,
        "data_root": str(data_root),
        "source_csv": str(source_csv),
        "output_csv": str(out_csv),
        "output_sha256": file_digest(out_csv),
        "counts": {f"{split}_label_{label}": count for (split, label), count in sorted(counts.items())},
        "selection": selection_summary,
        "source_rows_after_nonempty_filter": len(source),
        "source_unavailable_or_zero_byte": len(unavailable_paths),
        "zero_byte_files_under_sdv5": len(zero_byte_paths),
        "known_decode_bad_excluded": len(known_decode_bad),
        "verified_files": len(check_rows),
        "verify_all": bool(args.verify_all),
        "decoded_files": len(selected) if args.verify_decode else 0,
        "verify_decode": bool(args.verify_decode),
        "decode_errors": len(decode_errors),
        "duplicates": 0,
        "files_moved_or_modified": False,
    }
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
