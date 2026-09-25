#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copy only images referenced by one or more CSV index files into a new directory,
preserving relative paths so existing CSVs can continue to work after a simple
`data_root` switch.

Expected CSV columns: path,label,domain,split (only `path` is required here).

Examples
--------
python copy_images_from_csvs.py \
  --csv_dir F:\\FakeImageDetect\\indexes\\final_test_csvs_eval \
  --data_root F:\\FakeImageDetect\\data \
  --out_root F:\\FakeImageDetect\\export_test_subset

python copy_images_from_csvs.py \
  --csv_paths F:\\...\\glide.csv F:\\...\\sd21.csv \
  --data_root F:\\FakeImageDetect\\data \
  --out_root F:\\FakeImageDetect\\export_test_subset \
  --copy_csvs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable, List, Set

from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Copy images referenced by CSV files into a new root")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_dir", type=str, help="Directory containing CSV files")
    group.add_argument("--csv_paths", nargs="+", help="One or more CSV files")
    p.add_argument("--data_root", type=str, required=True, help="Original data root used by CSV path column")
    p.add_argument("--out_root", type=str, required=True, help="New root directory for copied subset")
    p.add_argument("--copy_csvs", action="store_true", help="Also copy CSV files into out_root/csvs")
    p.add_argument("--strict", action="store_true", help="Abort on first missing file")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing files in out_root")
    p.add_argument("--summary_json", type=str, default=None, help="Optional summary json path")
    return p.parse_args()


def list_csvs(args: argparse.Namespace) -> List[Path]:
    if args.csv_dir:
        csvs = sorted([p for p in Path(args.csv_dir).glob("*.csv") if p.name.lower() != "summary.csv"])
    else:
        csvs = [Path(p) for p in args.csv_paths]
    csvs = [p for p in csvs if p.suffix.lower() == ".csv"]
    if not csvs:
        raise FileNotFoundError("No CSV files found")
    return csvs


def collect_relpaths(csv_paths: Iterable[Path]) -> List[str]:
    ordered: List[str] = []
    seen: Set[str] = set()
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "path" not in reader.fieldnames:
                raise ValueError(f"CSV missing 'path' column: {csv_path}")
            for row in reader:
                rel = (row.get("path") or "").strip().replace("\\", "/")
                if not rel:
                    continue
                if rel not in seen:
                    seen.add(rel)
                    ordered.append(rel)
    return ordered


def sha1_of_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    csvs = list_csvs(args)
    relpaths = collect_relpaths(csvs)

    missing: List[str] = []
    copied = 0
    skipped_existing = 0
    total_bytes = 0

    print(f"[INFO] csv_count = {len(csvs)}")
    for p in csvs:
        print(f"  - {p}")
    print(f"[INFO] unique_paths = {len(relpaths)}")
    print(f"[INFO] data_root = {data_root}")
    print(f"[INFO] out_root = {out_root}")

    for rel in tqdm(relpaths, desc="copy images", unit="img"):
        src = data_root / Path(rel)
        dst = out_root / Path(rel)

        if src.suffix.lower() not in IMAGE_EXTS:
            # Still try to copy; some projects may use unusual suffixes.
            pass

        if not src.exists():
            missing.append(rel)
            if args.strict:
                raise FileNotFoundError(f"Missing source file: {src}")
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        shutil.copy2(src, dst)
        copied += 1
        total_bytes += src.stat().st_size

    if args.copy_csvs:
        csv_out_dir = out_root / "csvs"
        csv_out_dir.mkdir(parents=True, exist_ok=True)
        for p in tqdm(csvs, desc="copy csvs", unit="csv"):
            shutil.copy2(p, csv_out_dir / p.name)

    summary = {
        "csv_count": len(csvs),
        "csvs": [str(p) for p in csvs],
        "unique_paths": len(relpaths),
        "copied_files": copied,
        "skipped_existing": skipped_existing,
        "missing_files": len(missing),
        "total_bytes_copied": total_bytes,
        "data_root": str(data_root),
        "out_root": str(out_root),
        "missing_examples": missing[:20],
    }

    summary_path = Path(args.summary_json) if args.summary_json else out_root / "copy_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] copied_files = {copied}")
    print(f"[OK] skipped_existing = {skipped_existing}")
    print(f"[OK] missing_files = {len(missing)}")
    print(f"[OK] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
