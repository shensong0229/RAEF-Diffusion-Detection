import argparse
import csv
import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Dict, Set

from tqdm import tqdm


def read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_rel_path(p: str) -> str:
    p = p.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def collect_csv_files(csv_dir: Path, patterns: List[str]) -> List[Path]:
    all_csvs = sorted(csv_dir.glob("*.csv"))
    if not patterns:
        return all_csvs

    selected: List[Path] = []
    pattern_set = set(patterns)
    for csv_path in all_csvs:
        if csv_path.name in pattern_set:
            selected.append(csv_path)
    return selected


def zip_dir(src_dir: Path, zip_path: Path) -> None:
    ensure_parent(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        files = [p for p in src_dir.rglob("*") if p.is_file()]
        for file_path in tqdm(files, desc="zipping", unit="file"):
            arcname = file_path.relative_to(src_dir)
            zf.write(file_path, arcname)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export evaluation images referenced by CSVs and pack them into a zip.")
    parser.add_argument("--data_root", type=str, required=True, help="Project data root. CSV path column is relative to this root.")
    parser.add_argument("--csv_dir", type=str, required=True, help="Directory containing evaluation CSV files.")
    parser.add_argument("--split", type=str, default="test", help="Only export rows with this split value. Use all to disable filtering.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for exported bundle.")
    parser.add_argument("--zip_path", type=str, required=True, help="Final zip output path.")
    parser.add_argument(
        "--patterns",
        type=str,
        nargs="*",
        default=[],
        help='Optional CSV filenames to export, e.g. "SDXL.csv" "SD_2.1.csv". Leave empty to export all CSVs in csv_dir.'
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    csv_dir = Path(args.csv_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    zip_path = Path(args.zip_path).resolve()

    export_images_root = out_dir / "images"
    export_csv_root = out_dir / "csvs"
    export_meta_root = out_dir / "meta"

    export_images_root.mkdir(parents=True, exist_ok=True)
    export_csv_root.mkdir(parents=True, exist_ok=True)
    export_meta_root.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(csv_dir, args.patterns)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {csv_dir}")

    copied_rel_paths: Set[str] = set()
    missing_rows: List[Dict[str, str]] = []
    manifest_rows: List[Dict[str, str]] = []
    per_csv_summary: List[Dict[str, str]] = []

    print(f"[INFO] data_root = {data_root}")
    print(f"[INFO] csv_dir   = {csv_dir}")
    print(f"[INFO] out_dir   = {out_dir}")
    print(f"[INFO] zip_path  = {zip_path}")
    print(f"[INFO] csv count = {len(csv_files)}")

    total_rows = 0
    total_kept_rows = 0

    for csv_path in tqdm(csv_files, desc="processing csvs", unit="csv"):
        rows = read_csv_rows(csv_path)
        total_rows += len(rows)

        kept_rows_this_csv = []
        copied_this_csv = 0
        missing_this_csv = 0

        for row in rows:
            split_value = str(row.get("split", "")).strip()
            if args.split.lower() != "all" and split_value != args.split:
                continue

            rel_path = normalize_rel_path(str(row.get("path", "")))
            if not rel_path:
                missing_this_csv += 1
                missing_rows.append({
                    "csv": csv_path.name,
                    "reason": "empty_path",
                    "path": "",
                    "label": str(row.get("label", "")),
                    "domain": str(row.get("domain", "")),
                    "split": split_value,
                })
                continue

            src_path = data_root / Path(rel_path)
            kept_rows_this_csv.append(row)
            total_kept_rows += 1

            if not src_path.exists():
                missing_this_csv += 1
                missing_rows.append({
                    "csv": csv_path.name,
                    "reason": "missing_file",
                    "path": rel_path,
                    "label": str(row.get("label", "")),
                    "domain": str(row.get("domain", "")),
                    "split": split_value,
                })
                continue

            dst_path = export_images_root / Path(rel_path)
            if rel_path not in copied_rel_paths:
                ensure_parent(dst_path)
                shutil.copy2(src_path, dst_path)
                copied_rel_paths.add(rel_path)
                copied_this_csv += 1

            manifest_rows.append({
                "csv": csv_path.name,
                "path": rel_path,
                "src_path": str(src_path),
                "dst_path": str(dst_path),
                "label": str(row.get("label", "")),
                "domain": str(row.get("domain", "")),
                "split": split_value,
            })

        # copy original filtered csv into bundle
        out_csv_path = export_csv_root / csv_path.name
        if kept_rows_this_csv:
            fieldnames = list(kept_rows_this_csv[0].keys())
            write_csv(out_csv_path, kept_rows_this_csv, fieldnames)
        else:
            # keep an empty CSV with original header if possible
            if rows:
                fieldnames = list(rows[0].keys())
                write_csv(out_csv_path, [], fieldnames)

        per_csv_summary.append({
            "csv": csv_path.name,
            "total_rows_in_csv": str(len(rows)),
            "rows_after_split_filter": str(len(kept_rows_this_csv)),
            "new_unique_images_copied": str(copied_this_csv),
            "missing_rows": str(missing_this_csv),
        })

    # write metadata
    write_csv(
        export_meta_root / "summary.csv",
        per_csv_summary,
        ["csv", "total_rows_in_csv", "rows_after_split_filter", "new_unique_images_copied", "missing_rows"],
    )

    if manifest_rows:
        write_csv(
            export_meta_root / "manifest.csv",
            manifest_rows,
            ["csv", "path", "src_path", "dst_path", "label", "domain", "split"],
        )

    if missing_rows:
        write_csv(
            export_meta_root / "missing_files.csv",
            missing_rows,
            ["csv", "reason", "path", "label", "domain", "split"],
        )

    # also save a simple readme
    readme_path = export_meta_root / "README.txt"
    with readme_path.open("w", encoding="utf-8") as f:
        f.write("Export bundle created from evaluation CSV files.\n")
        f.write(f"data_root = {data_root}\n")
        f.write(f"csv_dir   = {csv_dir}\n")
        f.write(f"split     = {args.split}\n")
        f.write(f"csv_count = {len(csv_files)}\n")
        f.write(f"total_rows_in_all_csvs = {total_rows}\n")
        f.write(f"rows_after_split_filter = {total_kept_rows}\n")
        f.write(f"unique_images_copied = {len(copied_rel_paths)}\n")
        f.write(f"missing_rows = {len(missing_rows)}\n")

    print(f"[DONE] unique_images_copied = {len(copied_rel_paths)}")
    print(f"[DONE] total_rows_in_all_csvs = {total_rows}")
    print(f"[DONE] rows_after_split_filter = {total_kept_rows}")
    print(f"[DONE] missing_rows = {len(missing_rows)}")

    print("[INFO] start zipping...")
    zip_dir(out_dir, zip_path)
    print(f"[DONE] zip saved to: {zip_path}")


if __name__ == "__main__":
    main()