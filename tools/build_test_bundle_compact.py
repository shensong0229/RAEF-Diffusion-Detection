# -*- coding: utf-8 -*-
"""
build_test_bundle_compact.py

功能：
1. 读取一个目录下的测试 CSV（每个 CSV 对应一个测试域）
2. 只复制 CSV 中 path 列实际引用到的图片
3. 将图片整理为：
       out_dir/
         images/
           test/
             <domain_name>/
               ai/
               nature/
         csvs/
         meta/
4. 自动把输出 bundle 压缩成 zip
5. 生成新的 CSV，其中 path 已改为相对于 out_dir/images 的新路径

适用场景：
- 正式评测前，把 25 个测试域各自 1000 真 + 1000 假导出成精简 bundle
- 避免把整个 sdv4 / 全量 real pool 一起上传云端

CSV 约定（至少包含）：
- path: 相对 data_root 的路径
- label: 0=real, 1=fake
其余列（如 domain, split）会原样保留

输出结构：
out_dir/
  images/
    test/
      <domain_name>/
        ai/
        nature/
  csvs/
    <domain_name>.csv
  meta/
    bundle_summary.json
    per_domain_counts.csv
    copy_manifest.csv
    missing_files.csv (若有缺失)

作者：OpenAI ChatGPT
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact test bundle by copying only images referenced by test CSVs, then zip it."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="原始数据根目录。CSV 中的 path 会相对于这个目录去找图片。",
    )
    parser.add_argument(
        "--csv_dir",
        type=str,
        required=True,
        help="测试 CSV 所在目录。默认读取该目录下全部 .csv 文件。",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="输出 bundle 目录。例如 F:\\FakeImageDetect\\exports\\dragon_eval_25domains_unique_real_1k_bundle",
    )
    parser.add_argument(
        "--zip_path",
        type=str,
        default=None,
        help="输出 zip 路径。若不填，则默认生成 <out_dir>.zip",
    )
    parser.add_argument(
        "--csv_glob",
        type=str,
        default="*.csv",
        help="用于匹配 CSV 的 glob。默认 *.csv",
    )
    parser.add_argument(
        "--domain_from",
        type=str,
        default="filename",
        choices=["filename", "column"],
        help="域名来源：filename=用 CSV 文件名作为域名；column=优先用 CSV 的 domain 列（若该列唯一）。默认 filename。",
    )
    parser.add_argument(
        "--image_topdir",
        type=str,
        default="test",
        help="输出图片目录一级名称，默认 test。最终会变成 images/test/<domain>/ai|nature",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 out_dir 已存在，则先删除再重建。",
    )
    parser.add_argument(
        "--allow_missing",
        action="store_true",
        help="允许缺失文件。若不加该参数，只要发现缺失文件，脚本最后会报错退出。",
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=6,
        choices=list(range(0, 10)),
        help="zip 压缩等级，0-9。默认 6。",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path, overwrite: bool = False) -> None:
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"输出目录已存在：{path}\n"
                f"如果你确认要覆盖，请加 --overwrite"
            )
    path.mkdir(parents=True, exist_ok=True)


def normalize_rel_path(p: str) -> str:
    return str(Path(p.replace("\\", "/")).as_posix())


def parse_label_to_class(label_value) -> str:
    """
    0 -> nature
    1 -> ai
    也兼容 real/fake, nature/ai 文本写法
    """
    s = str(label_value).strip().lower()

    if s in {"0", "real", "nature", "true_real"}:
        return "nature"
    if s in {"1", "fake", "ai", "generated"}:
        return "ai"

    try:
        iv = int(float(s))
        if iv == 0:
            return "nature"
        if iv == 1:
            return "ai"
    except Exception:
        pass

    raise ValueError(f"无法识别的 label 值：{label_value}")


def infer_domain_name(df: pd.DataFrame, csv_path: Path, domain_from: str) -> str:
    if domain_from == "filename":
        return csv_path.stem

    if "domain" in df.columns:
        uniq = [str(x) for x in df["domain"].dropna().astype(str).unique().tolist()]
        if len(uniq) == 1:
            return uniq[0]

    return csv_path.stem


def safe_target_name(src_rel_path: str, original_name: str) -> str:
    """
    为避免不同源目录下同名文件冲突，这里给文件名前加一个短 hash 前缀。
    """
    stem = Path(original_name).stem
    suffix = Path(original_name).suffix
    h = hashlib.md5(src_rel_path.encode("utf-8")).hexdigest()[:10]
    return f"{h}_{stem}{suffix}"


def collect_csv_files(csv_dir: Path, pattern: str) -> List[Path]:
    files = sorted(csv_dir.glob(pattern))
    files = [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]
    if not files:
        raise FileNotFoundError(f"在 {csv_dir} 下没有找到匹配 {pattern} 的 CSV 文件")
    return files


def zip_directory_with_progress(src_dir: Path, zip_path: Path, compression_level: int = 6) -> None:
    all_files = sorted([p for p in src_dir.rglob("*") if p.is_file()])
    if zip_path.exists():
        zip_path.unlink()

    compress_type = zipfile.ZIP_DEFLATED

    with zipfile.ZipFile(
        zip_path,
        mode="w",
        compression=compress_type,
        compresslevel=compression_level,
        allowZip64=True,
    ) as zf:
        for file_path in tqdm(all_files, desc="zipping files", unit="file"):
            arcname = file_path.relative_to(src_dir)
            zf.write(file_path, arcname.as_posix())


def main() -> int:
    args = parse_args()

    data_root = Path(args.data_root).resolve()
    csv_dir = Path(args.csv_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    zip_path = Path(args.zip_path).resolve() if args.zip_path else Path(str(out_dir) + ".zip").resolve()

    if not data_root.exists():
        raise FileNotFoundError(f"--data_root 不存在：{data_root}")
    if not csv_dir.exists():
        raise FileNotFoundError(f"--csv_dir 不存在：{csv_dir}")

    ensure_clean_dir(out_dir, overwrite=args.overwrite)

    images_root = out_dir / "images"
    csvs_root = out_dir / "csvs"
    meta_root = out_dir / "meta"

    images_root.mkdir(parents=True, exist_ok=True)
    csvs_root.mkdir(parents=True, exist_ok=True)
    meta_root.mkdir(parents=True, exist_ok=True)

    csv_files = collect_csv_files(csv_dir, args.csv_glob)

    manifest_rows: List[Dict] = []
    missing_rows: List[Dict] = []
    domain_summary_rows: List[Dict] = []

    # 记录已经复制过的“目标文件”，避免重复 copy
    copied_target_abs_paths = set()

    total_rows = 0
    total_copied = 0
    total_missing = 0
    total_real = 0
    total_fake = 0

    print("=" * 100)
    print("[INFO] build compact test bundle")
    print(f"[INFO] data_root = {data_root}")
    print(f"[INFO] csv_dir   = {csv_dir}")
    print(f"[INFO] out_dir   = {out_dir}")
    print(f"[INFO] zip_path  = {zip_path}")
    print("=" * 100)

    for csv_path in tqdm(csv_files, desc="processing csv files", unit="csv"):
        df = pd.read_csv(csv_path)

        if "path" not in df.columns:
            raise KeyError(f"{csv_path} 缺少必须列：path")
        if "label" not in df.columns:
            raise KeyError(f"{csv_path} 缺少必须列：label")

        domain_name = infer_domain_name(df, csv_path, args.domain_from)

        new_records = []

        domain_total = 0
        domain_real = 0
        domain_fake = 0
        domain_missing = 0

        # 输出图片目录
        domain_ai_dir = images_root / args.image_topdir / domain_name / "ai"
        domain_nature_dir = images_root / args.image_topdir / domain_name / "nature"
        domain_ai_dir.mkdir(parents=True, exist_ok=True)
        domain_nature_dir.mkdir(parents=True, exist_ok=True)

        row_iter = df.to_dict(orient="records")
        for row in tqdm(
            row_iter,
            desc=f"rows::{domain_name}",
            unit="img",
            leave=False,
            total=len(row_iter),
        ):
            domain_total += 1
            total_rows += 1

            src_rel = normalize_rel_path(str(row["path"]))
            src_abs = data_root / src_rel

            cls_name = parse_label_to_class(row["label"])
            if cls_name == "nature":
                domain_real += 1
                total_real += 1
            else:
                domain_fake += 1
                total_fake += 1

            target_dir = domain_nature_dir if cls_name == "nature" else domain_ai_dir
            target_name = safe_target_name(src_rel, Path(src_rel).name)
            target_abs = target_dir / target_name

            # 新 CSV 中相对于 images_root 的 path
            new_rel_path = normalize_rel_path(
                str(Path(args.image_topdir) / domain_name / cls_name / target_name)
            )

            if not src_abs.exists():
                domain_missing += 1
                total_missing += 1
                missing_rows.append(
                    {
                        "csv_file": csv_path.name,
                        "domain_name": domain_name,
                        "original_path": src_rel,
                        "expected_abs_path": str(src_abs),
                        "label": row["label"],
                    }
                )
                # 缺失文件时，该样本不写入新 CSV
                continue

            if str(target_abs) not in copied_target_abs_paths:
                shutil.copy2(src_abs, target_abs)
                copied_target_abs_paths.add(str(target_abs))
                total_copied += 1

            new_row = dict(row)
            new_row["path"] = new_rel_path
            new_records.append(new_row)

            manifest_rows.append(
                {
                    "csv_file": csv_path.name,
                    "domain_name": domain_name,
                    "label": row["label"],
                    "class_name": cls_name,
                    "original_path": src_rel,
                    "original_abs_path": str(src_abs),
                    "bundle_path": new_rel_path,
                    "bundle_abs_path": str(target_abs),
                }
            )

        # 保存新的域 CSV
        new_df = pd.DataFrame(new_records, columns=df.columns.tolist())
        out_csv_path = csvs_root / csv_path.name
        new_df.to_csv(out_csv_path, index=False, encoding="utf-8-sig")

        domain_summary_rows.append(
            {
                "domain_name": domain_name,
                "csv_file": csv_path.name,
                "n_rows_input": int(len(df)),
                "n_rows_output": int(len(new_df)),
                "real_count_output": int((new_df["label"].astype(str).isin(["0", "0.0", "real", "nature"])).sum())
                if len(new_df) > 0
                else 0,
                "fake_count_output": int((new_df["label"].astype(str).isin(["1", "1.0", "fake", "ai"])).sum())
                if len(new_df) > 0
                else 0,
                "domain_total_seen": int(domain_total),
                "domain_real_seen": int(domain_real),
                "domain_fake_seen": int(domain_fake),
                "domain_missing": int(domain_missing),
            }
        )

    # 写 meta 文件
    manifest_csv = meta_root / "copy_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    per_domain_csv = meta_root / "per_domain_counts.csv"
    pd.DataFrame(domain_summary_rows).to_csv(per_domain_csv, index=False, encoding="utf-8-sig")

    if missing_rows:
        missing_csv = meta_root / "missing_files.csv"
        pd.DataFrame(missing_rows).to_csv(missing_csv, index=False, encoding="utf-8-sig")

    # 统计 bundle 中的文件
    bundle_image_files = [p for p in images_root.rglob("*") if p.is_file()]
    bundle_csv_files = [p for p in csvs_root.rglob("*.csv") if p.is_file()]

    bundle_summary = {
        "data_root": str(data_root),
        "csv_dir": str(csv_dir),
        "out_dir": str(out_dir),
        "zip_path": str(zip_path),
        "csv_glob": args.csv_glob,
        "domain_from": args.domain_from,
        "image_topdir": args.image_topdir,
        "n_input_csv_files": len(csv_files),
        "n_output_csv_files": len(bundle_csv_files),
        "n_total_rows_seen": total_rows,
        "n_total_files_copied": total_copied,
        "n_total_missing_files": total_missing,
        "n_total_real_seen": total_real,
        "n_total_fake_seen": total_fake,
        "images_root": str(images_root),
        "csvs_root": str(csvs_root),
        "meta_root": str(meta_root),
    }

    with open(meta_root / "bundle_summary.json", "w", encoding="utf-8") as f:
        json.dump(bundle_summary, f, ensure_ascii=False, indent=2)

    # 若有缺失且未允许 missing，则先报错，不压缩
    if total_missing > 0 and not args.allow_missing:
        print("=" * 100)
        print("[ERROR] 检测到缺失文件，已生成 meta/missing_files.csv")
        print("[ERROR] 由于你没有加 --allow_missing，脚本现在退出，不进行 zip 压缩。")
        print("=" * 100)
        return 2

    # 自动压缩
    print("=" * 100)
    print("[INFO] start zipping bundle ...")
    zip_directory_with_progress(out_dir, zip_path, compression_level=args.compression)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] zip done: {zip_path}")
    print(f"[INFO] zip size: {zip_size_mb:.2f} MB")
    print("=" * 100)

    print("[INFO] finished successfully.")
    print(json.dumps(bundle_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[INFO] interrupted by user")
        raise SystemExit(130)