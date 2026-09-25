#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将多个测试 CSV 对应的图片，按“每个域一个目录 / ai=假图 / nature=真图”的结构重新导出，
并基于新目录重新生成 CSV。

输入：
- 一个 csv_dir，里面放多个域 CSV（如 glide.csv / sd21.csv / ...）
- 一个 data_root，CSV 中的 path 相对于它

输出：
- out_root/
    glide/
      ai/...
      nature/...
    sd21/
      ai/...
      nature/...
    ...
  out_root/csvs/
      glide.csv
      sd21.csv
      ...
  out_root/repackage_summary.json
  out_root/repackage_summary.csv

说明：
- 会按每个域分别复制，因此同一张 real 图若被多个域复用，会在多个域目录下各复制一份。
- 为避免重名冲突，默认会把原相对路径编码进文件名；若仍冲突，会自动追加序号。
- 生成的新 CSV 中：
    label=1 -> ai
    label=0 -> nature
    split 统一写为 test（可改参数）
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv_dir", type=str, required=True, help="包含多个域 CSV 的目录")
    p.add_argument("--data_root", type=str, required=True, help="CSV 里 path 所相对的数据根目录")
    p.add_argument("--out_root", type=str, required=True, help="导出后的新根目录")
    p.add_argument("--split_name", type=str, default="test", help="新 CSV 写入的 split 值，默认 test")
    p.add_argument("--domains", type=str, default="", help="逗号分隔；为空则自动读取 csv_dir 下所有 csv（跳过 summary）")
    p.add_argument("--overwrite", action="store_true", help="若目标文件已存在则覆盖")
    return p.parse_args()


def sanitize_relpath_to_name(rel_path: str) -> str:
    rel = rel_path.replace("\\", "/")
    rel = rel.lstrip("./")
    name = rel.replace("/", "__")
    return name


def ensure_unique_path(dst: Path) -> Path:
    if not dst.exists():
        return dst
    stem = dst.stem
    suffix = dst.suffix
    parent = dst.parent
    idx = 1
    while True:
        cand = parent / f"{stem}__dup{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def collect_csvs(csv_dir: Path, domains_arg: str) -> List[Path]:
    if domains_arg.strip():
        domains = [x.strip() for x in domains_arg.split(",") if x.strip()]
        return [csv_dir / f"{d}.csv" for d in domains]
    csvs = []
    for p in sorted(csv_dir.glob("*.csv")):
        if p.stem.lower().startswith("summary"):
            continue
        csvs.append(p)
    return csvs


def validate_csv(df: pd.DataFrame, csv_path: Path) -> None:
    need = {"path", "label", "domain"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"CSV 缺少必要列 {sorted(miss)}: {csv_path}")


def main() -> None:
    args = parse_args()
    csv_dir = Path(args.csv_dir)
    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_csv_dir = out_root / "csvs"
    out_csv_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = collect_csvs(csv_dir, args.domains)
    if not csv_paths:
        raise FileNotFoundError(f"在 {csv_dir} 下没有找到可用 csv")

    summary_rows: List[Dict[str, object]] = []

    for csv_path in tqdm(csv_paths, desc="domains", unit="domain"):
        if not csv_path.exists():
            raise FileNotFoundError(f"找不到 CSV: {csv_path}")

        df = pd.read_csv(csv_path)
        validate_csv(df, csv_path)

        domain = str(df["domain"].iloc[0]) if len(df) > 0 else csv_path.stem
        domain_dir = out_root / domain
        ai_dir = domain_dir / "ai"
        nature_dir = domain_dir / "nature"
        ai_dir.mkdir(parents=True, exist_ok=True)
        nature_dir.mkdir(parents=True, exist_ok=True)

        records_out: List[Dict[str, object]] = []
        missing = 0
        copied = 0
        real_count = 0
        fake_count = 0

        rows_iter = df[["path", "label"]].to_dict("records")
        bar = tqdm(rows_iter, desc=f"copy {domain}", unit="img", leave=False)
        for row in bar:
            rel_path = str(row["path"]).replace("\\", "/")
            label = int(row["label"])
            src = data_root / rel_path
            if not src.exists():
                missing += 1
                continue
            if src.suffix.lower() not in IMG_EXTS:
                # 非图片，跳过
                missing += 1
                continue

            target_dir = ai_dir if label == 1 else nature_dir
            flat_name = sanitize_relpath_to_name(rel_path)
            dst = target_dir / flat_name
            if dst.exists() and not args.overwrite:
                # 说明此 csv 内路径有重复；沿用已有文件
                pass
            else:
                dst = ensure_unique_path(dst) if (dst.exists() and args.overwrite) else dst
                shutil.copy2(src, dst)
                copied += 1

            new_rel = f"{domain}/{'ai' if label == 1 else 'nature'}/{dst.name}".replace("\\", "/")
            records_out.append(
                {
                    "path": new_rel,
                    "label": label,
                    "domain": domain,
                    "split": args.split_name,
                }
            )
            if label == 1:
                fake_count += 1
            else:
                real_count += 1

            bar.set_postfix(real=real_count, fake=fake_count, missing=missing)

        out_csv = out_csv_dir / f"{domain}.csv"
        out_df = pd.DataFrame(records_out, columns=["path", "label", "domain", "split"])
        out_df.to_csv(out_csv, index=False, encoding="utf-8")

        summary_rows.append(
            {
                "domain": domain,
                "src_csv": str(csv_path),
                "out_csv": str(out_csv),
                "rows_out": int(len(out_df)),
                "real_count": int(real_count),
                "fake_count": int(fake_count),
                "missing": int(missing),
                "domain_dir": str(domain_dir),
            }
        )

    summary_csv = out_root / "repackage_summary.csv"
    summary_json = out_root / "repackage_summary.json"

    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print(f"[OK] wrote summary csv: {summary_csv}")
    print(f"[OK] wrote summary json: {summary_json}")
    print(f"[OK] wrote csv dir: {out_csv_dir}")
    print(f"[OK] Done.")


if __name__ == "__main__":
    main()
