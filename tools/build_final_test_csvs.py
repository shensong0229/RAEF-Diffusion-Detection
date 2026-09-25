
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build final per-domain test CSVs for zero-shot evaluation.

Rules (current project convention):
- Each output CSV contains exactly N real + N fake rows (default 10000 + 10000).
- glide / sd21 / sdturbo:
    fake comes from their own domain CSV (label=1)
    real comes from shared pool: sdv4 label=0, prefer split=val, fallback=train
- sdv4 / sdv5:
    real and fake come from their own domain CSV
    prefer split=val, fallback=train if val is insufficient
- Output columns:
    path,label,domain,split
  where split is fixed to "test" for all output rows.
- All expensive loops show tqdm progress.

Example:
python tools\build_final_test_csvs.py ^
  --domain_csv_dir "F:\FakeImageDetect\indexes\domain_csvs" ^
  --out_dir "F:\FakeImageDetect\indexes\final_test_csvs" ^
  --domains "glide,sd21,sdturbo,sdv4,sdv5" ^
  --n_real 10000 ^
  --n_fake 10000 ^
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final per-domain test CSVs (10000 real + 10000 fake each).")
    p.add_argument("--domain_csv_dir", type=str, required=True, help="Directory containing glide.csv / sd21.csv / ...")
    p.add_argument("--out_dir", type=str, required=True, help="Output directory for final per-domain test CSVs")
    p.add_argument("--domains", type=str, default="glide,sd21,sdturbo,sdv4,sdv5", help="Comma-separated domains to build")
    p.add_argument("--n_real", type=int, default=10000, help="Number of real samples per output CSV")
    p.add_argument("--n_fake", type=int, default=10000, help="Number of fake samples per output CSV")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--shared_real_domain", type=str, default="sdv4", help="Shared real pool source for fake-only domains")
    p.add_argument("--shared_real_prefer_split", type=str, default="val", help="Preferred split for shared real pool")
    p.add_argument("--summary_csv", type=str, default="", help="Optional summary CSV path")
    p.add_argument("--summary_json", type=str, default="", help="Optional summary JSON path")
    return p.parse_args()


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def read_domain_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing domain CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"path", "label", "domain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")
    if "split" not in df.columns:
        df["split"] = "all"
    df["label"] = df["label"].astype(int)
    df["path"] = df["path"].astype(str).str.replace("\\", "/", regex=False)
    df["domain"] = df["domain"].astype(str)
    df["split"] = df["split"].fillna("all").astype(str)
    return df


def sample_with_preference(
    df: pd.DataFrame,
    *,
    label: int,
    n: int,
    prefer_splits: Optional[List[str]],
    rng: random.Random,
    desc: str,
) -> pd.DataFrame:
    """
    Sample exactly n rows with a preference order on splits.
    Will backfill from later splits if the preferred split is insufficient.
    """
    base = df[df["label"] == label].copy()
    if len(base) < n:
        raise ValueError(f"{desc}: available {len(base)} rows for label={label}, need {n}")

    chosen_parts: List[pd.DataFrame] = []
    chosen_indices = set()

    # Preferred splits first
    if prefer_splits:
        for sp in prefer_splits:
            part = base[(base["split"] == sp) & (~base.index.isin(chosen_indices))]
            if len(part) == 0:
                continue
            remaining = n - sum(len(x) for x in chosen_parts)
            if remaining <= 0:
                break
            if len(part) > remaining:
                idx = rng.sample(list(part.index), remaining)
                part = part.loc[idx]
            chosen_parts.append(part)
            chosen_indices.update(part.index)

    # Backfill from whatever remains
    remaining = n - sum(len(x) for x in chosen_parts)
    if remaining > 0:
        rest = base[~base.index.isin(chosen_indices)]
        if len(rest) < remaining:
            raise ValueError(f"{desc}: after preferred split sampling, remaining pool {len(rest)} < need {remaining}")
        idx = rng.sample(list(rest.index), remaining)
        chosen_parts.append(rest.loc[idx])

    out = pd.concat(chosen_parts, axis=0).sample(frac=1.0, random_state=rng.randint(0, 10**9)).reset_index(drop=True)
    return out


def relabel_for_target(df: pd.DataFrame, target_domain: str, split_value: str = "test") -> pd.DataFrame:
    out = df[["path", "label"]].copy()
    out["domain"] = target_domain
    out["split"] = split_value
    return out


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    domain_csv_dir = Path(args.domain_csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = Path(args.summary_csv) if args.summary_csv else out_dir / "summary.csv"
    summary_json = Path(args.summary_json) if args.summary_json else out_dir / "summary.json"

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    print(f"[INFO] domain_csv_dir = {domain_csv_dir}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] domains = {domains}")
    print(f"[INFO] n_real = {args.n_real}, n_fake = {args.n_fake}")
    print(f"[INFO] shared_real_domain = {args.shared_real_domain}, prefer_split = {args.shared_real_prefer_split}")

    # Load all requested domain CSVs first
    loaded = {}
    load_targets = sorted(set(domains + [args.shared_real_domain]))
    for d in tqdm(load_targets, desc="[load_csvs]", unit="domain"):
        loaded[d] = read_domain_csv(domain_csv_dir / f"{d}.csv")

    # Shared real pool for fake-only domains
    shared_real_df = loaded[args.shared_real_domain]
    summary_rows = []

    fake_only_domains = {"glide", "sd21", "sdturbo"}

    for domain in tqdm(domains, desc="[build_final_csvs]", unit="domain"):
        df = loaded[domain]

        if domain in fake_only_domains:
            # Own fake + shared real
            fake_rows = sample_with_preference(
                df,
                label=1,
                n=args.n_fake,
                prefer_splits=["all", "val", "train"],
                rng=rng,
                desc=f"{domain} fake",
            )
            real_rows = sample_with_preference(
                shared_real_df,
                label=0,
                n=args.n_real,
                prefer_splits=[args.shared_real_prefer_split, "all", "train"],
                rng=rng,
                desc=f"{domain} shared real from {args.shared_real_domain}",
            )
        else:
            # Own real + own fake, prefer val then backfill train if insufficient
            fake_rows = sample_with_preference(
                df,
                label=1,
                n=args.n_fake,
                prefer_splits=["val", "all", "train"],
                rng=rng,
                desc=f"{domain} fake",
            )
            real_rows = sample_with_preference(
                df,
                label=0,
                n=args.n_real,
                prefer_splits=["val", "all", "train"],
                rng=rng,
                desc=f"{domain} real",
            )

        out_real = relabel_for_target(real_rows, target_domain=domain, split_value="test")
        out_fake = relabel_for_target(fake_rows, target_domain=domain, split_value="test")
        out_df = pd.concat([out_real, out_fake], axis=0).sample(frac=1.0, random_state=rng.randint(0, 10**9)).reset_index(drop=True)

        out_path = out_dir / f"{domain}.csv"
        out_df.to_csv(out_path, index=False)

        # summary
        used_real_src = {}
        used_fake_src = {}
        if "split" in real_rows.columns:
            for k, v in real_rows["split"].value_counts().to_dict().items():
                used_real_src[str(k)] = int(v)
        if "split" in fake_rows.columns:
            for k, v in fake_rows["split"].value_counts().to_dict().items():
                used_fake_src[str(k)] = int(v)

        summary_rows.append(
            {
                "domain": domain,
                "out_csv": str(out_path),
                "n_rows": int(len(out_df)),
                "real_count": int((out_df["label"] == 0).sum()),
                "fake_count": int((out_df["label"] == 1).sum()),
                "real_source_domain": args.shared_real_domain if domain in fake_only_domains else domain,
                "fake_source_domain": domain,
                "real_source_splits": json.dumps(used_real_src, ensure_ascii=False),
                "fake_source_splits": json.dumps(used_fake_src, ensure_ascii=False),
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, ensure_ascii=False, indent=2)

    print("\n[OK] wrote final per-domain test CSVs:")
    for row in summary_rows:
        print(
            f"  - {row['domain']}: rows={row['n_rows']}, real={row['real_count']}, fake={row['fake_count']}, "
            f"out={row['out_csv']}"
        )
    print(f"[OK] summary_csv = {summary_csv}")
    print(f"[OK] summary_json = {summary_json}")


if __name__ == "__main__":
    main()
