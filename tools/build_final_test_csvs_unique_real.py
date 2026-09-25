#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rebuild final test CSVs directly from existing domain CSVs, WITHOUT rescanning images.

Rules:
- Read existing complete domain CSVs from --domain_csv_dir.
- Build one output CSV per requested domain.
- Each output CSV contains:
    * n_real real images
    * n_fake fake images
- Real images are sampled GLOBALLY WITHOUT REPLACEMENT across all output domains,
  so the same real image path never appears in two different output CSVs.
- Output columns: path,label,domain,split
- split is always "test"
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from tqdm import tqdm

REQUIRED_COLS = ["path", "label", "domain"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--domain_csv_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--domains", type=str, required=True,
                   help="Comma-separated output domains, e.g. glide,sd21,sdturbo,sdv4,sdv5,vqdm")
    p.add_argument("--n_real", type=int, default=10000)
    p.add_argument("--n_fake", type=int, default=10000)
    p.add_argument("--real_pool_domains", type=str, default="sdv4,sdv5",
                   help="Comma-separated domains providing global real pool")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def norm_col(s: str) -> str:
    return s.replace("\ufeff", "").strip().lower()


def load_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"missing csv: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path} has no header")

        raw_cols = list(reader.fieldnames)
        cols = [norm_col(c) for c in raw_cols]
        miss = sorted(set(REQUIRED_COLS) - set(cols))
        if miss:
            raise ValueError(
                f"{csv_path} missing columns: {miss}; raw_cols={raw_cols}; normalized_cols={cols}"
            )

        idx = {norm_col(c): c for c in raw_cols}
        rows: List[Dict[str, str]] = []
        for r in reader:
            path_val = str(r[idx["path"]]).strip().replace("\\", "/")
            if not path_val:
                continue
            rows.append({
                "path": path_val,
                "label": str(r[idx["label"]]).strip(),
                "domain": str(r[idx["domain"]]).strip(),
                "split": str(r[idx["split"]]).strip() if "split" in idx and r[idx["split"]] is not None else "all",
            })
    return rows


def dedupe_by_path(rows: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out = []
    for r in rows:
        p = r["path"]
        if p not in seen:
            seen.add(p)
            out.append(r)
    return out


def split_groups(rows: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    g: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        g[r.get("split", "all") or "all"].append(r)
    return g


def choose_fake_rows(rows: List[Dict[str, str]], n_fake: int, rng: random.Random) -> Tuple[List[Dict[str, str]], Dict[str, int]]:
    """
    Prefer val > test > all > train, but fill from whatever exists.
    """
    rows = dedupe_by_path([r for r in rows if str(r["label"]) == "1"])
    if len(rows) < n_fake:
        raise ValueError(f"not enough fake rows: need {n_fake}, have {len(rows)}")

    groups = split_groups(rows)
    preferred = ["val", "test", "all", "train"]
    chosen: List[Dict[str, str]] = []
    used = set()
    src_counts: Counter = Counter()

    for sp in preferred:
        pool = [r for r in groups.get(sp, []) if r["path"] not in used]
        rng.shuffle(pool)
        take = min(n_fake - len(chosen), len(pool))
        if take > 0:
            pick = pool[:take]
            chosen.extend(pick)
            used.update(r["path"] for r in pick)
            src_counts[sp] += take
        if len(chosen) == n_fake:
            break

    if len(chosen) < n_fake:
        remain = [r for r in rows if r["path"] not in used]
        rng.shuffle(remain)
        take = n_fake - len(chosen)
        pick = remain[:take]
        chosen.extend(pick)
        for r in pick:
            src_counts[r.get("split", "all")] += 1

    if len(chosen) != n_fake:
        raise RuntimeError(f"failed to choose {n_fake} fake rows, got {len(chosen)}")
    return chosen, dict(src_counts)


def build_global_real_pool(
    all_rows_by_domain: Dict[str, List[Dict[str, str]]],
    real_pool_domains: List[str],
) -> List[Dict[str, str]]:
    pool: List[Dict[str, str]] = []
    for d in real_pool_domains:
        if d not in all_rows_by_domain:
            raise FileNotFoundError(f"real_pool_domain missing csv data: {d}")
        pool.extend([r for r in all_rows_by_domain[d] if str(r["label"]) == "0"])
    return dedupe_by_path(pool)


def choose_global_unique_real(
    real_pool: List[Dict[str, str]],
    domains: List[str],
    n_real: int,
    rng: random.Random,
) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, Dict[str, int]]]:
    need = len(domains) * n_real
    if len(real_pool) < need:
        raise ValueError(f"global real pool not enough: need {need}, have {len(real_pool)}")

    shuffled = list(real_pool)
    rng.shuffle(shuffled)
    chosen_all = shuffled[:need]

    assign: Dict[str, List[Dict[str, str]]] = {}
    src_counts_by_domain: Dict[str, Dict[str, int]] = {}

    for i, d in enumerate(domains):
        chunk = chosen_all[i * n_real:(i + 1) * n_real]
        if len(chunk) != n_real:
            raise RuntimeError(f"domain {d}: failed to assign {n_real} real rows")
        assign[d] = chunk
        c = Counter(r.get("split", "all") for r in chunk)
        src_counts_by_domain[d] = dict(c)

    all_real_paths = [r["path"] for rows in assign.values() for r in rows]
    if len(all_real_paths) != len(set(all_real_paths)):
        raise RuntimeError("global real uniqueness violated")
    return assign, src_counts_by_domain


def write_csv(rows: List[Dict[str, str]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "domain", "split"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    domain_csv_dir = Path(args.domain_csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = [x.strip() for x in args.domains.split(",") if x.strip()]
    real_pool_domains = [x.strip() for x in args.real_pool_domains.split(",") if x.strip()]

    print(f"[INFO] domain_csv_dir = {domain_csv_dir}")
    print(f"[INFO] out_dir = {out_dir}")
    print(f"[INFO] domains = {domains}")
    print(f"[INFO] real_pool_domains = {real_pool_domains}")
    print(f"[INFO] n_real = {args.n_real}, n_fake = {args.n_fake}, seed = {args.seed}")

    all_rows_by_domain: Dict[str, List[Dict[str, str]]] = {}
    for d in tqdm(domains, desc="load csv", unit="csv"):
        all_rows_by_domain[d] = load_csv_rows(domain_csv_dir / f"{d}.csv")

    for d in real_pool_domains:
        if d not in all_rows_by_domain:
            all_rows_by_domain[d] = load_csv_rows(domain_csv_dir / f"{d}.csv")

    real_pool = build_global_real_pool(all_rows_by_domain, real_pool_domains)
    print(f"[INFO] global real pool unique size = {len(real_pool)}")

    real_assign, real_src_by_domain = choose_global_unique_real(
        real_pool=real_pool,
        domains=domains,
        n_real=args.n_real,
        rng=rng,
    )

    written_csvs = []
    summary_rows = []

    for d in tqdm(domains, desc="build final csvs", unit="csv"):
        fake_rows, fake_src = choose_fake_rows(all_rows_by_domain[d], args.n_fake, rng)
        real_rows = real_assign[d]

        final_rows: List[Dict[str, str]] = []
        for r in real_rows:
            final_rows.append({"path": r["path"], "label": "0", "domain": d, "split": "test"})
        for r in fake_rows:
            final_rows.append({"path": r["path"], "label": "1", "domain": d, "split": "test"})

        rng.shuffle(final_rows)
        out_csv = out_dir / f"{d}.csv"
        write_csv(final_rows, out_csv)
        written_csvs.append(out_csv)

        real_paths = [r["path"] for r in final_rows if r["label"] == "0"]
        fake_paths = [r["path"] for r in final_rows if r["label"] == "1"]

        summary_rows.append({
            "domain": d,
            "out_csv": str(out_csv),
            "n_rows": len(final_rows),
            "real_count": len(real_paths),
            "fake_count": len(fake_paths),
            "real_unique_within": len(real_paths) == len(set(real_paths)),
            "fake_unique_within": len(fake_paths) == len(set(fake_paths)),
            "real_source_domains": ",".join(real_pool_domains),
            "real_source_splits": json.dumps(real_src_by_domain[d], ensure_ascii=False),
            "fake_source_domain": d,
            "fake_source_splits": json.dumps(fake_src, ensure_ascii=False),
        })

    all_real_out = []
    for p in written_csvs:
        rows = load_csv_rows(p)
        all_real_out.extend([r["path"] for r in rows if str(r["label"]) == "0"])
    global_real_unique = len(all_real_out) == len(set(all_real_out))
    if not global_real_unique:
        raise RuntimeError("global real uniqueness check failed after writing output CSVs")

    summary_csv = out_dir / "summary.csv"
    summary_json = out_dir / "summary.json"

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "domain", "out_csv", "n_rows", "real_count", "fake_count",
            "real_unique_within", "fake_unique_within",
            "real_source_domains", "real_source_splits",
            "fake_source_domain", "fake_source_splits",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    payload = {
        "domains": domains,
        "real_pool_domains": real_pool_domains,
        "n_real": args.n_real,
        "n_fake": args.n_fake,
        "seed": args.seed,
        "global_real_pool_unique_size": len(real_pool),
        "global_real_unique_across_outputs": global_real_unique,
        "rows": summary_rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote summary csv: {summary_csv}")
    print(f"[OK] wrote summary json: {summary_json}")
    print(f"[OK] global_real_unique_across_outputs = {global_real_unique}")
    print("[OK] Done.")


if __name__ == "__main__":
    main()
