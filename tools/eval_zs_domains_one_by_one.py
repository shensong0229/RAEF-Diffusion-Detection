# -*- coding: utf-8 -*-
"""
Sequential per-domain evaluator that does NOT require eval_zs_suite.py to support --append.

How it works:
- Discover per-domain CSVs under --csv_dir
- For each domain, spawn a fresh subprocess to run eval_zs_suite.py on exactly ONE CSV
- Write that domain's result to temporary json/csv files
- Merge the temp result into the final out_json/out_csv
- Skip domains already present in the final outputs
- If a domain crashes, previous completed domains remain saved

Recommended usage (PowerShell):
python .\tools\eval_zs_domains_one_by_one.py `
  --ckpt .\checkpoints\run_sfire_best_test\best_by_auc.pth `
  --data_root F:\FakeImageDetect `
  --csv_dir .\indexes\dragon_eval_25domains_unique_real_1k `
  --image_size 256 `
  --model_name sfire_crossattn_resnet50 `
  --batch_size 1 `
  --num_workers 0 `
  --out_json .\results\sfire_dragon25_best.json `
  --out_csv .\results\sfire_dragon25_best.csv
"""

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Set, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _helper_csv(p: Path) -> bool:
    helper_prefixes = ("summary", "results", "proof")
    stem = p.stem.lower()
    return any(stem.startswith(x) for x in helper_prefixes)


def _discover_domains(csv_dir: Path, pattern: str) -> List[Path]:
    return sorted([
        p for p in csv_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() == ".csv" and not _helper_csv(p)
    ])


def _row_key(r: Dict) -> Tuple[str, str, str]:
    return (
        str(r.get("domain", "")).strip(),
        str(r.get("perturb_type", "none")).strip().lower() or "none",
        str(r.get("perturb_level", "")).strip(),
    )


def _run_key(domain: str, perturb_type: str, perturb_level: str) -> Tuple[str, str, str]:
    return (str(domain).strip(), str(perturb_type).strip().lower() or "none", str(perturb_level).strip())


def _load_done_keys_from_csv(path: Path) -> Set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return {_row_key(dict(r)) for r in reader if str(r.get("domain", "")).strip()}
    except Exception:
        return set()


def _load_done_keys_from_json(path: Path) -> Set[Tuple[str, str, str]]:
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {_row_key(dict(x)) for x in data if isinstance(x, dict) and str(x.get("domain", "")).strip()}
    except Exception:
        return set()
    return set()


def _load_rows_csv(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _load_rows_json(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [dict(x) for x in data]
    return []


def _merge_by_key(existing: List[Dict], new_rows: List[Dict]) -> List[Dict]:
    merged: Dict[Tuple[str, str, str], Dict] = {}
    for r in existing:
        key = _row_key(r)
        if key[0]:
            merged[key] = dict(r)
    for r in new_rows:
        key = _row_key(r)
        if key[0]:
            merged[key] = dict(r)
    return sorted(merged.values(), key=lambda x: _row_key(x))


def _save_json(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _save_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["domain", "perturb_type", "perturb_level", "auc", "ap", "acc", "f1", "n", "split", "csv"]
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _merge_temp_into_final(temp_json: Path, temp_csv: Path, final_json: Path, final_csv: Path):
    # Prefer temp json; fall back to temp csv if needed.
    new_rows = _load_rows_json(temp_json)
    if not new_rows:
        new_rows = _load_rows_csv(temp_csv)

    existing_json = _load_rows_json(final_json)
    existing_csv = _load_rows_csv(final_csv)
    existing = existing_json if existing_json else existing_csv

    merged = _merge_by_key(existing, new_rows)

    _save_json(final_json, merged)
    _save_csv(final_csv, merged)


def _build_eval_command(args, csv_name: str, temp_json: str, temp_csv: str) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "tools" / "eval_zs_suite.py"),
        "--ckpt", args.ckpt,
        "--data_root", args.data_root,
        "--csv_dir", args.csv_dir,
        "--pattern", csv_name,
        "--split", args.split,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--image_size", str(args.image_size),
        "--model_name", args.model_name,
        "--perturb_type", args.perturb_type,
        "--perturb_level", args.perturb_level,
        "--out_json", temp_json,
        "--out_csv", temp_csv,
    ]
    if args.paired_root:
        cmd += ["--paired_root", args.paired_root]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--model_name", default="sfire_crossattn_resnet50")
    ap.add_argument("--paired_root", default="")
    ap.add_argument("--perturb_type", default="none", choices=["none", "jpeg", "resize", "blur"])
    ap.add_argument("--perturb_level", default="")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--stop_on_error", action="store_true")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"csv_dir not found: {csv_dir}")

    final_json = Path(args.out_json).resolve()
    final_csv = Path(args.out_csv).resolve()
    final_json.parent.mkdir(parents=True, exist_ok=True)
    final_csv.parent.mkdir(parents=True, exist_ok=True)

    all_csvs = _discover_domains(csv_dir, args.pattern)
    if not all_csvs:
        raise FileNotFoundError(f"No CSV matched pattern={args.pattern!r} in {csv_dir}")

    done = set()
    done |= _load_done_keys_from_csv(final_csv)
    done |= _load_done_keys_from_json(final_json)

    print(f"[INFO] ckpt         = {args.ckpt}")
    print(f"[INFO] data_root    = {args.data_root}")
    print(f"[INFO] csv_dir      = {csv_dir}")
    print(f"[INFO] model        = {args.model_name}")
    print(f"[INFO] perturb_type = {args.perturb_type}")
    print(f"[INFO] perturb_level= {args.perturb_level}")
    print(f"[INFO] split        = {args.split}")
    print(f"[INFO] bs/nw        = {args.batch_size}/{args.num_workers}")
    print(f"[INFO] found        = {len(all_csvs)} domain CSVs")
    if done:
        print(f"[INFO] already done = {len(done)} rows (will skip by domain+perturb key)")

    todo = [
        p for p in all_csvs
        if _run_key(p.stem, args.perturb_type, args.perturb_level) not in done
    ]
    print(f"[INFO] remaining    = {len(todo)} domains")

    failures = []

    with tempfile.TemporaryDirectory(prefix="eval_domains_") as td:
        td = Path(td)
        for idx, csv_path in enumerate(todo, start=1):
            domain = csv_path.stem
            print("=" * 80)
            print(f"[RUN] ({idx}/{len(todo)}) domain = {domain}")

            temp_json = td / f"{domain}.json"
            temp_csv = td / f"{domain}.csv"

            cmd = _build_eval_command(args, csv_path.name, str(temp_json), str(temp_csv))
            print("[CMD]", " ".join(cmd))

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    check=False,
                    text=True,
                )
            except KeyboardInterrupt:
                print("\n[STOP] interrupted by user")
                break

            if proc.returncode != 0:
                print(f"[FAIL] domain failed: {domain} | returncode={proc.returncode}")
                failures.append({
                    "domain": domain,
                    "perturb_type": args.perturb_type,
                    "perturb_level": args.perturb_level,
                    "returncode": proc.returncode,
                })
                if args.stop_on_error:
                    print("[STOP] stop_on_error is ON, stopping now.")
                    break
                continue

            if not temp_json.exists() and not temp_csv.exists():
                print(f"[FAIL] domain produced no output files: {domain}")
                failures.append({
                    "domain": domain,
                    "perturb_type": args.perturb_type,
                    "perturb_level": args.perturb_level,
                    "returncode": -1,
                })
                if args.stop_on_error:
                    print("[STOP] stop_on_error is ON, stopping now.")
                    break
                continue

            _merge_temp_into_final(temp_json, temp_csv, final_json, final_csv)
            print(f"[OK] domain finished and merged: {domain}")

    if failures:
        fail_path = final_csv.with_name(final_csv.stem + "_failures.csv")
        with open(fail_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["domain", "perturb_type", "perturb_level", "returncode"])
            w.writeheader()
            for r in failures:
                w.writerow(r)
        print(f"[WARN] failures written to: {fail_path}")

    print("[DONE] sequential evaluation finished.")


if __name__ == "__main__":
    main()
