#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm

# Make CNNDetection repo importable
CNNDET_REPO = Path("/mnt/proj/FakeImageDetect/baselines/CNNDetection")
if str(CNNDET_REPO) not in sys.path:
    sys.path.insert(0, str(CNNDET_REPO))

from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--bundle_root", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def read_rows(csv_path: Path):
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def locate_image(rel_path: str, data_root: Path, bundle_root: Path) -> Path | None:
    cand1 = data_root / rel_path
    cand2 = bundle_root / rel_path
    if cand1.exists():
        return cand1
    if cand2.exists():
        return cand2
    return None


def prepare_test_dir(rows, data_root: Path, bundle_root: Path, out_dir: Path):
    if out_dir.exists():
        shutil.rmtree(out_dir)

    real_dir = out_dir / "0_real"
    fake_dir = out_dir / "1_fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    linked = 0

    for i, row in enumerate(tqdm(rows, desc=f"build {out_dir.name}", ncols=100)):
        rel = row["path"]
        src = locate_image(rel, data_root, bundle_root)
        if src is None:
            missing.append(rel)
            continue

        label = int(float(row["label"]))
        dst_dir = real_dir if label == 0 else fake_dir
        dst = dst_dir / f"{i:05d}__{src.name}"

        if not dst.exists():
            os.symlink(src, dst)
        linked += 1

    return linked, missing


def make_test_opt(batch_size: int, num_workers: int, dataroot: str, model_path: str):
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]]
    try:
        opt = TestOptions().parse(print_options=False)
    finally:
        sys.argv = saved_argv

    opt.model_path = model_path
    opt.dataroot = dataroot
    opt.classes = [""]
    opt.no_resize = True
    opt.no_crop = True
    opt.eval = True
    opt.isTrain = False
    opt.batch_size = batch_size
    opt.num_threads = num_workers
    opt.mode = "binary"
    return opt


def main():
    args = parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(args.csv_dir.glob("*.csv"))
    if args.domains:
        wanted = set(args.domains)
        csv_paths = [p for p in csv_paths if p.stem in wanted]

    if not csv_paths:
        raise FileNotFoundError(f"No CSVs found under {args.csv_dir} for domains={args.domains}")

    summary = []

    for csv_path in tqdm(csv_paths, desc="domains", ncols=100):
        domain = csv_path.stem
        domain_root = args.out_root / domain
        done_json = domain_root / "_done.json"

        if done_json.exists() and not args.force:
            print(f"[SKIP] {domain} already done")
            with open(done_json, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        if domain_root.exists() and args.force:
            shutil.rmtree(domain_root)
        domain_root.mkdir(parents=True, exist_ok=True)

        rows = read_rows(csv_path)
        test_dir = domain_root / "testset"
        linked, missing = prepare_test_dir(
            rows=rows,
            data_root=args.data_root,
            bundle_root=args.bundle_root,
            out_dir=test_dir,
        )

        with open(domain_root / "missing.txt", "w", encoding="utf-8") as f:
            for item in missing:
                f.write(item + "\n")

        opt = make_test_opt(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            dataroot=str(test_dir),
            model_path=str(args.ckpt),
        )

        model = resnet50(num_classes=1)
        state_dict = torch.load(args.ckpt, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        model.cuda()
        model.eval()

        acc, ap, r_acc, f_acc, y_true, y_pred = validate(model, opt)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        auc = float(roc_auc_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred > 0.5, zero_division=0))

        row = {
            "domain": domain,
            "n_total": int(len(y_true)),
            "n_real": int((y_true == 0).sum()),
            "n_fake": int((y_true == 1).sum()),
            "linked": int(linked),
            "missing": int(len(missing)),
            "acc": float(acc),
            "ap": float(ap),
            "auc": auc,
            "f1": f1,
            "real_acc": float(r_acc),
            "fake_acc": float(f_acc),
            "ckpt": str(args.ckpt),
        }
        summary.append(row)

        with open(done_json, "w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=2)

        pd.DataFrame([row]).to_csv(domain_root / "metrics_summary.csv", index=False, encoding="utf-8")

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_root / "summary_all.csv", index=False, encoding="utf-8")
    summary_df.to_json(args.out_root / "summary_all.json", orient="records", force_ascii=False, indent=2)

    print("\n[DONE] all requested domains finished.")
    print(f"summary csv: {args.out_root / 'summary_all.csv'}")
    print(f"summary json: {args.out_root / 'summary_all.json'}")


if __name__ == "__main__":
    main()
