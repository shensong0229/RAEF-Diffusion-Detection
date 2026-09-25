# -*- coding: utf-8 -*-
"""
为 FakeImageDetect 构建 DiffusionForensics / FIRE ImageNet 协议索引。

功能
----
1) 扫描训练集：
   data_root/train/imagenet/real/**
   data_root/train/imagenet/adm/**
   生成一个 combined CSV，字段为：path,label,domain,split
   其中 split 由 train/val 随机划分得到。

2) 扫描测试集（若存在）：
   data_root/test/imagenet/real/**
   data_root/test/imagenet/<fake_domain>/**
   为每个 fake_domain 生成一个独立 CSV：
   out_test_dir/<fake_domain>.csv
   每个 CSV 都包含：
   - 全部（或按需平衡后的）real 样本，label=0, domain=real
   - 对应 fake_domain 样本，label=1, domain=<fake_domain>
   - split=test

设计约定
--------
- CSV 中的 path 一律写成相对于 --data_root 的相对路径，兼容当前 csv_dataset.py
- 所有耗时扫描都带 tqdm
- 训练集 train/val 划分按类别分层随机划分，保证 real / fake 比例稳定
- 默认不平衡 test；如需与旧项目风格一致，可加 --balance_test

推荐 data_root
--------------
例如：F:/DiffusionForensics/images
则 CSV 里的 path 形如：
train/imagenet/real/n01440764/xxx.JPEG
train/imagenet/adm/0/yyy.png
test/imagenet/real/n01440764/zzz.JPEG
test/imagenet/sdv1/12/aaa.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".jfif", ".jpeg"}


@dataclass
class Row:
    path: str
    label: int
    domain: str
    split: str


def norm_relpath(p: Path, root: Path) -> str:
    rel = p.relative_to(root).as_posix()
    return rel


def is_image_file(p: Path, exts: set[str]) -> bool:
    return p.is_file() and p.suffix.lower() in exts


def scan_images_recursive(root: Path, data_root: Path, exts: set[str], desc: str) -> List[str]:
    if not root.is_dir():
        raise FileNotFoundError(f"directory not found: {root}")

    paths: List[str] = []
    pbar = tqdm(desc=desc, unit="file", dynamic_ncols=True)
    for dirpath, _dirnames, filenames in os.walk(root):
        dir_path = Path(dirpath)
        for fn in filenames:
            fp = dir_path / fn
            if is_image_file(fp, exts):
                paths.append(norm_relpath(fp, data_root))
                pbar.update(1)
    pbar.close()
    return sorted(paths)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Sequence[Row], out_csv: Path) -> None:
    ensure_parent(out_csv)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "domain", "split"])
        for r in rows:
            writer.writerow([r.path, int(r.label), r.domain, r.split])


def stratified_split(
    real_paths: Sequence[str],
    fake_paths: Sequence[str],
    val_ratio: float,
    seed: int,
    real_domain: str = "real",
    fake_domain: str = "adm",
) -> List[Row]:
    if not (0.0 < val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0,1), got {val_ratio}")

    rng = random.Random(seed)

    real_list = list(real_paths)
    fake_list = list(fake_paths)
    rng.shuffle(real_list)
    rng.shuffle(fake_list)

    n_real_val = max(1, int(round(len(real_list) * val_ratio))) if len(real_list) > 1 else 0
    n_fake_val = max(1, int(round(len(fake_list) * val_ratio))) if len(fake_list) > 1 else 0

    real_val = set(real_list[:n_real_val])
    fake_val = set(fake_list[:n_fake_val])

    rows: List[Row] = []
    for p in real_list:
        rows.append(Row(path=p, label=0, domain=real_domain, split="val" if p in real_val else "train"))
    for p in fake_list:
        rows.append(Row(path=p, label=1, domain=fake_domain, split="val" if p in fake_val else "train"))

    rows.sort(key=lambda x: (x.split, x.label, x.path))
    return rows


def sample_list(items: Sequence[str], n: int, rng: random.Random) -> List[str]:
    items = list(items)
    if n >= len(items):
        return list(items)
    idx = list(range(len(items)))
    rng.shuffle(idx)
    chosen = sorted(idx[:n])
    return [items[i] for i in chosen]


def build_test_rows(
    real_paths: Sequence[str],
    fake_paths: Sequence[str],
    fake_domain: str,
    balance_test: bool,
    seed: int,
    real_domain: str = "real",
) -> List[Row]:
    rng = random.Random(seed)
    rp = list(real_paths)
    fp = list(fake_paths)

    if balance_test:
        k = min(len(rp), len(fp))
        rp = sample_list(rp, k, rng)
        fp = sample_list(fp, k, rng)

    rows: List[Row] = []
    for p in sorted(rp):
        rows.append(Row(path=p, label=0, domain=real_domain, split="test"))
    for p in sorted(fp):
        rows.append(Row(path=p, label=1, domain=fake_domain, split="test"))
    return rows


def summarize_rows(rows: Sequence[Row]) -> Dict:
    out: Dict[str, Dict[str, int]] = {}
    for r in rows:
        sp = str(r.split)
        if sp not in out:
            out[sp] = {"total": 0, "real": 0, "fake": 0}
        out[sp]["total"] += 1
        if int(r.label) == 0:
            out[sp]["real"] += 1
        else:
            out[sp]["fake"] += 1
    return out


def find_fake_test_domains(test_imagenet_root: Path, real_name: str) -> List[Path]:
    domains: List[Path] = []
    for p in sorted(test_imagenet_root.iterdir()):
        if not p.is_dir():
            continue
        if p.name.lower() == real_name.lower():
            continue
        domains.append(p)
    return domains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="DiffusionForensics images root, e.g. F:\\DiffusionForensics\\images")
    ap.add_argument("--out_train_csv", required=True, help="output combined train/val csv")
    ap.add_argument("--out_test_dir", default="", help="output dir for per-domain test csvs")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance_test", action="store_true", help="balance real/fake counts per test CSV")
    ap.add_argument("--train_real_subdir", default="train/imagenet/real")
    ap.add_argument("--train_fake_subdir", default="train/imagenet/adm")
    ap.add_argument("--test_imagenet_subdir", default="test/imagenet")
    ap.add_argument("--test_real_dirname", default="real")
    ap.add_argument("--real_domain", default="real")
    ap.add_argument("--train_fake_domain", default="adm")
    ap.add_argument("--summary_json", default="")
    args = ap.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    train_real_root = (data_root / args.train_real_subdir).resolve()
    train_fake_root = (data_root / args.train_fake_subdir).resolve()

    print(f"[INFO] data_root       = {data_root}")
    print(f"[INFO] train_real_root = {train_real_root}")
    print(f"[INFO] train_fake_root = {train_fake_root}")
    print(f"[INFO] out_train_csv   = {Path(args.out_train_csv).resolve()}")
    if args.out_test_dir:
        print(f"[INFO] out_test_dir    = {Path(args.out_test_dir).resolve()}")
    print(f"[INFO] val_ratio       = {args.val_ratio}")
    print(f"[INFO] seed            = {args.seed}")
    print(f"[INFO] balance_test    = {bool(args.balance_test)}")

    train_real_paths = scan_images_recursive(train_real_root, data_root, IMG_EXTS, desc="scan train real")
    train_fake_paths = scan_images_recursive(train_fake_root, data_root, IMG_EXTS, desc="scan train fake(adm)")

    if len(train_real_paths) == 0:
        raise RuntimeError(f"No image found under: {train_real_root}")
    if len(train_fake_paths) == 0:
        raise RuntimeError(f"No image found under: {train_fake_root}")

    train_rows = stratified_split(
        real_paths=train_real_paths,
        fake_paths=train_fake_paths,
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        real_domain=str(args.real_domain),
        fake_domain=str(args.train_fake_domain),
    )
    out_train_csv = Path(args.out_train_csv).resolve()
    write_csv(train_rows, out_train_csv)
    train_summary = summarize_rows(train_rows)
    print(f"[OK] wrote train csv: {out_train_csv}")
    print(f"[INFO] train summary: {json.dumps(train_summary, ensure_ascii=False)}")

    test_summary: Dict[str, Dict] = {}
    if args.out_test_dir:
        out_test_dir = Path(args.out_test_dir).resolve()
        out_test_dir.mkdir(parents=True, exist_ok=True)

        test_imagenet_root = (data_root / args.test_imagenet_subdir).resolve()
        if not test_imagenet_root.is_dir():
            print(f"[WARN] test root not found, skip test csv building: {test_imagenet_root}")
        else:
            test_real_root = test_imagenet_root / args.test_real_dirname
            if not test_real_root.is_dir():
                print(f"[WARN] test real root not found, skip test csv building: {test_real_root}")
            else:
                test_real_paths = scan_images_recursive(test_real_root, data_root, IMG_EXTS, desc="scan test real")
                fake_domain_dirs = find_fake_test_domains(test_imagenet_root, real_name=args.test_real_dirname)
                if not fake_domain_dirs:
                    print(f"[WARN] no fake test domains found under: {test_imagenet_root}")
                else:
                    outer = tqdm(fake_domain_dirs, desc="build test csv", unit="domain", dynamic_ncols=True)
                    for dom_dir in outer:
                        domain_name = dom_dir.name
                        outer.set_postfix(domain=domain_name)
                        fake_paths = scan_images_recursive(dom_dir, data_root, IMG_EXTS, desc=f"scan test {domain_name}")
                        if len(fake_paths) == 0:
                            print(f"[WARN] skip empty domain: {dom_dir}")
                            continue
                        rows = build_test_rows(
                            real_paths=test_real_paths,
                            fake_paths=fake_paths,
                            fake_domain=domain_name,
                            balance_test=bool(args.balance_test),
                            seed=int(args.seed),
                            real_domain=str(args.real_domain),
                        )
                        out_csv = out_test_dir / f"{domain_name}.csv"
                        write_csv(rows, out_csv)
                        test_summary[domain_name] = summarize_rows(rows)
                    outer.close()

    if args.summary_json:
        out_summary = Path(args.summary_json).resolve()
        ensure_parent(out_summary)
        payload = {
            "data_root": str(data_root),
            "train_real_root": str(train_real_root),
            "train_fake_root": str(train_fake_root),
            "out_train_csv": str(out_train_csv),
            "out_test_dir": str(Path(args.out_test_dir).resolve()) if args.out_test_dir else "",
            "val_ratio": float(args.val_ratio),
            "seed": int(args.seed),
            "balance_test": bool(args.balance_test),
            "train_summary": train_summary,
            "test_summary": test_summary,
        }
        with open(out_summary, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote summary json: {out_summary}")

    print("[DONE] index building finished.")


if __name__ == "__main__":
    main()
