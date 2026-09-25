import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_real_images(real_roots: List[Path], data_root: Path) -> List[str]:
    real_rel_paths = []
    for rr in real_roots:
        if not rr.exists():
            print(f"[WARN] real root not found, skip: {rr}")
            continue
        files = [p for p in rr.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]
        print(f"[INFO] real files from {rr}: {len(files)}")
        for p in tqdm(files, desc=f"Collect real from {rr.name}"):
            real_rel_paths.append(norm_rel(p, data_root))
    # 去重，避免 train/val 或其他目录里万一有重复
    real_rel_paths = sorted(set(real_rel_paths))
    print(f"[INFO] unique real images total = {len(real_rel_paths)}")
    return real_rel_paths


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def find_domain_from_json(data: dict) -> str:
    candidate_keys = [
        "model",
        "generator",
        "gen_model",
        "source_model",
        "model_name",
        "generator_name",
        "diffusion_model",
        "engine",
        "backend",
    ]
    flat = flatten_dict(data)
    lowered = {k.lower(): v for k, v in flat.items()}

    for key in candidate_keys:
        for fk, fv in lowered.items():
            if fk == key or fk.endswith("." + key) or fk.endswith("_" + key):
                val = str(fv).strip()
                if val:
                    return val

    fuzzy_words = ["model", "generator", "engine", "backend"]
    for fk, fv in lowered.items():
        if any(w in fk for w in fuzzy_words):
            val = str(fv).strip()
            if val:
                return val

    return "UNKNOWN"


def collect_dragon_fake_by_domain(dragon_root: Path, data_root: Path) -> Dict[str, List[str]]:
    json_files = sorted(dragon_root.rglob("*.json"))
    print(f"[INFO] dragon json files found = {len(json_files)}")

    domain_to_fake = defaultdict(list)

    for jf in tqdm(json_files, desc="Scan DRAGON json"):
        png_path = jf.with_suffix(".png")
        if not png_path.exists():
            continue

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(jf.read_text(encoding="utf-8-sig"))
        except Exception:
            try:
                data = json.loads(jf.read_text(encoding="latin-1"))
            except Exception:
                continue

        if not isinstance(data, dict):
            continue

        domain = find_domain_from_json(data)
        if domain == "UNKNOWN":
            continue

        rel_path = norm_rel(png_path, data_root)
        domain_to_fake[domain].append(rel_path)

    # 去重并排序
    cleaned = {}
    for domain, rels in domain_to_fake.items():
        cleaned[domain] = sorted(set(rels))

    print(f"[INFO] DRAGON domains found = {len(cleaned)}")
    for domain in sorted(cleaned.keys()):
        print(f"[INFO] {domain}: {len(cleaned[domain])} fake images")

    return cleaned


def write_csv(rows: List[dict], out_csv: Path):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "domain", "split"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="项目总 data_root，例如 F:\\FakeImageDetect")
    parser.add_argument("--dragon_root", type=str, required=True, help="DRAGON 解压目录，例如 F:\\FakeImageDetect\\extracted_test")
    parser.add_argument("--real_root_1", type=str, required=True, help="sdv4 train nature")
    parser.add_argument("--real_root_2", type=str, required=True, help="sdv4 val nature")
    parser.add_argument("--out_dir", type=str, required=True, help="输出 CSV 目录")
    parser.add_argument("--summary_csv", type=str, required=True, help="输出 summary.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_fake_per_domain", type=int, default=1000)
    parser.add_argument("--n_real_per_domain", type=int, default=1000)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    data_root = Path(args.data_root).resolve()
    dragon_root = Path(args.dragon_root).resolve()
    real_root_1 = Path(args.real_root_1).resolve()
    real_root_2 = Path(args.real_root_2).resolve()
    out_dir = Path(args.out_dir).resolve()
    summary_csv = Path(args.summary_csv).resolve()

    print(f"[INFO] data_root   = {data_root}")
    print(f"[INFO] dragon_root = {dragon_root}")
    print(f"[INFO] out_dir     = {out_dir}")

    # 1) 收集 DRAGON fake
    domain_to_fake = collect_dragon_fake_by_domain(dragon_root, data_root)

    # 2) 收集 sdv4 real
    real_pool = collect_real_images([real_root_1, real_root_2], data_root)

    # 3) 检查 fake 是否够
    insufficient_domains = []
    for domain, rels in domain_to_fake.items():
        if len(rels) < args.n_fake_per_domain:
            insufficient_domains.append((domain, len(rels)))
    if insufficient_domains:
        print("[ERROR] some domains do not have enough fake images:")
        for domain, n in insufficient_domains:
            print(f"  {domain}: only {n}, need {args.n_fake_per_domain}")
        raise RuntimeError("Insufficient fake images for some domains.")

    # 4) 检查 real 是否够
    n_domains = len(domain_to_fake)
    total_real_needed = n_domains * args.n_real_per_domain
    if len(real_pool) < total_real_needed:
        raise RuntimeError(
            f"Not enough real images: have {len(real_pool)}, "
            f"need {total_real_needed} ({n_domains} domains x {args.n_real_per_domain})"
        )

    # 5) 全局随机打乱 real pool，并按域无放回分配
    rng.shuffle(real_pool)

    summary_rows = []
    real_cursor = 0

    for domain in tqdm(sorted(domain_to_fake.keys()), desc="Build per-domain CSV"):
        fake_candidates = domain_to_fake[domain][:]
        rng.shuffle(fake_candidates)
        fake_selected = fake_candidates[: args.n_fake_per_domain]

        real_selected = real_pool[real_cursor: real_cursor + args.n_real_per_domain]
        real_cursor += args.n_real_per_domain

        rows = []
        for p in fake_selected:
            rows.append({
                "path": p,
                "label": 1,
                "domain": domain,
                "split": "test",
            })
        for p in real_selected:
            rows.append({
                "path": p,
                "label": 0,
                "domain": domain,
                "split": "test",
            })

        rng.shuffle(rows)

        out_csv = out_dir / f"{domain}.csv"
        write_csv(rows, out_csv)

        summary_rows.append({
            "domain": domain,
            "out_csv": str(out_csv),
            "n_rows": len(rows),
            "real_count": args.n_real_per_domain,
            "fake_count": args.n_fake_per_domain,
            "real_unique_within": len(set(real_selected)) == len(real_selected),
            "fake_unique_within": len(set(fake_selected)) == len(fake_selected),
        })

    # 6) summary.csv
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "domain", "out_csv", "n_rows",
                "real_count", "fake_count",
                "real_unique_within", "fake_unique_within"
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    # 7) 全局 real 唯一性检查
    all_real = []
    for domain in sorted(domain_to_fake.keys()):
        csv_path = out_dir / f"{domain}.csv"
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if int(row["label"]) == 0:
                    all_real.append(row["path"])

    print(f"[INFO] total domains = {n_domains}")
    print(f"[INFO] total real assigned = {len(all_real)}")
    print(f"[INFO] global_real_unique_across_outputs = {len(all_real) == len(set(all_real))}")
    print(f"[INFO] summary written to: {summary_csv}")
    print(f"[INFO] per-domain csv dir: {out_dir}")


if __name__ == "__main__":
    main()