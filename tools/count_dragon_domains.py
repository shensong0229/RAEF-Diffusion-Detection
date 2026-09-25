import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from tqdm import tqdm


# DRAGON 里可能出现的模型字段名候选
CANDIDATE_KEYS = [
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


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def normalize_value(v: Any) -> str:
    if v is None:
        return "UNKNOWN"
    if isinstance(v, (list, tuple)):
        return "|".join(str(x) for x in v)
    return str(v).strip()


def find_domain_from_json(data: Dict[str, Any]) -> str:
    flat = flatten_dict(data)
    lowered = {k.lower(): v for k, v in flat.items()}

    # 先按常见字段精确找
    for key in CANDIDATE_KEYS:
        for fk, fv in lowered.items():
            if fk == key or fk.endswith("." + key) or fk.endswith("_" + key):
                val = normalize_value(fv)
                if val:
                    return val

    # 再按模糊关键词找
    fuzzy_words = ["model", "generator", "engine", "backend"]
    for fk, fv in lowered.items():
        if any(w in fk for w in fuzzy_words):
            val = normalize_value(fv)
            if val:
                return val

    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="DRAGON 解压目录，例如 F:\\FakeImageDetect\\extracted_test",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="输出统计 CSV 路径",
    )
    parser.add_argument(
        "--show_unknown_examples",
        type=int,
        default=10,
        help="最多展示多少个 UNKNOWN 样例文件",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"root not found: {root}")

    json_files = sorted(root.rglob("*.json"))
    png_files = sorted(root.rglob("*.png"))

    print(f"[INFO] root = {root}")
    print(f"[INFO] json files found = {len(json_files)}")
    print(f"[INFO] png files found  = {len(png_files)}")

    domain_counter = Counter()
    unknown_files = []

    for jf in tqdm(json_files, desc="Scanning DRAGON json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(jf.read_text(encoding="utf-8-sig"))
        except Exception:
            try:
                data = json.loads(jf.read_text(encoding="latin-1"))
            except Exception:
                domain_counter["BROKEN_JSON"] += 1
                continue

        if not isinstance(data, dict):
            domain = "NON_DICT_JSON"
        else:
            domain = find_domain_from_json(data)

        domain_counter[domain] += 1

        if domain == "UNKNOWN" and len(unknown_files) < args.show_unknown_examples:
            unknown_files.append(str(jf))

    print("\n==================== DOMAIN COUNTS ====================")
    for domain, cnt in domain_counter.most_common():
        print(f"{domain}\t{cnt}")

    if unknown_files:
        print("\n==================== UNKNOWN EXAMPLES ====================")
        for p in unknown_files:
            print(p)

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        lines = ["domain,count"]
        for domain, cnt in domain_counter.most_common():
            lines.append(f"{domain},{cnt}")
        out_csv.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n[INFO] wrote csv to: {out_csv}")


if __name__ == "__main__":
    main()