"""Build a fixed SDV5 subset matching the original 72k/8k train/val scale."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "indexes" / "paper_sdv5_single_source" / "trainval_sdv5_100k_10k.csv"
SOURCE_SUMMARY = SOURCE.parent / "summary.json"
OUT_DIR = ROOT / "indexes" / "paper_sdv5_capacity_original_scale"
OUT_CSV = OUT_DIR / "trainval_sdv5_36k_4k.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"
SEED = 42
TARGETS = {("train", 0): 36_000, ("train", 1): 36_000, ("val", 0): 4_000, ("val", 1): 4_000}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    source_summary = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    if not source_summary.get("verify_decode") or source_summary.get("decode_errors") != 0:
        raise ValueError("The parent SDV5 index has not passed full decode verification.")
    if digest(SOURCE) != source_summary.get("output_sha256"):
        raise ValueError("The parent SDV5 index hash does not match its summary.")

    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((str(row["split"]), int(row["label"])), []).append(row)

    selected = []
    for offset, (key, count) in enumerate(sorted(TARGETS.items())):
        pool = sorted(groups[key], key=lambda row: row["path"])
        selected.extend(random.Random(SEED + 1009 * offset).sample(pool, count))
    random.Random(SEED).shuffle(selected)

    paths = [row["path"] for row in selected]
    counts = Counter((row["split"], int(row["label"])) for row in selected)
    if len(paths) != len(set(paths)):
        raise ValueError("Subset contains duplicate paths.")
    if dict(counts) != TARGETS:
        raise ValueError(f"Unexpected subset counts: {dict(counts)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "domain", "split"])
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "purpose": "Reviewer 1 Comment 5 original-scale capacity-controlled pilot",
        "protocol": "fixed stratified SDV5 subset matching the manuscript's 72k train / 8k val scale",
        "seed": SEED,
        "parent_index": str(SOURCE),
        "parent_sha256": digest(SOURCE),
        "output_csv": str(OUT_CSV),
        "output_sha256": digest(OUT_CSV),
        "counts": {f"{split}_label_{label}": count for (split, label), count in sorted(counts.items())},
        "duplicates": 0,
        "parent_fully_decode_verified": True,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
