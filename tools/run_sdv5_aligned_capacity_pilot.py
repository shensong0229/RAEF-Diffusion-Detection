"""Original-scale, single-seed capacity-controlled fusion pilot."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
INDEX = ROOT / "indexes" / "paper_sdv5_capacity_original_scale" / "trainval_sdv5_36k_4k.csv"
DATA_ROOT = ROOT / "data"
OUT = ROOT / "paper_assets" / "reviewer1_comment5_aligned_pilot"
CACHE = OUT / "feature_cache" / "source"
FUSION_OUT = OUT / "fusion_runs"
TEST_CACHE = ROOT / "paper_assets" / "reviewer1_comment5_fast" / "feature_cache" / "test_domains"
SPATIAL_CKPT = ROOT / "checkpoints" / "run_spatial_imagenet_adm_256" / "best_by_auc.pth"
FIRE_CKPT = ROOT / "checkpoints" / "run_fire_imagenet_adm_a30_std_fg" / "best_by_auc.pth"
VARIANTS = "concat_mlp_matched,standard_crossattn,full"
SEEDS = "42,123,2026"


def run(command: list[str], attempts: int = 1) -> None:
    for attempt in range(1, attempts + 1):
        print(f"[RUN attempt {attempt}/{attempts}] " + subprocess.list2cmdline(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= attempts:
                raise
            print("[WARN] interrupted child process; resumable retry in 15 seconds", flush=True)
            time.sleep(15)


def complete_cache(directory: Path) -> bool:
    try:
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8")).get("status") == "complete"
    except Exception:
        return False


def cache_command(split: str) -> list[str]:
    command = [
        str(PYTHON), str(ROOT / "tools" / "cache_sfire_branch_features.py"),
        "--index_csv", str(INDEX), "--data_root", str(DATA_ROOT), "--split", split,
        "--out_dir", str(CACHE / split), "--spatial_ckpt", str(SPATIAL_CKPT),
        "--fire_ckpt", str(FIRE_CKPT), "--batch_size", "1", "--num_workers", "2",
        "--checkpoint_every_batches", "100", "--cooldown_ms", "75",
    ]
    if split == "train":
        command.extend(["--transform_mode", "train_fixed", "--transform_seed", "42"])
    return command


def train_command(epochs: int) -> list[str]:
    return [
        str(PYTHON), str(ROOT / "tools" / "train_cached_sfire_fusion_ablation.py"), "train",
        "--train_cache", str(CACHE / "train"), "--val_cache", str(CACHE / "val"),
        "--out_dir", str(FUSION_OUT), "--variants", VARIANTS, "--seeds", SEEDS,
        "--epochs", str(epochs), "--patience", "2", "--batch_size", "32",
        "--eval_batch_size", "64",
    ]


def eval_commands() -> list[list[str]]:
    base = [
        str(PYTHON), str(ROOT / "tools" / "train_cached_sfire_fusion_ablation.py"),
        "--out_dir", str(FUSION_OUT), "--variants", VARIANTS, "--seeds", SEEDS,
    ]
    return [
        [*base, "eval", "--test_cache_root", str(TEST_CACHE), "--eval_batch_size", "64"],
        [*base, "summarize"],
    ]


def write_manifest(epochs: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Reviewer 1 Comment 5 original-scale aligned pilot",
        "source": "SDV5 only",
        "subset": {"train_real": 36000, "train_fake": 36000, "val_real": 4000, "val_fake": 4000},
        "fusion_variants": VARIANTS.split(","),
        "seeds": [42, 123, 2026],
        "epochs": epochs,
        "branch_policy": "reuse and freeze exact manuscript branch checkpoints",
        "train_transform": "one fixed seeded draw of the manuscript training augmentation, shared by all variants",
        "validation_and_test_transform": "deterministic evaluation transform",
        "controls": [
            "same SDV5 index", "same augmented branch evidence", "same initialization seed",
            "same optimizer/loss/schedule", "same validation and ten unseen test domains",
            "capacity-matched MLP differs from Ours by only 507 parameters",
            "standard cross-attention has exactly the same trainable parameter count as Ours",
        ],
    }
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def status() -> None:
    progress = {}
    for split in ("train", "val"):
        path = CACHE / split / "progress.json"
        if path.is_file():
            progress[split] = json.loads(path.read_text(encoding="utf-8"))
    completed = sum(
        int((FUSION_OUT / f"{variant}_seed{seed}" / "training_complete.json").is_file())
        for variant in VARIANTS.split(",") for seed in (42, 123, 2026)
    )
    print(json.dumps({
        "train_cache_complete": complete_cache(CACHE / "train"),
        "val_cache_complete": complete_cache(CACHE / "val"),
        "cache_progress": progress,
        "fusion_runs_complete": completed,
        "fusion_runs_total": 9,
        "paper_table_complete": (FUSION_OUT / "paper_capacity_controlled_table.csv").is_file(),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "status"])
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    write_manifest(args.epochs)
    if args.mode == "status":
        status()
        return
    if not INDEX.is_file():
        run([str(PYTHON), str(ROOT / "tools" / "build_sdv5_original_scale_capacity_subset.py")])
    for split in ("train", "val"):
        if not complete_cache(CACHE / split):
            run(cache_command(split), attempts=30)
    run(train_command(args.epochs), attempts=5)
    for command in eval_commands():
        run(command, attempts=5)


if __name__ == "__main__":
    main()
