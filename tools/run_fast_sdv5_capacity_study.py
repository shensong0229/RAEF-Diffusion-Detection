"""One-day SDV5 capacity-controlled study for Reviewer 1 Comment 5."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
INDEX = ROOT / "indexes" / "paper_sdv5_capacity_subset" / "trainval_sdv5_10k_1k.csv"
DATA_ROOT = ROOT / "data"
OUT = ROOT / "paper_assets" / "reviewer1_comment5_fast"
# Reuse the exact branch checkpoints used by the manuscript model.  Their
# directory names retain an older "adm" label, but the author confirmed that
# these are the SDV5 branch weights used for the reported method.
SPATIAL_RUN = "run_spatial_imagenet_adm_256"
FIRE_RUN = "run_fire_imagenet_adm_a30_std_fg"
SPATIAL_DIR = ROOT / "checkpoints" / SPATIAL_RUN
FIRE_DIR = ROOT / "checkpoints" / FIRE_RUN
SPATIAL_CKPT = SPATIAL_DIR / "best_by_auc.pth"
FIRE_CKPT = FIRE_DIR / "best_by_auc.pth"
SOURCE_CACHE = OUT / "feature_cache" / "source"
TEST_CACHE = OUT / "feature_cache" / "test_domains"
FUSION_OUT = OUT / "fusion_runs"
VARIANTS = "concat_linear,gated,concat_mlp_matched,standard_crossattn,full"
SEEDS = "42,123,2026"
TEST_DOMAINS = [
    "Flash_PixArt", "Flash_SD3", "JuggernautXL", "Lumina", "Flux_1",
    "PixArt_Alpha", "SDXL", "SDXL_Lightning", "Kolors", "SSD_1B",
]


def run(command: list[str], max_attempts: int = 1):
    for attempt in range(1, max_attempts + 1):
        print(f"[RUN attempt {attempt}/{max_attempts}] " + subprocess.list2cmdline(command), flush=True)
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt >= max_attempts:
                raise
            print("[WARN] child process stopped; waiting 15 seconds before resumable retry", flush=True)
            time.sleep(15)


def complete_run(directory: Path) -> bool:
    return (directory / "best_by_auc.pth").is_file() and (directory / "resolved.json").is_file()


def complete_cache(directory: Path) -> bool:
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        return manifest.get("status") == "complete"
    except Exception:
        return False


def validate_branch(directory: Path, expected_model: str):
    if not complete_run(directory):
        raise FileNotFoundError(f"Incomplete branch run: {directory}")
    resolved = json.loads((directory / "resolved.json").read_text(encoding="utf-8"))
    if resolved.get("model_name") != expected_model:
        raise ValueError(f"Unexpected branch model: {resolved.get('model_name')} != {expected_model}")


def branch_commands(args):
    common = ["--data_root", str(DATA_ROOT), "--index_csv", str(INDEX), "--seed", "42"]
    spatial = [
        str(PYTHON), str(ROOT / "train_supervised.py"),
        "--config", str(ROOT / "configs" / "paper_sdv5_spatial_resnet50.yaml"),
        "--run_name", SPATIAL_RUN, "--ckpt_dir", str(ROOT / "checkpoints"),
        "--epochs", str(args.branch_epochs), "--batch_size", "32",
        "--grad_accum_steps", "1", "--num_workers", "4", *common,
    ]
    fire = [
        str(PYTHON), str(ROOT / "train_supervised.py"),
        "--config", str(ROOT / "configs" / "paper_sdv5_fire_resnet50.yaml"),
        "--run_name", FIRE_RUN, "--ckpt_dir", str(ROOT / "checkpoints"),
        "--epochs", str(args.branch_epochs), "--batch_size", "2",
        "--grad_accum_steps", "8", "--num_workers", "2", *common,
    ]
    return spatial, fire


def cache_command(index: Path, data_root: Path, split: str, out_dir: Path):
    return [
        str(PYTHON), str(ROOT / "tools" / "cache_sfire_branch_features.py"),
        "--index_csv", str(index), "--data_root", str(data_root), "--split", split,
        "--out_dir", str(out_dir), "--spatial_ckpt", str(SPATIAL_CKPT),
        "--fire_ckpt", str(FIRE_CKPT), "--batch_size", "1", "--num_workers", "2",
        "--checkpoint_every_batches", "100", "--cooldown_ms", "75",
    ]


def fusion_train_command(args):
    return [
        str(PYTHON), str(ROOT / "tools" / "train_cached_sfire_fusion_ablation.py"), "train",
        "--train_cache", str(SOURCE_CACHE / "train"), "--val_cache", str(SOURCE_CACHE / "val"),
        "--out_dir", str(FUSION_OUT), "--variants", VARIANTS, "--seeds", SEEDS,
        "--epochs", str(args.fusion_epochs), "--patience", "2", "--batch_size", "32",
        "--eval_batch_size", "64",
    ]


def fusion_eval_commands():
    base = [
        str(PYTHON), str(ROOT / "tools" / "train_cached_sfire_fusion_ablation.py"),
        "--out_dir", str(FUSION_OUT), "--variants", VARIANTS, "--seeds", SEEDS,
    ]
    return [
        [*base, "eval", "--test_cache_root", str(TEST_CACHE), "--eval_batch_size", "64"],
        [*base, "summarize"],
    ]


def build_manifest(args):
    spatial, fire = branch_commands(args)
    return {
        "purpose": "capacity-controlled response to Reviewer 1 Comment 5",
        "source": "SDV5 only",
        "subset": {"train_real": 10000, "train_fake": 10000, "val_real": 1000, "val_fake": 1000},
        "branch_epochs": "reuse original manuscript checkpoints; no branch retraining",
        "branch_checkpoint_provenance": {
            "note": "legacy directory/index names contain 'adm'; author confirmed these are the manuscript SDV5 weights",
            "spatial": str(SPATIAL_CKPT),
            "fire": str(FIRE_CKPT),
        },
        "fusion_epochs": args.fusion_epochs,
        "fusion_seeds": [42, 123, 2026],
        "fusion_variants": VARIANTS.split(","),
        "test_domains": TEST_DOMAINS,
        "controls": [
            "same fixed subset", "same frozen branch checkpoints", "same cached features",
            "same optimizer and seeds", "capacity-matched concat MLP",
            "standard cross-attention with exactly the same parameter count as Ours",
        ],
        "commands": {"spatial": spatial, "fire": fire, "fusion": fusion_train_command(args)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "status", "run"])
    parser.add_argument("--stage", choices=["branches", "cache", "fusion", "eval", "all"], default="all")
    parser.add_argument("--branch_epochs", type=int, default=5)
    parser.add_argument("--fusion_epochs", type=int, default=5)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    (OUT / "protocol_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.mode == "plan":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.mode == "status":
        fusion_complete = sum(
            int((FUSION_OUT / f"{variant}_seed{seed}" / "training_complete.json").is_file())
            for variant in VARIANTS.split(",") for seed in (42, 123, 2026)
        )
        print(json.dumps({
            "spatial_complete": complete_run(SPATIAL_DIR),
            "fire_complete": complete_run(FIRE_DIR),
            "source_train_cache": complete_cache(SOURCE_CACHE / "train"),
            "source_val_cache": complete_cache(SOURCE_CACHE / "val"),
            "test_caches_complete": sum(int(complete_cache(TEST_CACHE / domain)) for domain in TEST_DOMAINS),
            "fusion_runs_complete": fusion_complete,
            "fusion_runs_total": 15,
            "paper_table_complete": (FUSION_OUT / "paper_capacity_controlled_table.csv").is_file(),
        }, ensure_ascii=False, indent=2))
        return

    spatial, fire = branch_commands(args)
    if args.stage in {"branches", "all"}:
        validate_branch(SPATIAL_DIR, "spatial_resnet50")
        validate_branch(FIRE_DIR, "fire_resnet50")
    if args.stage in {"cache", "all"}:
        validate_branch(SPATIAL_DIR, "spatial_resnet50")
        validate_branch(FIRE_DIR, "fire_resnet50")
        for split in ("train", "val"):
            destination = SOURCE_CACHE / split
            if not complete_cache(destination):
                run(cache_command(INDEX, DATA_ROOT, split, destination), max_attempts=20)
        bundle_root = ROOT / "exports" / "dragon_eval_25domains_unique_real_1k_bundle"
        csv_root = bundle_root / "csvs"
        test_data_root = bundle_root / "images"
        for domain in TEST_DOMAINS:
            destination = TEST_CACHE / domain
            if not complete_cache(destination):
                run(cache_command(csv_root / f"{domain}.csv", test_data_root, "test", destination), max_attempts=20)
    if args.stage in {"fusion", "all"}:
        if not complete_cache(SOURCE_CACHE / "train") or not complete_cache(SOURCE_CACHE / "val"):
            raise FileNotFoundError("Source feature cache is incomplete.")
        run(fusion_train_command(args), max_attempts=5)
    if args.stage in {"eval", "all"}:
        if any(not complete_cache(TEST_CACHE / domain) for domain in TEST_DOMAINS):
            raise FileNotFoundError("One or more test-domain feature caches are incomplete.")
        for command in fusion_eval_commands():
            run(command, max_attempts=5)


if __name__ == "__main__":
    main()
