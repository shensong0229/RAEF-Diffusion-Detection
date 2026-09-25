"""Run the manuscript-faithful SDV5 capacity-controlled ablation pipeline.

Stages are deliberately sequential:
  1) train the spatial branch on the fixed SDV5 single-source index;
  2) train the FIRE branch on exactly the same index;
  3) initialize every fusion control from those same two checkpoints and run
     three seeds while changing only the fusion/interaction implementation.

ADM-trained detector checkpoints are never accepted as formal initialization.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable).resolve()
DATA_ROOT = ROOT / "data"
INDEX_CSV = ROOT / "indexes" / "paper_sdv5_single_source" / "trainval_sdv5_100k_10k.csv"
INDEX_SUMMARY = ROOT / "indexes" / "paper_sdv5_single_source" / "summary.json"
OUT = ROOT / "paper_assets" / "reviewer1_comment5_ablation"

SPATIAL_CONFIG = ROOT / "configs" / "paper_sdv5_spatial_resnet50.yaml"
FIRE_CONFIG = ROOT / "configs" / "paper_sdv5_fire_resnet50.yaml"
FUSION_CONFIG = ROOT / "configs" / "paper_sdv5_fusion_capacity_ablation.yaml"

SPATIAL_RUN = "paper_sdv5_spatial_seed42"
FIRE_RUN = "paper_sdv5_fire_seed42"
SPATIAL_DIR = ROOT / "checkpoints" / SPATIAL_RUN
FIRE_DIR = ROOT / "checkpoints" / FIRE_RUN
SPATIAL_CKPT = SPATIAL_DIR / "best_by_auc.pth"
FIRE_CKPT = FIRE_DIR / "best_by_auc.pth"
FUSION_CKPT_ROOT = ROOT / "checkpoints" / "paper_sdv5_capacity_ablation"

VARIANTS = {
    "average": "sfire_ablation_average_resnet50",
    "concat_linear": "sfire_ablation_concat_linear_resnet50",
    "gated": "sfire_ablation_gated_resnet50",
    "concat_mlp_matched": "sfire_ablation_concat_mlp_matched_resnet50",
    "standard_crossattn": "sfire_ablation_standard_crossattn_resnet50",
    "full": "sfire_crossattn_resnet50",
}
DEFAULT_VARIANTS = ",".join(VARIANTS)
DEFAULT_SEEDS = "42,123,2026"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def split_list(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def assert_protocol_index() -> dict:
    if not INDEX_CSV.is_file() or not INDEX_SUMMARY.is_file():
        raise FileNotFoundError(
            "The fixed SDV5 paper index is missing. Run tools/build_paper_sdv5_source_index.py --verify_all first."
        )
    summary = json.loads(INDEX_SUMMARY.read_text(encoding="utf-8"))
    expected = {
        "train_label_0": 100000,
        "train_label_1": 100000,
        "val_label_0": 10000,
        "val_label_1": 10000,
    }
    if summary.get("counts") != expected:
        raise ValueError(f"Unexpected SDV5 index counts: {summary.get('counts')} != {expected}")
    digest = sha256(INDEX_CSV)
    if digest != summary.get("output_sha256"):
        raise ValueError("The fixed SDV5 index hash differs from summary.json; rebuild it before training.")
    if not summary.get("verify_all") or int(summary.get("verified_files", 0)) != 220000:
        raise ValueError("The SDV5 index has not passed full 220,000-file verification.")
    if not summary.get("verify_decode") or int(summary.get("decoded_files", 0)) != 220000:
        raise ValueError("The SDV5 index has not passed full 220,000-image decode verification.")
    if int(summary.get("decode_errors", -1)) != 0:
        raise ValueError("The SDV5 index contains images that failed decoding.")
    return summary


def command_string(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def branch_command(kind: str, args) -> list[str]:
    if kind == "spatial":
        config, run_name = SPATIAL_CONFIG, SPATIAL_RUN
        batch_size, accum, workers = args.spatial_batch_size, args.spatial_grad_accum, args.num_workers
        epochs = args.spatial_epochs
    elif kind == "fire":
        config, run_name = FIRE_CONFIG, FIRE_RUN
        batch_size, accum, workers = args.fire_batch_size, args.fire_grad_accum, args.num_workers
        epochs = args.fire_epochs
    else:
        raise ValueError(kind)
    command = [
        str(PYTHON), str(ROOT / "train_supervised.py"),
        "--config", str(config),
        "--data_root", str(DATA_ROOT),
        "--index_csv", str(INDEX_CSV),
        "--run_name", run_name,
        "--ckpt_dir", str(ROOT / "checkpoints"),
        "--seed", "42",
        "--batch_size", str(batch_size),
        "--grad_accum_steps", str(accum),
        "--num_workers", str(workers),
    ]
    if epochs > 0:
        command += ["--epochs", str(epochs)]
    return command


def fusion_commands(args) -> list[dict]:
    variants = split_list(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown fusion variants: {unknown}")
    seeds = [int(seed) for seed in split_list(args.seeds)]
    commands = []
    for variant in variants:
        for seed in seeds:
            run_name = f"paper_sdv5_{variant}_seed{seed}"
            command = [
                str(PYTHON), str(ROOT / "train_supervised.py"),
                "--config", str(FUSION_CONFIG),
                "--data_root", str(DATA_ROOT),
                "--index_csv", str(INDEX_CSV),
                "--model_name", VARIANTS[variant],
                "--run_name", run_name,
                "--ckpt_dir", str(FUSION_CKPT_ROOT),
                "--seed", str(seed),
                "--batch_size", str(args.fusion_batch_size),
                "--grad_accum_steps", str(args.fusion_grad_accum),
                "--num_workers", str(args.num_workers),
                "--spatial_ckpt", str(SPATIAL_CKPT),
                "--fire_ckpt", str(FIRE_CKPT),
            ]
            if args.fusion_epochs > 0:
                command += ["--epochs", str(args.fusion_epochs)]
            commands.append({"variant": variant, "seed": seed, "run_name": run_name, "command": command})
    return commands


def validate_branch_checkpoint(kind: str, ckpt: Path, run_dir: Path) -> None:
    resolved_path = run_dir / "resolved.json"
    complete_path = run_dir / "training_complete.json"
    for path in (ckpt, resolved_path, complete_path):
        if not path.is_file():
            raise FileNotFoundError(f"Formal {kind} branch is incomplete; missing: {path}")
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    if Path(resolved.get("index_csv", "")).resolve() != INDEX_CSV.resolve():
        raise ValueError(f"Refusing non-SDV5 {kind} checkpoint: index_csv={resolved.get('index_csv')}")
    if Path(resolved.get("data_root", "")).resolve() != DATA_ROOT.resolve():
        raise ValueError(f"Refusing {kind} checkpoint with a different data root: {resolved.get('data_root')}")
    if str(resolved.get("model_name")) != ("spatial_resnet50" if kind == "spatial" else "fire_resnet50"):
        raise ValueError(f"Unexpected {kind} model name: {resolved.get('model_name')}")


def completed(run_dir: Path) -> bool:
    return (run_dir / "best_by_auc.pth").is_file() and (run_dir / "training_complete.json").is_file()


def run_command(command: list[str], run_dir: Path, force: bool) -> None:
    if completed(run_dir) and not force:
        print(f"[SKIP] complete: {run_dir}", flush=True)
        return
    print(f"[RUN] {command_string(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def write_plan(args) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    spatial = branch_command("spatial", args)
    fire = branch_command("fire", args)
    fusion = fusion_commands(args)
    manifest = {
        "protocol": "single-source SDV5",
        "data_root": str(DATA_ROOT),
        "index_csv": str(INDEX_CSV),
        "index_sha256": sha256(INDEX_CSV),
        "branch_seed": 42,
        "fusion_seeds": [int(seed) for seed in split_list(args.seeds)],
        "spatial_command": spatial,
        "fire_command": fire,
        "fusion_runs": fusion,
        "fairness_constraints": [
            "identical SDV5 train/validation index",
            "identical spatial and FIRE branch checkpoints",
            "identical optimizer, losses, epochs and random seeds",
            "only the fusion/interaction implementation changes",
            "ADM-trained detector checkpoints are excluded",
        ],
    }
    (OUT / "paper_sdv5_training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [command_string(spatial), command_string(fire)] + [command_string(item["command"]) for item in fusion]
    (OUT / "paper_sdv5_training_commands.ps1").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def status_rows(args) -> list[dict]:
    rows = [
        {"stage": "spatial", "run": SPATIAL_RUN, "complete": completed(SPATIAL_DIR)},
        {"stage": "fire", "run": FIRE_RUN, "complete": completed(FIRE_DIR)},
    ]
    for item in fusion_commands(args):
        run_dir = FUSION_CKPT_ROOT / item["run_name"]
        rows.append({"stage": "fusion", "run": item["run_name"], "complete": completed(run_dir)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "status", "run"])
    parser.add_argument("--stage", choices=["spatial", "fire", "fusion", "all"], default="all")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--spatial_epochs", type=int, default=0)
    parser.add_argument("--fire_epochs", type=int, default=0)
    parser.add_argument("--fusion_epochs", type=int, default=0)
    parser.add_argument("--spatial_batch_size", type=int, default=32)
    parser.add_argument("--spatial_grad_accum", type=int, default=1)
    # On the 6-GB RTX 3060, batch 2 is the fastest stable setting. The
    # accumulation values preserve the original effective batches (16/8).
    parser.add_argument("--fire_batch_size", type=int, default=2)
    parser.add_argument("--fire_grad_accum", type=int, default=8)
    parser.add_argument("--fusion_batch_size", type=int, default=2)
    parser.add_argument("--fusion_grad_accum", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("FIRE_VAE_DIR", str(ROOT / "pretrained" / "sd15_vae"))
    summary = assert_protocol_index()
    manifest = write_plan(args)
    if args.mode == "plan":
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.mode == "status":
        print(json.dumps({"index": summary, "runs": status_rows(args)}, ensure_ascii=False, indent=2))
        return

    if args.stage in {"spatial", "all"}:
        run_command(branch_command("spatial", args), SPATIAL_DIR, args.force)
    if args.stage in {"fire", "all"}:
        run_command(branch_command("fire", args), FIRE_DIR, args.force)
    if args.stage in {"fusion", "all"}:
        validate_branch_checkpoint("spatial", SPATIAL_CKPT, SPATIAL_DIR)
        validate_branch_checkpoint("fire", FIRE_CKPT, FIRE_DIR)
        for item in fusion_commands(args):
            run_dir = FUSION_CKPT_ROOT / item["run_name"]
            run_command(item["command"], run_dir, args.force)


if __name__ == "__main__":
    main()
