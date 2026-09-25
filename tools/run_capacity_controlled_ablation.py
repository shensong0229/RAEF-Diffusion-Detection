"""Capacity-controlled fusion ablations for Reviewer 1 Comment 5.

This tool keeps the two branches, feature dimensions, losses and training
protocol fixed while swapping only the interaction/fusion strategy.  It can:

1. inspect trainable/frozen parameter counts;
2. smoke-test the real forward path for every variant;
3. generate or execute reproducible training commands for multiple seeds;
4. evaluate completed checkpoints on the same unseen-domain CSVs; and
5. aggregate completed runs into a paper-ready CSV.

No target-domain calibration is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT = PROJECT_ROOT / "paper_assets" / "reviewer1_comment5_ablation"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "sfire_capacity_controlled_ablation.yaml"

VARIANTS = {
    "average": {
        "model_name": "sfire_ablation_average_resnet50",
        "label": "Average fusion",
        "category": "conventional",
    },
    "concat_linear": {
        "model_name": "sfire_ablation_concat_linear_resnet50",
        "label": "Concat + linear projection",
        "category": "reviewer-requested",
    },
    "gated": {
        "model_name": "sfire_ablation_gated_resnet50",
        "label": "Conventional gated fusion",
        "category": "conventional",
    },
    "concat_mlp_matched": {
        "model_name": "sfire_ablation_concat_mlp_matched_resnet50",
        "label": "Capacity-matched concat MLP",
        "category": "capacity-matched",
    },
    "standard_crossattn": {
        "model_name": "sfire_ablation_standard_crossattn_resnet50",
        "label": "Standard cross-attention",
        "category": "capacity-matched",
    },
    "router_only": {
        "model_name": "sfire_ablation_router_only_resnet50",
        "label": "Reliability routing only",
        "category": "proposed-component",
    },
    "no_routing": {
        "model_name": "sfire_ablation_no_routing_resnet50",
        "label": "Anomaly-guided interaction without routing",
        "category": "proposed-component",
    },
    "full": {
        "model_name": "sfire_crossattn_resnet50",
        "label": "Full model",
        "category": "proposed",
    },
}


def parse_csv_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def selected_variants(text: str) -> list[str]:
    names = parse_csv_list(text) if text else list(VARIANTS)
    unknown = [x for x in names if x not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}; choose from {list(VARIANTS)}")
    return names


def ensure_local_vae():
    local_vae = PROJECT_ROOT / "pretrained" / "sd15_vae"
    if local_vae.is_dir():
        os.environ.setdefault("FIRE_VAE_DIR", str(local_vae))


def parameter_row(key: str, model) -> dict:
    trainable = 0
    frozen = 0
    branch_trainable = 0
    fusion_trainable = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        if param.requires_grad:
            trainable += count
            if name.startswith(("spatial_branch.", "fire_branch.")):
                branch_trainable += count
            else:
                fusion_trainable += count
        else:
            frozen += count
    matched_hidden = getattr(getattr(model, "matched_concat_head", None), "hidden_dim", "")
    return {
        "variant": key,
        "label": VARIANTS[key]["label"],
        "category": VARIANTS[key]["category"],
        "model_name": VARIANTS[key]["model_name"],
        "trainable_params": trainable,
        "branch_trainable_params": branch_trainable,
        "fusion_trainable_params": fusion_trainable,
        "frozen_params": frozen,
        "total_params": trainable + frozen,
        "matched_concat_hidden": matched_hidden,
    }


def save_rows(stem: str, rows: list[dict]):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{stem}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with (OUT / f"{stem}.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def inspect_parameters(names: list[str]) -> list[dict]:
    ensure_local_vae()
    import torch
    from core.models import build_model

    rows = []
    for key in names:
        print(f"[PARAM] {key}: building model", flush=True)
        model = build_model(VARIANTS[key]["model_name"], num_classes=2, pretrained=False, dropout=0.1)
        rows.append(parameter_row(key, model))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    full = next((r for r in rows if r["variant"] == "full"), None)
    if full:
        for row in rows:
            row["fusion_param_delta_vs_full"] = row["fusion_trainable_params"] - full["fusion_trainable_params"]
            row["fusion_param_ratio_vs_full"] = row["fusion_trainable_params"] / full["fusion_trainable_params"]
    save_rows("parameter_counts", rows)
    return rows


def smoke_forward(names: list[str], device: str, image_size: int) -> list[dict]:
    ensure_local_vae()
    import torch
    from core.models import build_model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for key in names:
        print(f"[SMOKE] {key}: device={device}", flush=True)
        model = None
        try:
            model = build_model(VARIANTS[key]["model_name"], num_classes=2, pretrained=False, dropout=0.0)
            model.eval().to(device)
            x = torch.rand(1, 3, image_size, image_size, device=device)
            with torch.inference_mode():
                out = model(x, return_aux=True)
            required = {
                "logits", "spatial_logits", "fire_logits_2c", "router_weights",
                "router_balance", "router_entropy", "spatial_proj", "fire_proj",
                "anomaly_prior", "prior_mean", "prior_std",
            }
            missing = sorted(required.difference(out))
            if missing:
                raise KeyError(f"missing auxiliary outputs: {missing}")
            if tuple(out["logits"].shape) != (1, 2):
                raise ValueError(f"unexpected logits shape: {tuple(out['logits'].shape)}")
            rows.append({
                "variant": key,
                "model_name": VARIANTS[key]["model_name"],
                "device": device,
                "status": "pass",
                "logits_shape": str(tuple(out["logits"].shape)),
                "router_weights": json.dumps(out["router_weights"].detach().cpu().tolist()),
            })
        except Exception as exc:
            rows.append({
                "variant": key,
                "model_name": VARIANTS[key]["model_name"],
                "device": device,
                "status": "fail",
                "logits_shape": "",
                "router_weights": "",
                "error": repr(exc),
            })
            print(f"[SMOKE][FAIL] {key}: {exc!r}", flush=True)
        finally:
            if model is not None:
                del model
            if "x" in locals():
                del x
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_rows("smoke_forward", rows)
    return rows


def preflight_index(index_csv: Path, data_root: Path, sample_count: int = 200) -> dict:
    if not index_csv.is_file():
        return {"ok": False, "reason": f"index CSV not found: {index_csv}"}
    if not data_root.is_dir():
        return {"ok": False, "reason": f"data root not found: {data_root}"}
    rows = []
    with index_csv.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") in {"train", "val"}:
                rows.append(row)
                if len(rows) >= sample_count:
                    break
    if not rows:
        return {"ok": False, "reason": "index contains no train/val rows"}
    missing = [str(data_root / row["path"]) for row in rows if not (data_root / row["path"]).is_file()]
    return {
        "ok": not missing,
        "sampled": len(rows),
        "missing": len(missing),
        "first_missing": missing[:5],
        "index_csv": str(index_csv),
        "data_root": str(data_root),
    }


def command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def train_and_evaluate(args, names: list[str]):
    OUT.mkdir(parents=True, exist_ok=True)
    index_csv = Path(args.index_csv).resolve()
    data_root = Path(args.data_root).resolve()
    report = preflight_index(index_csv, data_root)
    (OUT / "training_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    commands = []
    for key in names:
        for seed in args.seeds:
            run_name = f"capacity_ablation_{key}_seed{seed}"
            command = [
                sys.executable,
                str(PROJECT_ROOT / "train_supervised.py"),
                "--config", str(Path(args.config).resolve()),
                "--data_root", str(data_root),
                "--index_csv", str(index_csv),
                "--model_name", VARIANTS[key]["model_name"],
                "--run_name", run_name,
                "--ckpt_dir", str(Path(args.ckpt_dir).resolve()),
                "--seed", str(seed),
                "--batch_size", str(args.batch_size),
                "--grad_accum_steps", str(args.grad_accum_steps),
                "--num_workers", str(args.num_workers),
            ]
            if args.epochs > 0:
                command += ["--epochs", str(args.epochs)]
            commands.append({"variant": key, "seed": seed, "run_name": run_name, "command": command})

    (OUT / "training_commands.ps1").write_text(
        "\n".join(command_text(item["command"]) for item in commands) + "\n", encoding="utf-8"
    )
    print(f"[PLAN] wrote {len(commands)} commands to {OUT / 'training_commands.ps1'}", flush=True)

    if not args.execute:
        return
    if not report.get("ok"):
        raise FileNotFoundError(
            "Training preflight failed. The requested source index does not resolve under the data root. "
            f"See {OUT / 'training_preflight.json'}"
        )

    for item in commands:
        run_dir = Path(args.ckpt_dir).resolve() / item["run_name"]
        best = run_dir / "best_by_auc.pth"
        if best.is_file() and not args.force:
            print(f"[SKIP] completed checkpoint exists: {best}", flush=True)
        else:
            print(f"[TRAIN] {item['variant']} seed={item['seed']}", flush=True)
            subprocess.run(item["command"], cwd=PROJECT_ROOT, check=True)
        if args.eval_csv_dir and best.is_file():
            result_stem = f"{item['variant']}_seed{item['seed']}"
            eval_command = [
                sys.executable,
                str(PROJECT_ROOT / "tools" / "eval_zs_suite.py"),
                "--ckpt", str(best),
                "--data_root", str(Path(args.eval_data_root).resolve()),
                "--csv_dir", str(Path(args.eval_csv_dir).resolve()),
                "--pattern", args.eval_pattern,
                "--image_size", str(args.image_size),
                "--model_name", VARIANTS[item["variant"]]["model_name"],
                "--batch_size", str(args.eval_batch_size),
                "--num_workers", str(args.eval_num_workers),
                "--out_json", str(OUT / f"{result_stem}.json"),
                "--out_csv", str(OUT / f"{result_stem}.csv"),
            ]
            print(f"[EVAL] {item['variant']} seed={item['seed']}", flush=True)
            subprocess.run(eval_command, cwd=PROJECT_ROOT, check=True)


def summarize(names: list[str], seeds: list[int]):
    import statistics

    rows = []
    for key in names:
        seed_means = []
        for seed in seeds:
            path = OUT / f"{key}_seed{seed}.csv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8-sig", newline="") as f:
                values = [float(r["auc"]) for r in csv.DictReader(f) if r.get("auc") not in {None, ""}]
            if values:
                seed_means.append(sum(values) / len(values))
        rows.append({
            "variant": key,
            "label": VARIANTS[key]["label"],
            "completed_seeds": len(seed_means),
            "avg_auc_mean": statistics.mean(seed_means) if seed_means else "",
            "avg_auc_std": statistics.stdev(seed_means) if len(seed_means) > 1 else "",
            "seed_values": json.dumps(seed_means),
        })
    save_rows("auc_summary", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["inspect", "smoke", "plan", "run", "summarize"])
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data_root", default=str(PROJECT_ROOT))
    parser.add_argument("--index_csv", default=str(PROJECT_ROOT / "indexes" / "df_imagenet" / "trainval_imagenet_adm.csv"))
    parser.add_argument("--ckpt_dir", default=str(PROJECT_ROOT / "checkpoints" / "capacity_ablation"))
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_csv_dir", default="")
    parser.add_argument("--eval_data_root", default=str(PROJECT_ROOT))
    parser.add_argument("--eval_pattern", default="*.csv")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.seeds = [int(x) for x in parse_csv_list(args.seeds)]
    names = selected_variants(args.variants)

    if args.mode == "inspect":
        rows = inspect_parameters(names)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    elif args.mode == "smoke":
        rows = smoke_forward(names, args.device, args.image_size)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        if any(r["status"] != "pass" for r in rows):
            raise SystemExit(1)
    elif args.mode in {"plan", "run"}:
        args.execute = args.mode == "run"
        train_and_evaluate(args, names)
    else:
        rows = summarize(names, args.seeds)
        print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
