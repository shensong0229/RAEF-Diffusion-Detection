# -*- coding: utf-8 -*-
"""
Faster batch evaluate per-domain CSVs in a directory, with per-domain progress bars.

Changes vs old version:
- load model/checkpoint only once for all CSVs
- skip summary/results/proof helper CSVs automatically
- supports --paired_root for re_/sr_ models with precomputed residual maps
- optional cudnn benchmark on CUDA
- use inference_mode for slightly lower overhead
- supports FIRE single-logit evaluation with sigmoid probability
- auto-fixes stale cache/tmp env vars pointing to a missing drive (e.g. E:\\)
- auto-adds project root to sys.path so tools\\eval_zs_suite.py can import core directly
- supports deterministic robustness evaluation with jpeg / resize / blur at test time
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _drive_root(p: str):
    try:
        drive = os.path.splitdrive(p)[0]
        if drive:
            return drive + "\\"
    except Exception:
        pass
    return None


def _redirect_env_path(var_name: str, default_dir: Path, changed: Dict[str, Dict[str, str]]):
    old_v = os.environ.get(var_name, "")
    if not old_v:
        return
    dr = _drive_root(old_v)
    if dr and (not os.path.exists(dr)):
        default_dir.mkdir(parents=True, exist_ok=True)
        new_v = str(default_dir)
        os.environ[var_name] = new_v
        changed[var_name] = {"old": old_v, "new": new_v}


def _fix_runtime_dirs(project_root: Path):
    """Avoid WinError 3 when old cache/tmp env vars still point to a missing drive like E:\\."""
    cache_base = project_root / ".cache"
    tmp_base = project_root / ".tmp"
    cache_base.mkdir(parents=True, exist_ok=True)
    tmp_base.mkdir(parents=True, exist_ok=True)

    changed: Dict[str, Dict[str, str]] = {}

    cache_map = {
        "HF_HOME": cache_base / "hf",
        "HUGGINGFACE_HUB_CACHE": cache_base / "hf" / "hub",
        "HF_HUB_CACHE": cache_base / "hf" / "hub",
        "TRANSFORMERS_CACHE": cache_base / "hf" / "transformers",
        "TORCH_HOME": cache_base / "torch",
        "TIMM_HOME": cache_base / "timm",
        "XDG_CACHE_HOME": cache_base,
    }
    tmp_map = {
        "TEMP": tmp_base,
        "TMP": tmp_base,
        "TMPDIR": tmp_base,
    }

    for var_name, default_dir in cache_map.items():
        _redirect_env_path(var_name, default_dir, changed)
    for var_name, default_dir in tmp_map.items():
        _redirect_env_path(var_name, default_dir, changed)

    # Give FIRE a stable local default VAE path when the env var is not already set.
    local_vae_dir = project_root / "pretrained" / "sd15_vae"
    if "FIRE_VAE_DIR" not in os.environ and local_vae_dir.is_dir():
        os.environ["FIRE_VAE_DIR"] = str(local_vae_dir)

    if changed:
        print("[WARN] detected stale cache/tmp dirs on a missing drive; redirected to project-local paths")
        for k, vv in changed.items():
            print(f"  - {k}: {vv['old']} -> {vv['new']}")


_fix_runtime_dirs(PROJECT_ROOT)

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.models import build_model
from core.dataset.csv_dataset import CSVIndexDataset
from core.dataset.transforms import (
    build_transforms_for_model,
    build_paired_transforms_for_model,
    build_fixed_robustness_perturb,
    wrap_transform_with_pre,
)


def _is_re_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("re_")


def _is_sr_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("sr_")


def _is_residual_model(name: str) -> bool:
    return _is_re_model(name) or _is_sr_model(name)


def _is_fire_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("fire_")


def _move_to_device(x, device: str):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_move_to_device(xx, device) for xx in x)
    raise TypeError(f"Unsupported batch item type: {type(x)}")


def _build_eval_loader(ds, batch_size: int, num_workers: int, pin_memory: bool):
    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def _build_and_load_model(ckpt_path: str, model_name: str, device: str):
    nc = 1 if _is_fire_model(model_name) else 2
    model = build_model(model_name, num_classes=nc, pretrained=False, dropout=0.0).to(device)
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model


def _extract_fire_logits(model_out: torch.Tensor):
    if isinstance(model_out, tuple):
        return model_out[0]
    return model_out


def _compute_prob_from_logits(logits: torch.Tensor, model_name: str) -> np.ndarray:
    if _is_fire_model(model_name):
        logits = _extract_fire_logits(logits)
        if not torch.is_tensor(logits):
            raise TypeError(f"FIRE logits must be a Tensor, got {type(logits)}")

        if logits.ndim == 2:
            if logits.shape[1] != 1:
                raise ValueError(
                    f"FIRE model should output single-logit tensor with shape [B, 1], got {tuple(logits.shape)}"
                )
            logits = logits[:, 0]
        elif logits.ndim != 1:
            raise ValueError(
                f"FIRE model should output [B] or [B, 1] logits, got {tuple(logits.shape)}"
            )

        prob = torch.sigmoid(logits)
        return prob.detach().cpu().numpy()

    if isinstance(logits, tuple):
        logits = logits[0]
    if not torch.is_tensor(logits):
        raise TypeError(f"Model logits must be a Tensor, got {type(logits)}")
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError(
            f"Non-FIRE model should output class logits with shape [B, C>=2], got {tuple(logits.shape)}"
        )

    prob = torch.softmax(logits, dim=1)[:, 1]
    return prob.detach().cpu().numpy()


def eval_one_csv(model, device: str, data_root: str, index_csv: str, split: str,
                 batch_size: int, num_workers: int, image_size: int, model_name: str,
                 paired_root: str = "", perturb_type: str = "none", perturb_level: str = "") -> Dict:
    use_residual_model = _is_residual_model(model_name)
    perturb_type = str(perturb_type or "none").lower().strip()
    perturb_level = str(perturb_level or "").strip()

    if use_residual_model:
        if perturb_type not in {"", "none", "clean"}:
            raise NotImplementedError(
                f"Robustness perturbation is currently only implemented for non-residual evaluation paths; got model={model_name}"
            )
        if not paired_root:
            raise ValueError(f"{model_name} requires --paired_root for evaluation.")
        pair_mode = "aux_only" if _is_re_model(model_name) else "pair"
        paired_tf = build_paired_transforms_for_model(model_name, image_size=image_size, is_train=False)
        ds = CSVIndexDataset(
            index_csv, data_root, split=split, domains=None,
            paired_root=paired_root, paired_transform=paired_tf, pair_mode=pair_mode
        )
    else:
        base_tf = build_transforms_for_model(model_name, image_size=image_size, is_train=False)
        pre_tf = build_fixed_robustness_perturb(perturb_type=perturb_type, perturb_level=perturb_level)
        tf = wrap_transform_with_pre(base_tf, pre_tf)
        ds = CSVIndexDataset(index_csv, data_root, split=split, domains=None, transform=tf)

    dl = _build_eval_loader(
        ds=ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    y_true, y_prob = [], []
    seen = 0

    pbar = tqdm(dl, total=len(dl), desc=f"eval {Path(index_csv).stem}", unit="batch",
                dynamic_ncols=True, leave=False)
    with torch.inference_mode():
        for x, y, _dom in pbar:
            bs = int(y.size(0))
            seen += bs
            x = _move_to_device(x, device)
            if device == "cuda" and torch.is_tensor(x):
                x = x.contiguous(memory_format=torch.channels_last)
            logits = model(x)
            prob = _compute_prob_from_logits(logits, model_name=model_name)
            y_prob.append(prob)
            y_true.append(y.numpy())
            pbar.set_postfix(seen=seen)

    from core.utils.metrics import compute_binary_metrics
    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob).astype(float)

    m = compute_binary_metrics(y_true, y_prob, thr=0.5)
    m.update(
        {
            "n": int(len(y_true)),
            "split": split,
            "perturb_type": "none" if perturb_type in {"", "clean"} else perturb_type,
            "perturb_level": perturb_level,
        }
    )
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--pattern", default="*.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--model_name", default="spatial_resnet50")
    ap.add_argument("--paired_root", default="")
    ap.add_argument("--perturb_type", default="none", choices=["none", "jpeg", "resize", "blur"])
    ap.add_argument("--perturb_level", default="")
    ap.add_argument("--out_json", default="")
    ap.add_argument("--out_csv", default="")
    args = ap.parse_args()

    csv_dir = Path(args.csv_dir).resolve()
    if not csv_dir.is_dir():
        raise FileNotFoundError(f"csv_dir not found: {csv_dir}")

    csv_paths = sorted([p for p in csv_dir.glob(args.pattern) if p.is_file()])
    keep = []
    for p in csv_paths:
        n = p.name.lower()
        if "results" in n or "proof" in n or "swap_proof" in n or "summary" in n:
            continue
        keep.append(p)
    csv_paths = keep

    if not csv_paths:
        raise FileNotFoundError(f"No CSV found in {csv_dir} with pattern={args.pattern}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"[INFO] ckpt         = {args.ckpt}")
    print(f"[INFO] data_root    = {args.data_root}")
    print(f"[INFO] paired_root  = {args.paired_root}")
    print(f"[INFO] csv_dir      = {csv_dir}")
    print(f"[INFO] model        = {args.model_name}")
    print(f"[INFO] perturb_type = {args.perturb_type}")
    print(f"[INFO] perturb_level= {args.perturb_level}")
    print(f"[INFO] device       = {device}")
    print(f"[INFO] split={args.split}, bs={args.batch_size}, nw={args.num_workers}")

    model = _build_and_load_model(args.ckpt, args.model_name, device)

    results: List[Dict] = []
    outer = tqdm(csv_paths, desc="batch eval", unit="csv", dynamic_ncols=True)
    for p in outer:
        outer.set_postfix(domain=p.stem)
        m = eval_one_csv(
            model=model,
            device=device,
            data_root=args.data_root,
            index_csv=str(p),
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            image_size=args.image_size,
            model_name=args.model_name,
            paired_root=args.paired_root,
            perturb_type=args.perturb_type,
            perturb_level=args.perturb_level,
        )
        m["csv"] = str(p)
        m["domain"] = p.stem
        results.append(m)

    results = sorted(results, key=lambda x: x.get("domain", ""))

    if args.out_json:
        outp = Path(args.out_json).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote json: {outp}")

    if args.out_csv:
        outp = Path(args.out_csv).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)
        cols = ["domain", "perturb_type", "perturb_level", "auc", "ap", "acc", "f1", "n", "split", "csv"]
        with open(outp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in results:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"[OK] wrote csv: {outp}")

    print("[OK] Done.")


if __name__ == "__main__":
    main()
