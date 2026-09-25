# -*- coding: utf-8 -*-
# 训练脚本（S-only / F-only / SF / RE / SR / Spatial+FIRE 双流 主入口）
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Any

import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from core.models import build_model
from core.dataset.csv_dataset import CSVIndexDataset, inspect_sample_paths
from core.dataset.transforms import (
    build_transforms_for_model,
    build_paired_transforms_for_model,
    build_consistency_augment,
    wrap_transform_with_pre,
)
from core.utils.seed import set_seed
from core.utils.metrics import compute_binary_metrics
from core.utils.io import ensure_dir, save_json


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cosine_lr(optimizer, base_lr, min_lr, progress):
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = float(lr)
    return float(lr)


def _drive_root(p: str):
    try:
        drive = os.path.splitdrive(p)[0]
        if drive:
            return drive + "\\"
    except Exception:
        pass
    return None


def _fix_cache_dirs_if_bad(project_root: str):
    """Avoid WinError 3 when HF/timm cache points to a non-existent drive (e.g., 'E:\\')."""
    cache_base = os.path.join(project_root, ".cache")
    tmp_base = os.path.join(project_root, ".tmp")
    os.makedirs(cache_base, exist_ok=True)
    os.makedirs(tmp_base, exist_ok=True)

    cache_map = {
        "HF_HOME": os.path.join(cache_base, "hf"),
        "HUGGINGFACE_HUB_CACHE": os.path.join(cache_base, "hf", "hub"),
        "HF_HUB_CACHE": os.path.join(cache_base, "hf", "hub"),
        "TRANSFORMERS_CACHE": os.path.join(cache_base, "hf", "transformers"),
        "TORCH_HOME": os.path.join(cache_base, "torch"),
        "TIMM_HOME": os.path.join(cache_base, "timm"),
        "XDG_CACHE_HOME": cache_base,
    }
    tmp_map = {
        "TEMP": tmp_base,
        "TMP": tmp_base,
        "TMPDIR": tmp_base,
    }

    changed = {}
    for var, target in {**cache_map, **tmp_map}.items():
        os.makedirs(target, exist_ok=True)
        v = os.environ.get(var, "")
        if not v:
            continue
        dr = _drive_root(v)
        if dr and (not os.path.exists(dr)):
            os.environ[var] = target
            changed[var] = {"old": v, "new": target}

    if changed:
        print("[WARN] detected cache/tmp dirs on missing drive; redirected to project-local dirs")
        for k, vv in changed.items():
            print(f"  - {k}: {vv['old']} -> {vv['new']}")


def _is_freq_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("freq_")


def _is_re_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("re_")


def _is_sr_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("sr_")


def _is_residual_model(name: str) -> bool:
    return _is_re_model(name) or _is_sr_model(name)


def _is_fire_model(name: str) -> bool:
    return (name or "").lower().strip().startswith("fire_")


def _is_sfire_model(name: str) -> bool:
    n = (name or "").lower().strip()
    return n.startswith("sfire_") or n.startswith("spatial_fire_dualstream_")


def _move_to_device(x, device: str):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (tuple, list)):
        return type(x)(_move_to_device(xx, device) for xx in x)
    raise TypeError(f"Unsupported batch item type: {type(x)}")


def _as_two_views(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return x[0], x[1]
    return x, None


def _consistency_loss(logits1, logits2, kind: str = "mse", temperature: float = 1.0, detach_target: bool = True):
    t = float(max(1e-6, temperature))
    p1 = torch.softmax(logits1 / t, dim=1)
    p2 = torch.softmax(logits2 / t, dim=1)
    if detach_target:
        p2 = p2.detach()

    kind = (kind or "mse").lower().strip()
    if kind == "kl":
        eps = 1e-8
        logp1 = torch.log(p1.clamp_min(eps))
        logp2 = torch.log(p2.clamp_min(eps))
        return torch.mean(torch.sum(p1 * (logp1 - logp2), dim=1))

    return torch.mean((p1 - p2) ** 2)


def _contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    if z1.size(0) <= 1:
        return z1.new_zeros(())
    t = float(max(1e-6, temperature))
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.matmul(z1, z2.t()) / t
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


@torch.no_grad()
def evaluate(model, loader, device, model_name: str):
    model.eval()
    y_true, y_prob = [], []

    residual_mode = _is_residual_model(model_name)

    pbar = tqdm(
        loader,
        total=len(loader),
        desc="  [val]",
        unit="batch",
        ncols=110,
        ascii=True,
        dynamic_ncols=True,
        leave=False,
        file=sys.stdout,
        mininterval=0.2,
    )

    for x, y, _dom in pbar:
        if (not residual_mode) and isinstance(x, (tuple, list)):
            x = x[0]
        x = _move_to_device(x, device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Non-finite logits detected during evaluation.")
        if _is_fire_model(model_name):
            prob = torch.sigmoid(logits.reshape(-1)).detach().cpu().numpy()
        else:
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.numpy())

    y_true = np.concatenate(y_true).astype(int)
    y_prob = np.concatenate(y_prob).astype(float)
    return compute_binary_metrics(y_true, y_prob, thr=0.5)


def _save_training_state(
    path: str,
    *,
    model: nn.Module,
    optimizer: AdamW,
    scaler,
    epoch: int,
    best_auc: float,
    resolved: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_auc": float(best_auc),
        "resolved": resolved,
        "meta": meta or {},
    }
    if scaler is not None:
        try:
            payload["scaler"] = scaler.state_dict()
        except Exception:
            pass
    torch.save(payload, path)


def _load_resume_state(path: str, model: nn.Module, optimizer: AdamW | None = None, scaler=None):
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    model.load_state_dict(state, strict=True)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and "scaler" in payload:
        try:
            scaler.load_state_dict(payload["scaler"])
        except Exception:
            print("[WARN] failed to load scaler state from resume checkpoint; continuing without it.")
    start_epoch = int(payload.get("epoch", 0)) + 1
    best_auc = float(payload.get("best_auc", -1.0))
    return payload, start_epoch, best_auc


def _compute_class_weight(index_csv: str, domains: str, device: str):
    import pandas as pd

    df = pd.read_csv(index_csv)
    if "split" in df.columns:
        df = df[df["split"].astype(str) == "train"]
    if domains:
        doms = set([d.strip() for d in domains.split(",") if d.strip()])
        df = df[df["domain"].astype(str).isin(doms)]
    n0 = int((df["label"].astype(int) == 0).sum())
    n1 = int((df["label"].astype(int) == 1).sum())
    w0 = (n0 + n1) / max(1, 2 * n0)
    w1 = (n0 + n1) / max(1, 2 * n1)
    return torch.tensor([w0, w1], dtype=torch.float32, device=device)


def _maybe_load_branch_init(model, model_name: str, spatial_ckpt: str, fire_ckpt: str, strict: bool = False):
    if not _is_sfire_model(model_name):
        return {}
    if not hasattr(model, "load_pretrained_branches"):
        raise AttributeError(f"{model_name} does not implement load_pretrained_branches().")
    return model.load_pretrained_branches(spatial_ckpt=spatial_ckpt, fire_ckpt=fire_ckpt, strict=bool(strict))


def main():
    train_started_at = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--domains", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument("--ckpt_dir", default=".\\checkpoints")
    ap.add_argument("--paired_root", default="")

    # resume / initialization
    ap.add_argument("--resume", default="")
    ap.add_argument("--spatial_ckpt", default="")
    ap.add_argument("--fire_ckpt", default="")

    # data integrity / missing-file policy
    ap.add_argument("--missing_policy", default="", choices=["", "strict", "resample", "fallback"])
    ap.add_argument("--preflight_sample_count", type=int, default=500)
    ap.add_argument("--preflight_min_ok_ratio", type=float, default=0.98)

    # optional overrides
    ap.add_argument("--seed", type=int, default=-1)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--weight_decay", type=float, default=-1.0)
    ap.add_argument("--image_size", type=int, default=0)
    ap.add_argument("--model_name", default="")
    ap.add_argument("--grad_accum_steps", type=int, default=0)

    # legacy consistency regularization (non-paired only)
    ap.add_argument("--consis_lambda", type=float, default=0.0)
    ap.add_argument("--consis_kind", type=str, default="mse", choices=["mse", "kl"])
    ap.add_argument("--consis_temp", type=float, default=1.0)
    ap.add_argument("--consis_detach_target", action="store_true")
    ap.add_argument("--consis_p", type=float, default=1.0)
    ap.add_argument("--consis_jpeg_p", type=float, default=0.5)
    ap.add_argument("--consis_jpeg_qmin", type=int, default=30)
    ap.add_argument("--consis_jpeg_qmax", type=int, default=95)
    ap.add_argument("--consis_resize_p", type=float, default=0.5)
    ap.add_argument("--consis_resize_min", type=float, default=0.5)
    ap.add_argument("--consis_resize_max", type=float, default=1.0)
    ap.add_argument("--consis_blur_p", type=float, default=0.2)
    ap.add_argument("--consis_blur_rmin", type=float, default=0.5)
    ap.add_argument("--consis_blur_rmax", type=float, default=1.5)

    args = ap.parse_args()
    cfg = load_yaml(args.config)

    seed = int(args.seed if args.seed >= 0 else cfg.get("seed", 42))
    set_seed(seed)

    project_root = os.path.abspath(os.path.dirname(__file__))
    _fix_cache_dirs_if_bad(project_root)

    image_size = int(args.image_size or cfg.get("image_size", 224))
    epochs = int(args.epochs or cfg.get("epochs", 8))
    batch_size = int(args.batch_size or cfg.get("batch_size", 32))
    grad_accum_steps = int(args.grad_accum_steps or cfg.get("grad_accum_steps", 1))
    if grad_accum_steps < 1:
        grad_accum_steps = 1
    num_workers = int(cfg.get("num_workers", 0) if args.num_workers < 0 else args.num_workers)
    amp = bool(cfg.get("amp", True))

    model_cfg = cfg.get("model", {}) or {}
    model_name = (args.model_name.strip() or str(model_cfg.get("name", "spatial_resnet50"))).strip()
    use_fire = _is_fire_model(model_name)
    use_sfire = _is_sfire_model(model_name)

    pretrained = bool(model_cfg.get("pretrained", True))
    dropout = float(model_cfg.get("dropout", 0.2))

    base_lr = float(args.lr or (cfg.get("optim", {}) or {}).get("lr", 3e-4))
    wd = float((cfg.get("optim", {}) or {}).get("weight_decay", 1e-4) if args.weight_decay < 0 else args.weight_decay)
    grad_clip = float((cfg.get("optim", {}) or {}).get("grad_clip", 1.0))

    warmup_epochs = int((cfg.get("sched", {}) or {}).get("warmup_epochs", 1))
    min_lr = float((cfg.get("sched", {}) or {}).get("min_lr", 1e-6))

    use_residual_model = _is_residual_model(model_name)
    paired_root = args.paired_root.strip()

    data_cfg = cfg.get("data", {}) or {}
    missing_policy = (args.missing_policy.strip() or str(data_cfg.get("missing_policy", "strict")).strip() or "strict").lower()
    preflight_sample_count = int(data_cfg.get("preflight_sample_count", args.preflight_sample_count))
    preflight_min_ok_ratio = float(data_cfg.get("preflight_min_ok_ratio", args.preflight_min_ok_ratio))
    if preflight_sample_count < 0:
        preflight_sample_count = 0
    if preflight_min_ok_ratio < 0:
        preflight_min_ok_ratio = 0.0
    if preflight_min_ok_ratio > 1.0:
        preflight_min_ok_ratio = 1.0

    runtime_cfg = cfg.get("runtime", {}) or {}
    disable_vae_slicing = bool(runtime_cfg.get("disable_vae_slicing", False))
    disable_vae_tiling = bool(runtime_cfg.get("disable_vae_tiling", False))

    dual_cfg = cfg.get("dualstream", {}) or {}
    save_freq = int(dual_cfg.get("save_freq", 1))
    freeze_backbones_epochs = int(dual_cfg.get("freeze_backbones_epochs", 0))
    aux_spatial_weight = float(dual_cfg.get("aux_spatial_weight", 0.3))
    aux_fire_weight = float(dual_cfg.get("aux_fire_weight", 0.3))
    contrastive_weight = float(dual_cfg.get("contrastive_weight", 0.05))
    contrastive_temp = float(dual_cfg.get("contrastive_temp", 0.07))
    router_balance_weight = float(dual_cfg.get("router_balance_weight", 0.02))
    router_entropy_weight = float(dual_cfg.get("router_entropy_weight", 0.005))
    branch_init_strict = bool(dual_cfg.get("branch_init_strict", False))

    spatial_ckpt = args.spatial_ckpt.strip() or str(dual_cfg.get("spatial_ckpt", "")).strip()
    fire_ckpt = args.fire_ckpt.strip() or str(dual_cfg.get("fire_ckpt", "")).strip()

    if use_residual_model and (not paired_root):
        raise ValueError(f"{model_name} requires --paired_root pointing to precomputed residual maps.")
    if (not use_residual_model) and paired_root:
        print(f"[WARN] ignoring --paired_root for non-residual model: {model_name}")

    use_consis = (float(args.consis_lambda) > 0) and (not use_fire) and (not use_sfire)
    if use_residual_model and use_consis:
        raise NotImplementedError(
            "Consistency regularization for re_/sr_ paired residual models is not implemented in this version. "
            "Please train re_/sr_ models with --consis_lambda 0."
        )

    run_name = args.run_name.strip() or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = os.path.join(args.ckpt_dir, run_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "tb"))

    resolved = {
        "seed": seed,
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "num_workers": num_workers,
        "model_name": model_name,
        "pretrained": pretrained,
        "dropout": dropout,
        "lr": base_lr,
        "weight_decay": wd,
        "grad_clip": grad_clip,
        "warmup_epochs": warmup_epochs,
        "min_lr": min_lr,
        "domains": args.domains,
        "index_csv": args.index_csv,
        "data_root": args.data_root,
        "paired_root": paired_root,
        "resume": args.resume,
        "missing_policy": missing_policy,
        "preflight_sample_count": preflight_sample_count,
        "preflight_min_ok_ratio": preflight_min_ok_ratio,
        "runtime": {
            "disable_vae_slicing": disable_vae_slicing,
            "disable_vae_tiling": disable_vae_tiling,
        },
        "dualstream": {
            "save_freq": save_freq,
            "freeze_backbones_epochs": freeze_backbones_epochs,
            "aux_spatial_weight": aux_spatial_weight,
            "aux_fire_weight": aux_fire_weight,
            "contrastive_weight": contrastive_weight,
            "contrastive_temp": contrastive_temp,
            "router_balance_weight": router_balance_weight,
            "router_entropy_weight": router_entropy_weight,
            "spatial_ckpt": spatial_ckpt,
            "fire_ckpt": fire_ckpt,
            "branch_init_strict": branch_init_strict,
        },
        "consis_lambda": float(args.consis_lambda),
        "consis_kind": str(args.consis_kind),
        "consis_temp": float(args.consis_temp),
        "consis_detach_target": bool(args.consis_detach_target),
        "consis_p": float(args.consis_p),
        "consis_jpeg_p": float(args.consis_jpeg_p),
        "consis_jpeg_qmin": int(args.consis_jpeg_qmin),
        "consis_jpeg_qmax": int(args.consis_jpeg_qmax),
        "consis_resize_p": float(args.consis_resize_p),
        "consis_resize_min": float(args.consis_resize_min),
        "consis_resize_max": float(args.consis_resize_max),
        "consis_blur_p": float(args.consis_blur_p),
        "consis_blur_rmin": float(args.consis_blur_rmin),
        "consis_blur_rmax": float(args.consis_blur_rmax),
    }
    save_json(os.path.join(run_dir, "resolved.json"), resolved)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if _is_freq_model(model_name) and pretrained:
        print("[WARN] freq model: forcing pretrained=False to avoid timm/HF weight download & cache path issues.")
        pretrained = False

    num_classes = 1 if use_fire else 2
    model = build_model(model_name, num_classes=num_classes, pretrained=pretrained, dropout=dropout).to(device)

    fire_module = model if use_fire else getattr(model, "fire_branch", None)
    vae = getattr(fire_module, "vae", None)
    if vae is not None and disable_vae_slicing and hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
        print("[INFO] disabled FIRE VAE slicing for measured batch-2 throughput.")
    if vae is not None and disable_vae_tiling and hasattr(vae, "disable_tiling"):
        vae.disable_tiling()
        print("[INFO] disabled FIRE VAE tiling for 256x256 inputs.")

    if use_sfire:
        if not spatial_ckpt or not fire_ckpt:
            raise ValueError(
                f"{model_name} requires both spatial_ckpt and fire_ckpt for branch initialization. "
                f"Got spatial_ckpt={spatial_ckpt!r}, fire_ckpt={fire_ckpt!r}"
            )
        init_report = _maybe_load_branch_init(model, model_name, spatial_ckpt, fire_ckpt, strict=branch_init_strict)
        print("[INFO] dual-stream branch init report:")
        for branch_name, branch_report in init_report.items():
            miss = len(branch_report.get("missing", []))
            unexp = len(branch_report.get("unexpected", []))
            print(f"  - {branch_name}: missing={miss}, unexpected={unexp}")

    if missing_policy in ["strict", "resample"] and preflight_sample_count > 0:
        train_report = inspect_sample_paths(
            index_csv=args.index_csv,
            data_root=args.data_root,
            split="train",
            domains=args.domains if args.domains else None,
            paired_root=paired_root if use_residual_model else None,
            sample_count=preflight_sample_count,
            seed=seed,
        )
        print(
            f"[INFO] preflight(train): ok={train_report['ok_count']}/{train_report['checked_count']} "
            f"ratio={train_report['ok_ratio']:.4f} data_root={args.data_root}"
        )
        if train_report["ok_ratio"] < preflight_min_ok_ratio:
            raise RuntimeError(
                "Training data preflight failed: "
                f"ok_ratio={train_report['ok_ratio']:.4f} < min_ok_ratio={preflight_min_ok_ratio:.4f}. "
                "Please finish decompression / fix data_root / fix CSV paths before formal training."
            )

        val_report = inspect_sample_paths(
            index_csv=args.index_csv,
            data_root=args.data_root,
            split="val",
            domains=args.domains if args.domains else None,
            paired_root=paired_root if use_residual_model else None,
            sample_count=min(max(64, preflight_sample_count // 2), preflight_sample_count),
            seed=seed + 1,
        )
        print(
            f"[INFO] preflight(val): ok={val_report['ok_count']}/{val_report['checked_count']} "
            f"ratio={val_report['ok_ratio']:.4f} data_root={args.data_root}"
        )
        if val_report["ok_ratio"] < preflight_min_ok_ratio:
            raise RuntimeError(
                "Validation data preflight failed: "
                f"ok_ratio={val_report['ok_ratio']:.4f} < min_ok_ratio={preflight_min_ok_ratio:.4f}. "
                "Please finish decompression / fix data_root / fix CSV paths before formal training."
            )

    if use_residual_model:
        paired_train_tf = build_paired_transforms_for_model(model_name, image_size=image_size, is_train=True)
        paired_val_tf = build_paired_transforms_for_model(model_name, image_size=image_size, is_train=False)
        pair_mode = "aux_only" if _is_re_model(model_name) else "pair"

        ds_train = CSVIndexDataset(
            args.index_csv,
            args.data_root,
            split="train",
            domains=args.domains if args.domains else None,
            paired_root=paired_root,
            paired_transform=paired_train_tf,
            pair_mode=pair_mode,
            missing_policy=missing_policy,
        )
        ds_val = CSVIndexDataset(
            args.index_csv,
            args.data_root,
            split="val",
            domains=args.domains if args.domains else None,
            paired_root=paired_root,
            paired_transform=paired_val_tf,
            pair_mode=pair_mode,
            missing_policy=missing_policy,
        )
    else:
        train_tf = build_transforms_for_model(model_name, image_size=image_size, is_train=True)
        val_tf = build_transforms_for_model(model_name, image_size=image_size, is_train=False)

        train_tf2 = None
        if use_consis:
            aug = build_consistency_augment(
                p=args.consis_p,
                jpeg_p=args.consis_jpeg_p,
                jpeg_qmin=args.consis_jpeg_qmin,
                jpeg_qmax=args.consis_jpeg_qmax,
                resize_p=args.consis_resize_p,
                resize_scale_min=args.consis_resize_min,
                resize_scale_max=args.consis_resize_max,
                blur_p=args.consis_blur_p,
                blur_radius_min=args.consis_blur_rmin,
                blur_radius_max=args.consis_blur_rmax,
            )
            train_tf2 = wrap_transform_with_pre(train_tf, aug)
            print(
                f"[INFO] consistency enabled: lambda={args.consis_lambda}, kind={args.consis_kind}, temp={args.consis_temp}, "
                f"detach_target={bool(args.consis_detach_target)}"
            )

        ds_train = CSVIndexDataset(
            args.index_csv,
            args.data_root,
            split="train",
            domains=args.domains if args.domains else None,
            transform=train_tf,
            transform2=train_tf2,
            missing_policy=missing_policy,
        )
        ds_val = CSVIndexDataset(
            args.index_csv,
            args.data_root,
            split="val",
            domains=args.domains if args.domains else None,
            transform=val_tf,
            missing_policy=missing_policy,
        )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_batch_size = batch_size if (use_fire or use_sfire) else max(32, batch_size)
    dl_val = DataLoader(
        ds_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    weight = _compute_class_weight(args.index_csv, args.domains, device=device)
    optimizer = AdamW(model.parameters(), lr=base_lr, weight_decay=wd)

    # FIRE keeps its frozen VAE reconstruction path in its explicitly selected
    # dtype/autocast context. Enabling AMP here safely accelerates the trainable
    # ResNet/fusion portions and is controlled per experiment config.
    use_amp = bool(amp and device == "cuda")
    if hasattr(torch, "amp"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        autocast = lambda: torch.amp.autocast("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        autocast = lambda: torch.cuda.amp.autocast(enabled=use_amp)

    criterion = nn.BCEWithLogitsLoss() if use_fire else nn.CrossEntropyLoss(weight=weight)
    writer = SummaryWriter(log_dir=os.path.join(run_dir, "tb"))

    best_auc = -1.0
    start_epoch = 1
    resume_path = args.resume.strip()
    if resume_path:
        payload, start_epoch, best_auc = _load_resume_state(resume_path, model=model, optimizer=optimizer, scaler=scaler)
        print(f"[INFO] resumed from {resume_path} | next_epoch={start_epoch} | best_auc={best_auc:.6f}")

    total_steps = epochs * max(1, len(dl_train))
    warmup_steps = warmup_epochs * max(1, len(dl_train))
    current_epoch = max(0, start_epoch - 1)

    def _checkpoint_meta(epoch: int, val_metrics: Dict[str, Any] | None = None, status: str = "running"):
        return {"val": val_metrics or {}, "status": status}

    try:
        for epoch in range(start_epoch, epochs + 1):
            current_epoch = epoch
            model.train()
            if use_sfire and hasattr(model, "set_backbones_trainable"):
                freeze_now = epoch <= max(0, freeze_backbones_epochs)
                model.set_backbones_trainable(not freeze_now)
                if freeze_now:
                    print(f"[INFO] dual-stream warmup phase: frozen branch backbones at epoch {epoch}/{epochs}")

            epoch_loss = 0.0
            epoch_cons = 0.0
            epoch_aux_s = 0.0
            epoch_aux_f = 0.0
            epoch_ctr = 0.0
            epoch_router = 0.0

            pbar = tqdm(
                dl_train,
                total=len(dl_train),
                desc=f"Epoch {epoch}/{epochs} [train]",
                unit="batch",
                ncols=110,
                ascii=True,
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                mininterval=0.2,
            )

            optimizer.zero_grad(set_to_none=True)

            for it, (x, y, _dom) in enumerate(pbar, start=1):
                if use_consis and (not use_residual_model):
                    x1, x2 = _as_two_views(x)
                else:
                    x1, x2 = x, None

                x1 = _move_to_device(x1, device)
                if x2 is not None:
                    x2 = _move_to_device(x2, device)

                y = y.to(device, non_blocking=True)

                step = (epoch - 1) * max(1, len(dl_train)) + (it - 1)
                progress = step / max(1, total_steps - 1)

                if step < warmup_steps:
                    lr = base_lr * float(step + 1) / float(max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(lr)
                else:
                    lr = cosine_lr(optimizer, base_lr, min_lr, progress)

                with autocast():
                    cons_loss = None
                    fire_loss_b = None
                    fire_loss_mse_rec = None
                    fire_loss_mse_mask = None
                    sfire_loss_main = None
                    sfire_loss_aux_s = None
                    sfire_loss_aux_f = None
                    sfire_loss_ctr = None
                    sfire_loss_router = None

                    if use_fire:
                        fire_out = model(x1, return_aux=True)
                        logits1 = fire_out["logits"]
                        fire_loss_mse_rec = F.mse_loss(fire_out["middle_freq_image"], fire_out["raw_reconstructions_delta"])
                        all_mask = torch.ones_like(fire_out["ideal_comp_mask"])
                        loss_mse_mask_mid_frq = F.mse_loss(fire_out["mask_mid_frq"], fire_out["ideal_mid_mask"])
                        loss_mse_mask_mid_filterd = F.mse_loss(fire_out["mask_mid_filterd"], fire_out["ideal_comp_mask"])
                        loss_mse_mask_norm = F.mse_loss(fire_out["mask_mid_frq"] + fire_out["mask_mid_filterd"], all_mask)
                        fire_loss_mse_mask = loss_mse_mask_mid_frq + loss_mse_mask_mid_filterd + loss_mse_mask_norm
                        fire_loss_b = criterion(logits1.reshape(-1), y.float())
                        loss_sup = fire_loss_b
                        loss = 0.6 * fire_loss_b + 0.2 * fire_loss_mse_rec + 0.2 * fire_loss_mse_mask
                    elif use_sfire:
                        dual_out = model(x1, return_aux=True)
                        logits1 = dual_out["logits"]
                        sfire_loss_main = criterion(logits1, y)
                        sfire_loss_aux_s = criterion(dual_out["spatial_logits"], y)
                        sfire_loss_aux_f = criterion(dual_out["fire_logits_2c"], y)
                        sfire_loss_ctr = _contrastive_loss(
                            dual_out["spatial_proj"], dual_out["fire_proj"], temperature=contrastive_temp
                        )
                        sfire_loss_router = (
                            float(router_balance_weight) * dual_out["router_balance"]
                            - float(router_entropy_weight) * dual_out["router_entropy"]
                        )
                        loss_sup = sfire_loss_main
                        loss = (
                            sfire_loss_main
                            + float(aux_spatial_weight) * sfire_loss_aux_s
                            + float(aux_fire_weight) * sfire_loss_aux_f
                            + float(contrastive_weight) * sfire_loss_ctr
                            + sfire_loss_router
                        )
                    else:
                        logits1 = model(x1)
                        loss_sup = criterion(logits1, y)
                        loss = loss_sup
                        if use_consis and (x2 is not None):
                            logits2 = model(x2)
                            cons_loss = _consistency_loss(
                                logits1,
                                logits2,
                                kind=args.consis_kind,
                                temperature=args.consis_temp,
                                detach_target=bool(args.consis_detach_target),
                            )
                            loss = loss + float(args.consis_lambda) * cons_loss

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss detected at epoch={epoch}, iter={it}: "
                        f"loss={float(loss.detach().cpu())}, loss_sup={float(loss_sup.detach().cpu())}"
                    )

                loss_to_backprop = loss / float(grad_accum_steps)
                scaler.scale(loss_to_backprop).backward()

                do_step = (it % grad_accum_steps == 0) or (it == len(dl_train))
                if do_step:
                    if grad_clip > 0:
                        if use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += float(loss.item() if use_fire or use_sfire else loss_sup.item())
                if cons_loss is not None:
                    epoch_cons += float(cons_loss.item())
                if use_sfire:
                    epoch_aux_s += float(sfire_loss_aux_s.item())
                    epoch_aux_f += float(sfire_loss_aux_f.item())
                    epoch_ctr += float(sfire_loss_ctr.item())
                    epoch_router += float(sfire_loss_router.item())

                if it == 1 or it % 20 == 0:
                    if use_fire:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}", bce=f"{fire_loss_b.item():.4f}", rec=f"{fire_loss_mse_rec.item():.4f}",
                            mask=f"{fire_loss_mse_mask.item():.4f}", lr=f"{lr:.2e}"
                        )
                    elif use_sfire:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}", main=f"{sfire_loss_main.item():.4f}", auxs=f"{sfire_loss_aux_s.item():.4f}",
                            auxf=f"{sfire_loss_aux_f.item():.4f}", ctr=f"{sfire_loss_ctr.item():.4f}", lr=f"{lr:.2e}"
                        )
                    elif cons_loss is None:
                        pbar.set_postfix(loss=f"{loss_sup.item():.4f}", lr=f"{lr:.2e}")
                    else:
                        pbar.set_postfix(loss=f"{loss_sup.item():.4f}", cons=f"{cons_loss.item():.4f}", lr=f"{lr:.2e}")

                if it % 50 == 0:
                    global_step = step + 1
                    writer.add_scalar("train/loss", float(loss.item() if use_fire or use_sfire else loss_sup.item()), global_step)
                    writer.add_scalar("train/lr", float(lr), global_step)
                    if use_fire:
                        writer.add_scalar("train/fire_bce", float(fire_loss_b.item()), global_step)
                        writer.add_scalar("train/fire_mse_rec", float(fire_loss_mse_rec.item()), global_step)
                        writer.add_scalar("train/fire_mse_mask", float(fire_loss_mse_mask.item()), global_step)
                    elif use_sfire:
                        writer.add_scalar("train/dual_main", float(sfire_loss_main.item()), global_step)
                        writer.add_scalar("train/dual_aux_spatial", float(sfire_loss_aux_s.item()), global_step)
                        writer.add_scalar("train/dual_aux_fire", float(sfire_loss_aux_f.item()), global_step)
                        writer.add_scalar("train/dual_contrastive", float(sfire_loss_ctr.item()), global_step)
                        writer.add_scalar("train/dual_router", float(sfire_loss_router.item()), global_step)
                    elif cons_loss is not None:
                        writer.add_scalar("train/cons_loss", float(cons_loss.item()), global_step)

            epoch_loss /= max(1, len(dl_train))
            writer.add_scalar("train/epoch_loss", epoch_loss, epoch)
            if use_consis:
                epoch_cons /= max(1, len(dl_train))
                writer.add_scalar("train/epoch_cons_loss", epoch_cons, epoch)
            if use_sfire:
                epoch_aux_s /= max(1, len(dl_train))
                epoch_aux_f /= max(1, len(dl_train))
                epoch_ctr /= max(1, len(dl_train))
                epoch_router /= max(1, len(dl_train))
                writer.add_scalar("train/epoch_dual_aux_spatial", epoch_aux_s, epoch)
                writer.add_scalar("train/epoch_dual_aux_fire", epoch_aux_f, epoch)
                writer.add_scalar("train/epoch_dual_contrastive", epoch_ctr, epoch)
                writer.add_scalar("train/epoch_dual_router", epoch_router, epoch)

            val_metrics = evaluate(model, dl_val, device, model_name=model_name)
            writer.add_scalar("val/auc", val_metrics["auc"], epoch)
            writer.add_scalar("val/ap", val_metrics["ap"], epoch)

            if use_sfire:
                print(
                    f"[Epoch {epoch}/{epochs}] loss={epoch_loss:.4f} auxs={epoch_aux_s:.4f} auxf={epoch_aux_f:.4f} "
                    f"ctr={epoch_ctr:.4f} router={epoch_router:.4f} | "
                    f"val auc={val_metrics['auc']:.4f} ap={val_metrics['ap']:.4f} "
                    f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
                )
            elif use_consis:
                print(
                    f"[Epoch {epoch}/{epochs}] loss={epoch_loss:.4f} cons={epoch_cons:.4f} | "
                    f"val auc={val_metrics['auc']:.4f} ap={val_metrics['ap']:.4f} "
                    f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
                )
            else:
                print(
                    f"[Epoch {epoch}/{epochs}] loss={epoch_loss:.4f} | "
                    f"val auc={val_metrics['auc']:.4f} ap={val_metrics['ap']:.4f} "
                    f"acc={val_metrics['acc']:.4f} f1={val_metrics['f1']:.4f}"
                )

            latest_paths = [os.path.join(run_dir, "latest.pth"), os.path.join(run_dir, "last.pth")]
            for latest_path in latest_paths:
                _save_training_state(
                    latest_path,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_auc=best_auc,
                    resolved=resolved,
                    meta=_checkpoint_meta(epoch, val_metrics=val_metrics, status="running"),
                )

            if save_freq > 0 and (epoch % save_freq == 0):
                _save_training_state(
                    os.path.join(run_dir, f"epoch_{epoch:03d}.pth"),
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_auc=best_auc,
                    resolved=resolved,
                    meta=_checkpoint_meta(epoch, val_metrics=val_metrics, status="epoch_save"),
                )

            if val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                _save_training_state(
                    os.path.join(run_dir, "best_by_auc.pth"),
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    epoch=epoch,
                    best_auc=best_auc,
                    resolved=resolved,
                    meta=_checkpoint_meta(epoch, val_metrics=val_metrics, status="best"),
                )
                save_json(os.path.join(run_dir, "best_metrics.json"), {"epoch": epoch, **val_metrics})
                print(f"  [*] New best_by_auc: {best_auc:.6f}")

        save_json(
            os.path.join(run_dir, "training_complete.json"),
            {
                "status": "complete",
                "completed_epochs": int(current_epoch),
                "best_auc": float(best_auc),
                "elapsed_seconds": float(time.time() - train_started_at),
                "seed": int(seed),
                "model_name": model_name,
                "index_csv": os.path.abspath(args.index_csv),
                "data_root": os.path.abspath(args.data_root),
            },
        )
        print(f"Done. Best val AUC={best_auc:.6f} | run_dir={run_dir}")

    except KeyboardInterrupt:
        interrupt_path = os.path.join(run_dir, f"interrupt_epoch_{current_epoch:03d}.pth")
        _save_training_state(
            interrupt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=current_epoch,
            best_auc=best_auc,
            resolved=resolved,
            meta=_checkpoint_meta(current_epoch, status="keyboard_interrupt"),
        )
        print(f"\n[WARN] interrupted by user. Saved resumable checkpoint to: {interrupt_path}")
        raise
    except Exception:
        interrupt_path = os.path.join(run_dir, f"interrupt_epoch_{current_epoch:03d}.pth")
        try:
            _save_training_state(
                interrupt_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=current_epoch,
                best_auc=best_auc,
                resolved=resolved,
                meta=_checkpoint_meta(current_epoch, status="exception"),
            )
            print(f"[WARN] exception checkpoint saved to: {interrupt_path}")
        finally:
            writer.close()
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
