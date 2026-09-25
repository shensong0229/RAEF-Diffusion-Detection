"""Short, checkpoint-free throughput benchmark for the SDV5 paper pipeline."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset.csv_dataset import CSVIndexDataset
from core.dataset.transforms import build_transforms_for_model
from core.models import build_model


INDEX_CSV = ROOT / "indexes" / "paper_sdv5_single_source" / "trainval_sdv5_100k_10k.csv"
DATA_ROOT = ROOT / "data"
MODEL_NAMES = {
    "spatial": "spatial_resnet50",
    "fire": "fire_resnet50",
    "fusion_frozen": "sfire_crossattn_resnet50",
    "fusion_unfrozen": "sfire_crossattn_resnet50",
}


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.07):
    if z1.size(0) <= 1:
        return z1.new_zeros(())
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_loss(stage: str, model, x, y):
    if stage == "spatial":
        return F.cross_entropy(model(x), y)
    if stage == "fire":
        out = model(x, return_aux=True)
        bce = F.binary_cross_entropy_with_logits(out["logits"].reshape(-1), y.float())
        rec = F.mse_loss(out["middle_freq_image"], out["raw_reconstructions_delta"])
        ones = torch.ones_like(out["ideal_comp_mask"])
        mask = (
            F.mse_loss(out["mask_mid_frq"], out["ideal_mid_mask"])
            + F.mse_loss(out["mask_mid_filterd"], out["ideal_comp_mask"])
            + F.mse_loss(out["mask_mid_frq"] + out["mask_mid_filterd"], ones)
        )
        return 0.6 * bce + 0.2 * rec + 0.2 * mask
    out = model(x, return_aux=True)
    main = F.cross_entropy(out["logits"], y)
    aux_s = F.cross_entropy(out["spatial_logits"], y)
    aux_f = F.cross_entropy(out["fire_logits_2c"], y)
    ctr = contrastive_loss(out["spatial_proj"], out["fire_proj"])
    router = 0.02 * out["router_balance"] - 0.005 * out["router_entropy"]
    return main + 0.3 * aux_s + 0.3 * aux_f + 0.05 * ctr + router


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=list(MODEL_NAMES))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable_vae_slicing", action="store_true")
    parser.add_argument("--disable_vae_tiling", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the laptop throughput benchmark.")
    os.environ.setdefault("FIRE_VAE_DIR", str(ROOT / "pretrained" / "sd15_vae"))
    stage = args.stage
    model_name = MODEL_NAMES[stage]
    transform = build_transforms_for_model(model_name, image_size=args.image_size, is_train=True)
    dataset = CSVIndexDataset(
        str(INDEX_CSV), str(DATA_ROOT), split="train", transform=transform, missing_policy="resample"
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    model = build_model(
        model_name,
        num_classes=1 if stage == "fire" else 2,
        pretrained=(stage == "spatial"),
        dropout=0.2 if stage == "spatial" else 0.0,
    ).cuda()
    fire_module = model if stage == "fire" else getattr(model, "fire_branch", None)
    vae = getattr(fire_module, "vae", None)
    if vae is not None and args.disable_vae_slicing and hasattr(vae, "disable_slicing"):
        vae.disable_slicing()
    if vae is not None and args.disable_vae_tiling and hasattr(vae, "disable_tiling"):
        vae.disable_tiling()
    model.train()
    if stage == "fusion_frozen":
        model.set_backbones_trainable(False)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4)
    use_amp = stage == "spatial" or bool(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    iterator = iter(loader)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    data_durations = []
    losses = []
    total = args.warmup + args.steps
    for step in range(total):
        data_start = time.perf_counter()
        x, y, _ = next(iterator)
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)
        torch.cuda.synchronize()
        data_elapsed = time.perf_counter() - data_start
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = compute_loss(stage, model, x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if step >= args.warmup:
            durations.append(elapsed)
            data_durations.append(data_elapsed)
            losses.append(float(loss.detach().cpu()))

    seconds_per_batch = sum(durations) / len(durations)
    seconds_data_per_batch = sum(data_durations) / len(data_durations)
    seconds_end_to_end = seconds_per_batch + seconds_data_per_batch
    train_batches = math.ceil(len(dataset) / args.batch_size)
    report = {
        "stage": stage,
        "model_name": model_name,
        "batch_size": args.batch_size,
        "amp": use_amp,
        "vae_slicing_disabled": bool(args.disable_vae_slicing),
        "vae_tiling_disabled": bool(args.disable_vae_tiling),
        "timed_steps": args.steps,
        "seconds_per_batch_compute_only": seconds_per_batch,
        "seconds_per_batch_data_and_transfer": seconds_data_per_batch,
        "seconds_per_batch_end_to_end": seconds_end_to_end,
        "train_samples": len(dataset),
        "estimated_compute_hours_per_epoch": seconds_per_batch * train_batches / 3600.0,
        "estimated_end_to_end_hours_per_epoch": seconds_end_to_end * train_batches / 3600.0,
        "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / (1024 ** 3),
        "loss_mean": sum(losses) / len(losses),
        "note": "Estimate excludes validation, checkpoint I/O and most data-loading overhead.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
