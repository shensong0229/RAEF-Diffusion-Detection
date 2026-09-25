"""Measure end-to-end model-forward cost for Reviewer 3 on the local GPU.

All measurements start with a preprocessed 256x256 RGB tensor and include
frequency filtering, both VAE reconstructions, both branches, and fusion.
Disk image decoding is excluded. The two test-time interventions use the
same trained checkpoint and parameter set as the full model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
os.environ.setdefault("FIRE_VAE_DIR", str(PROJECT / "pretrained" / "sd15_vae"))

from core.dataset.transforms import build_transforms_for_model
from core.models.builder import build_model


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def trainable_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = int(position), min(len(ordered) - 1, int(position) + 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def measure(model, image, mode: str, warmup: int, repeats: int) -> dict:
    if mode == "no_anomaly_guidance":
        model.fusion_variant = "standard_crossattn"
    else:
        model.fusion_variant = "full"
    hook = None
    if mode == "no_reliability_routing":
        hook = model.router.register_forward_hook(
            lambda _module, _inputs, output: torch.full_like(output, 0.5)
        )
    wall_times = []
    gpu_times = []
    try:
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=False):
            for _ in range(warmup):
                output = model(image)
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Non-finite warmup output: {mode}")
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated = torch.cuda.memory_allocated()
            baseline_reserved = torch.cuda.memory_reserved()
            for _ in range(repeats):
                torch.cuda.synchronize()
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_wall = time.perf_counter()
                start_event.record()
                output = model(image)
                end_event.record()
                torch.cuda.synchronize()
                wall_times.append((time.perf_counter() - start_wall) * 1000.0)
                gpu_times.append(float(start_event.elapsed_time(end_event)))
                if not torch.isfinite(output).all():
                    raise FloatingPointError(f"Non-finite measured output: {mode}")
            peak_allocated = torch.cuda.max_memory_allocated()
            peak_reserved = torch.cuda.max_memory_reserved()
    finally:
        if hook is not None:
            hook.remove()
        model.fusion_variant = "full"
    return {
        "mode": mode,
        "warmup_runs": warmup,
        "measured_runs": repeats,
        "wall_mean_ms": statistics.mean(wall_times),
        "wall_median_ms": statistics.median(wall_times),
        "wall_p95_ms": percentile(wall_times, 0.95),
        "gpu_event_mean_ms": statistics.mean(gpu_times),
        "gpu_event_median_ms": statistics.median(gpu_times),
        "baseline_allocated_mib": baseline_allocated / 2**20,
        "peak_allocated_mib": peak_allocated / 2**20,
        "incremental_peak_allocated_mib": (peak_allocated - baseline_allocated) / 2**20,
        "baseline_reserved_mib": baseline_reserved / 2**20,
        "peak_reserved_mib": peak_reserved / 2**20,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=PROJECT / "checkpoints" / "sfire_best" / "best_by_auc.pth")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "results" / "reviewer3_test_time_ablation")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.warmup < 1 or args.repeats < 3:
        raise ValueError("At least one warmup and three measured runs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the device-specific latency/memory report")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    properties = torch.cuda.get_device_properties(0)
    free_before, total_memory = torch.cuda.mem_get_info()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model("sfire_crossattn_resnet50", num_classes=2,
                        pretrained=False, dropout=0.1)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.set_backbones_trainable(False)
    warmup_trainable = trainable_parameters(model)
    model.set_backbones_trainable(True)
    joint_trainable = trainable_parameters(model)
    model.cuda().eval()
    model.to(memory_format=torch.channels_last)
    torch.cuda.synchronize()

    test_image = (PROJECT / "exports" / "dragon_eval_25domains_unique_real_1k_bundle" /
                  "images" / "test" / "Flash_PixArt" / "nature" /
                  "31d264b828_n03843555_294.JPEG")
    if not test_image.is_file():
        raise FileNotFoundError(test_image)
    transform = build_transforms_for_model("sfire_crossattn_resnet50", 256, False)
    with Image.open(test_image) as source:
        tensor = transform(source.convert("RGB")).unsqueeze(0)
    image = tensor.cuda().contiguous(memory_format=torch.channels_last)

    spatial_count = count_parameters(model.spatial_branch)
    fire_count = count_parameters(model.fire_branch)
    total_count = count_parameters(model)
    report = {
        "protocol": "single-image 256x256 end-to-end model forward; excludes disk I/O and image decoding",
        "checkpoint": str(args.checkpoint),
        "hardware": {
            "gpu": properties.name,
            "gpu_total_memory_mib": total_memory / 2**20,
            "gpu_free_before_load_mib": free_before / 2**20,
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "platform": platform.platform(),
        },
        "precision": "FP32 outer forward; VAE uses its checkpoint half-precision dtype",
        "batch_size": 1,
        "input_shape": list(image.shape),
        "warmup_runs": args.warmup,
        "measured_runs": args.repeats,
        "parameter_count": {
            "total": total_count,
            "spatial_branch": spatial_count,
            "fire_branch_including_vae": fire_count,
            "fusion_and_projection": total_count - spatial_count - fire_count,
            "trainable_initial_frozen_branch_phase": warmup_trainable,
            "trainable_joint_finetuning_phase": joint_trainable,
        },
        "feature_dimensions": {
            "spatial_global": model.spatial_dim,
            "fire_global": model.fire_dim,
            "fusion_embedding": model.fuse_dim,
        },
        "interventions_share_all_trained_parameters": True,
        "measurements": [],
    }
    output_path = args.output_dir / "compute_cost.json"
    for mode in ("full", "no_anomaly_guidance", "no_reliability_routing"):
        measurement = measure(model, image, mode, args.warmup, args.repeats)
        report["measurements"].append(measurement)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[COST] {mode}: median {measurement['wall_median_ms']:.2f} ms, "
              f"peak {measurement['peak_allocated_mib']:.1f} MiB", flush=True)
    csv_path = args.output_dir / "compute_cost.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report["measurements"][0]))
        writer.writeheader()
        writer.writerows(report["measurements"])
    print(f"[COMPLETE] {output_path}", flush=True)


if __name__ == "__main__":
    main()
