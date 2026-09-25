"""Verify that cached fusion algebra matches the manuscript full model."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dataset.csv_dataset import CSVIndexDataset
from core.dataset.transforms import build_transforms_for_model
from core.models import build_model
from core.models.sfire_cached_fusion import CachedSpatialFIREFusion


FULL_CKPT = ROOT / "checkpoints" / "run_sfire_best_test" / "best_by_auc.pth"
SPATIAL_CKPT = ROOT / "checkpoints" / "run_spatial_imagenet_adm_256" / "best_by_auc.pth"
FIRE_CKPT = ROOT / "checkpoints" / "run_fire_imagenet_adm_a30_std_fg" / "best_by_auc.pth"
INDEX = ROOT / "paper_assets" / "reviewer1_comment5_ablation" / "smoke_train_index.csv"
DATA_ROOT = ROOT / "data"
OUT = ROOT / "paper_assets" / "reviewer1_comment5_fast" / "cached_full_equivalence_report.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model", payload)
    cleaned = {}
    for key, value in state.items():
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        cleaned[key] = value
    return cleaned


def tensor_error(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.detach().float().cpu()
    candidate = candidate.detach().float().cpu()
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-12)
    return {
        "max_abs": float(difference.abs().max()),
        "mean_abs": float(difference.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(difference) / denominator),
    }


def state_drift(full_state: dict[str, torch.Tensor], branch_state: dict[str, torch.Tensor], prefix: str) -> dict:
    squared_difference = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    changed = 0
    compared = 0
    missing = []
    for key, reference in branch_state.items():
        full_key = prefix + key
        if full_key not in full_state:
            missing.append(key)
            continue
        candidate = full_state[full_key]
        if candidate.shape != reference.shape:
            missing.append(key)
            continue
        compared += 1
        if not (torch.is_floating_point(reference) or torch.is_complex(reference)):
            changed += int(not torch.equal(candidate, reference))
            continue
        difference = candidate.float() - reference.float()
        squared_difference += float(torch.sum(difference * difference))
        squared_reference += float(torch.sum(reference.float() * reference.float()))
        max_abs = max(max_abs, float(difference.abs().max()))
        changed += int(bool(torch.any(difference != 0)))
    return {
        "compared_tensors": compared,
        "changed_tensors": changed,
        "missing_or_shape_mismatch": missing,
        "max_abs": max_abs,
        "relative_l2": (squared_difference / max(squared_reference, 1e-24)) ** 0.5,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the equivalence check.")

    full_payload = torch.load(FULL_CKPT, map_location="cpu")
    full_state = checkpoint_state(FULL_CKPT)
    spatial_state = checkpoint_state(SPATIAL_CKPT)
    fire_state = checkpoint_state(FIRE_CKPT)

    branch_drift = {
        "spatial": state_drift(full_state, spatial_state, "spatial_branch."),
        "fire": state_drift(full_state, fire_state, "fire_branch."),
    }

    transform = build_transforms_for_model("sfire_crossattn_resnet50", image_size=256, is_train=False)
    dataset = CSVIndexDataset(str(INDEX), str(DATA_ROOT), split="train", transform=transform, missing_policy="strict")
    images, labels, domains = next(iter(DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)))
    images = images.cuda()

    full = build_model("sfire_crossattn_resnet50", num_classes=2, pretrained=False, dropout=0.1)
    full.load_state_dict(full_state, strict=True)
    full.cuda().eval()

    cached = CachedSpatialFIREFusion("full", dropout=0.1)
    cached_keys = set(cached.state_dict())
    fusion_state = {key: value for key, value in full_state.items() if key in cached_keys}
    missing_fusion_keys = sorted(cached_keys - set(fusion_state))
    unexpected_fusion_keys = sorted(set(fusion_state) - cached_keys)
    cached.load_state_dict(fusion_state, strict=True)
    cached.cuda().eval()

    captured = {}
    original_spatial_extract = full.spatial_branch.extract_features
    original_fire_extract = full.fire_branch.extract_features

    def capture_spatial(*args, **kwargs):
        output = original_spatial_extract(*args, **kwargs)
        captured["spatial"] = output
        return output

    def capture_fire(*args, **kwargs):
        output = original_fire_extract(*args, **kwargs)
        captured["fire"] = output
        return output

    full.spatial_branch.extract_features = capture_spatial
    full.fire_branch.extract_features = capture_fire
    with torch.inference_mode():
        full_output = full(images, return_aux=True)
        spatial = captured["spatial"]
        fire = captured["fire"]
        exact_features = (
            spatial["feature_map"], spatial["global_feature"], spatial["logits"],
            fire["feature_map"], fire["global_feature"], fire["logits"].view(-1, 1), fire["anomaly_prior"],
        )
        cached_exact = cached(*exact_features, return_aux=True)
        quantized_features = tuple(value.half().float() for value in exact_features)
        cached_quantized = cached(*quantized_features, return_aux=True)
    full.spatial_branch.extract_features = original_spatial_extract
    full.fire_branch.extract_features = original_fire_extract

    compared_outputs = ["logits", "router_weights", "spatial_proj", "fire_proj"]
    exact_errors = {key: tensor_error(full_output[key], cached_exact[key]) for key in compared_outputs}
    quantized_errors = {key: tensor_error(full_output[key], cached_quantized[key]) for key in compared_outputs}
    full_probability = torch.softmax(full_output["logits"].float(), dim=1)[:, 1]
    exact_probability = torch.softmax(cached_exact["logits"].float(), dim=1)[:, 1]
    quantized_probability = torch.softmax(cached_quantized["logits"].float(), dim=1)[:, 1]

    report = {
        "status": "complete",
        "purpose": "full-model versus cached-fusion equivalence check",
        "samples": len(labels),
        "labels": labels.tolist(),
        "domains": list(domains),
        "full_checkpoint": str(FULL_CKPT),
        "full_checkpoint_sha256": sha256(FULL_CKPT),
        "full_checkpoint_epoch": int(full_payload.get("epoch", -1)),
        "full_checkpoint_best_auc": float(full_payload.get("best_auc", float("nan"))),
        "fusion_state": {
            "loaded_keys": len(fusion_state),
            "missing_keys": missing_fusion_keys,
            "unexpected_keys": unexpected_fusion_keys,
        },
        "branch_state_drift_from_initial_checkpoints": branch_drift,
        "exact_float32_feature_equivalence": exact_errors,
        "float16_cache_quantization_error": quantized_errors,
        "fake_probabilities": {
            "full_model": full_probability.cpu().tolist(),
            "cached_exact_float32": exact_probability.cpu().tolist(),
            "cached_after_float16_storage": quantized_probability.cpu().tolist(),
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
