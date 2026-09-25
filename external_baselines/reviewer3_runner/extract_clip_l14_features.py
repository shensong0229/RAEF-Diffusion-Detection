"""Cache frozen OpenAI CLIP ViT-L/14 features for GFRE and attribution baselines."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


DOMAIN_FILES = [
    "Flash_PixArt.csv", "Flash_SD3.csv", "JuggernautXL.csv", "Lumina.csv",
    "Flux_1.csv", "PixArt_Alpha.csv", "SDXL.csv", "SDXL_Lightning.csv",
    "Kolors.csv", "SSD_1B.csv",
]


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    tmp.write_text(content, encoding="utf-8")
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    path.write_text(content, encoding="utf-8")
    tmp.unlink(missing_ok=True)


class ImageFrameDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, preprocess) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.image_root / str(row["path"])
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, int(row["label"]), index


def valid_cache(path: Path, expected: int) -> bool:
    if not path.exists():
        return False
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload["features"].shape[0]) == expected
    except Exception:
        return False


def extract_one(
    name: str,
    frame: pd.DataFrame,
    image_root: Path,
    cache_path: Path,
    model,
    preprocess,
    device: torch.device,
    batch_size: int,
    progress_path: Path,
    state: dict,
) -> None:
    if valid_cache(cache_path, len(frame)):
        state["sets_complete"] += 1
        state["images_complete"] += len(frame)
        state["current_set"] = name
        state["status"] = "running"
        atomic_json(progress_path, state)
        print(f"[resume] {name}: {len(frame)} cached", flush=True)
        return

    dataset = ImageFrameDataset(frame, image_root, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                        pin_memory=True, drop_last=False)
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.time()
    for batch_index, (images, target, _indices) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode():
            encoded = model.encode_image(images)
            encoded = F.normalize(encoded.float(), dim=1)
        features.append(encoded.cpu().half())
        labels.append(target.to(torch.int8).cpu())
        done = min(batch_index * batch_size, len(dataset))
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == len(loader):
            elapsed = time.time() - started
            rate = done / elapsed if elapsed > 0 else 0.0
            state.update(
                {
                    "status": "running",
                    "current_set": name,
                    "current_set_images_complete": done,
                    "current_set_images_total": len(dataset),
                    "current_set_images_per_second": round(rate, 2),
                    "images_complete": state["base_images_complete"] + done,
                }
            )
            atomic_json(progress_path, state)
            print(f"[{name}] {done}/{len(dataset)} | {rate:.2f} images/s", flush=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": torch.cat(features, dim=0),
        "labels": torch.cat(labels, dim=0),
        "paths": frame["path"].astype(str).tolist(),
        "domains": frame["domain"].astype(str).tolist(),
        "split": frame["split"].astype(str).tolist(),
        "feature_backbone": "OpenAI CLIP ViT-L/14",
        "feature_normalization": "L2",
    }
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, cache_path)
    state["sets_complete"] += 1
    state["base_images_complete"] += len(dataset)
    state["images_complete"] = state["base_images_complete"]
    atomic_json(progress_path, state)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-test", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    vendor = root / "external_baselines" / "spai" / "vendor"
    sys.path.insert(0, str(vendor))
    import clip

    output = (args.output_dir or
              root / "results" / "reviewer3_baselines" / "clip_l14_feature_cache").resolve()
    output.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    model_path = Path.home() / ".cache" / "clip" / "ViT-L-14.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"CLIP weight not found: {model_path}")

    train_csv = args.train_index.resolve()
    train_frame = pd.read_csv(train_csv)
    train_frame["domain"] = train_frame.get("domain", "SDV5")
    sources: list[tuple[str, pd.DataFrame, Path, Path]] = []
    for split in ("train", "val"):
        frame = train_frame[train_frame["split"] == split].copy()
        sources.append((f"sdv5_{split}", frame, root / "data", output / f"sdv5_{split}.pt"))

    if not args.skip_test:
        bundle = root / "exports" / "dragon_eval_25domains_unique_real_1k_bundle"
        for filename in DOMAIN_FILES:
            frame = pd.read_csv(bundle / "csvs" / filename)
            sources.append((Path(filename).stem, frame, bundle / "images",
                            output / f"test_{Path(filename).stem}.pt"))

    total_images = sum(len(frame) for _, frame, _, _ in sources)
    state = {
        "status": "loading_model",
        "method": "Shared frozen CLIP ViT-L/14 feature cache",
        "sets_total": len(sources),
        "sets_complete": 0,
        "images_total": total_images,
        "images_complete": 0,
        "base_images_complete": 0,
        "batch_size": args.batch_size,
        "train_index": str(train_csv),
        "skip_test": args.skip_test,
    }
    atomic_json(progress_path, state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(str(model_path), device=device, jit=False)
    model.eval()
    state["status"] = "running"
    atomic_json(progress_path, state)

    for name, frame, image_root, cache_path in sources:
        state["base_images_complete"] = state["images_complete"]
        extract_one(name, frame, image_root, cache_path, model, preprocess, device,
                    args.batch_size, progress_path, state)

    state.update({"status": "complete", "sets_complete": len(sources),
                  "images_complete": total_images})
    atomic_json(progress_path, state)
    print(f"CLIP feature cache complete: {total_images} images", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        output_arg = sys.argv.index("--output-dir") if "--output-dir" in sys.argv else -1
        output_dir = (Path(sys.argv[output_arg + 1]) if output_arg >= 0 else
                      Path(__file__).resolve().parents[2] / "results" /
                      "reviewer3_baselines" / "clip_l14_feature_cache")
        path = output_dir / "progress.json"
        atomic_json(path, {"status": "failed", "error": repr(exc)})
        raise
