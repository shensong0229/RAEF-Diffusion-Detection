"""Paper-based GFRE and single-class-attribution reimplementations.

Both papers describe frozen CLIP ViT-L/14 features, an encoder-decoder trained
with an L1 reconstruction objective on a single class, and an MLP classifier on
absolute feature residuals.  The papers do not release code or fully specify
layer widths/epoch counts, so those choices are recorded with the results.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TEST_NAMES = [
    "Flash_PixArt", "Flash_SD3", "JuggernautXL", "Lumina", "Flux_1",
    "PixArt_Alpha", "SDXL", "SDXL_Lightning", "Kolors", "SSD_1B",
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FeatureAutoencoder(nn.Module):
    def __init__(self, dim: int = 768, bottleneck: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 384), nn.GELU(),
                                     nn.Linear(384, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 384), nn.GELU(),
                                     nn.Linear(384, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class ResidualClassifier(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(256, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def loader(x: torch.Tensor, y: torch.Tensor | None, batch: int, shuffle: bool) -> DataLoader:
    if y is None:
        dataset = TensorDataset(x)
    else:
        dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=True, generator=generator)


def train_ae_epoch(ae, x, optimizer, batch, device) -> float:
    ae.train()
    losses = []
    for (xb,) in loader(x, None, batch, True):
        xb = xb.to(device, non_blocking=True).float()
        loss = torch.mean(torch.abs(xb - ae(xb)))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def ae_loss(ae, x, batch, device) -> float:
    ae.eval()
    values = []
    for (xb,) in loader(x, None, batch, False):
        xb = xb.to(device, non_blocking=True).float()
        values.append(float(torch.mean(torch.abs(xb - ae(xb))).cpu()))
    return float(np.mean(values))


def train_classifier_epoch(ae, classifier, x, y, optimizer, batch, device) -> float:
    ae.eval()
    classifier.train()
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for xb, yb in loader(x, y, batch, True):
        xb = xb.to(device, non_blocking=True).float()
        yb = yb.to(device, non_blocking=True).float()
        with torch.no_grad():
            residual = torch.abs(xb - ae(xb))
        logits = classifier(residual)
        loss = criterion(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.inference_mode()
def predict(ae, classifier, x, batch, device) -> np.ndarray:
    ae.eval()
    classifier.eval()
    scores = []
    for (xb,) in loader(x, None, batch, False):
        xb = xb.to(device, non_blocking=True).float()
        residual = torch.abs(xb - ae(xb))
        scores.append(torch.sigmoid(classifier(residual)).cpu())
    return torch.cat(scores).numpy()


def metrics(y: np.ndarray, score: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "accuracy": float(accuracy_score(y, score >= 0.5)),
    }


def fit_method(
    method: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    output: Path,
    device: torch.device,
    train_protocol: str,
) -> tuple[FeatureAutoencoder, ResidualClassifier]:
    output.mkdir(parents=True, exist_ok=True)
    progress = output / "progress.json"
    seed_everything(42)
    ae = FeatureAutoencoder().to(device)
    classifier = ResidualClassifier().to(device)
    fake_train = train_x[train_y == 1]
    fake_val = val_x[val_y == 1]

    if method == "gfre":
        ae_lr, clf_lr, batch, max_epochs = 1e-4, 1e-4, 512, 30
        ae_opt = torch.optim.Adam(ae.parameters(), lr=ae_lr)
        best_ae = copy.deepcopy(ae.state_dict())
        best_loss, bad = float("inf"), 0
        history = []
        for epoch in range(1, max_epochs + 1):
            train_loss = train_ae_epoch(ae, fake_train, ae_opt, batch, device)
            val_loss = ae_loss(ae, fake_val, batch, device)
            history.append({"stage": "autoencoder", "epoch": epoch,
                            "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_loss - 1e-6:
                best_loss, best_ae, bad = val_loss, copy.deepcopy(ae.state_dict()), 0
            else:
                bad += 1
            atomic_json(progress, {"status": "training", "method": "GFRE reimplementation",
                                   "stage": "autoencoder", "epoch": epoch,
                                   "max_epochs": max_epochs, "val_l1": val_loss})
            if bad >= 5:
                break
        ae.load_state_dict(best_ae)

        clf_opt = torch.optim.Adam(classifier.parameters(), lr=clf_lr)
        best_auc, best_clf, bad = -1.0, copy.deepcopy(classifier.state_dict()), 0
        val_np = val_y.numpy()
        for epoch in range(1, max_epochs + 1):
            train_loss = train_classifier_epoch(ae, classifier, train_x, train_y,
                                                  clf_opt, batch, device)
            val_score = predict(ae, classifier, val_x, batch, device)
            val_auc = float(roc_auc_score(val_np, val_score))
            history.append({"stage": "classifier", "epoch": epoch,
                            "train_loss": train_loss, "val_auc": val_auc})
            if val_auc > best_auc + 1e-6:
                best_auc, best_clf, bad = val_auc, copy.deepcopy(classifier.state_dict()), 0
            else:
                bad += 1
            atomic_json(progress, {"status": "training", "method": "GFRE reimplementation",
                                   "stage": "classifier", "epoch": epoch,
                                   "max_epochs": max_epochs, "val_auc": val_auc})
            if bad >= 5:
                break
        classifier.load_state_dict(best_clf)
    else:
        # Paper setting: both modules use Adam, lr=2e-4, batch size 256.
        ae_lr = clf_lr = 2e-4
        batch, max_epochs = 256, 30
        ae_opt = torch.optim.Adam(ae.parameters(), lr=ae_lr)
        clf_opt = torch.optim.Adam(classifier.parameters(), lr=clf_lr)
        best_auc, best_pair, bad = -1.0, None, 0
        history = []
        val_np = val_y.numpy()
        for epoch in range(1, max_epochs + 1):
            reconstruction_loss = train_ae_epoch(ae, fake_train, ae_opt, batch, device)
            classification_loss = train_classifier_epoch(ae, classifier, train_x, train_y,
                                                           clf_opt, batch, device)
            val_score = predict(ae, classifier, val_x, batch, device)
            val_auc = float(roc_auc_score(val_np, val_score))
            history.append({"stage": "alternating", "epoch": epoch,
                            "reconstruction_loss": reconstruction_loss,
                            "classification_loss": classification_loss,
                            "val_auc": val_auc})
            if val_auc > best_auc + 1e-6:
                best_auc = val_auc
                best_pair = (copy.deepcopy(ae.state_dict()),
                             copy.deepcopy(classifier.state_dict()))
                bad = 0
            else:
                bad += 1
            atomic_json(progress, {"status": "training",
                                   "method": "Single-class attribution reimplementation",
                                   "stage": "alternating", "epoch": epoch,
                                   "max_epochs": max_epochs, "val_auc": val_auc})
            if bad >= 5:
                break
        if best_pair is not None:
            ae.load_state_dict(best_pair[0])
            classifier.load_state_dict(best_pair[1])

    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    torch.save({"autoencoder": ae.state_dict(), "classifier": classifier.state_dict()},
               output / "best_model.pth")
    details = {
        "implementation": "paper-based reimplementation; no official code/checkpoint available",
        "backbone": "frozen OpenAI CLIP ViT-L/14, L2-normalized 768-D image features",
        "attribution_source": "SDV5 generated images only",
        "autoencoder": "768-384-128-384-768, GELU, L1 reconstruction loss",
        "classifier": "768-256-1 MLP, GELU, dropout 0.1, BCEWithLogitsLoss",
        "seed": 42,
        "train_protocol": train_protocol,
        "optimizer": "Adam",
        "learning_rates": {"autoencoder": ae_lr, "classifier": clf_lr},
        "batch_size": batch,
        "max_epochs": max_epochs,
        "early_stopping_patience": 5,
    }
    atomic_json(output / "reimplementation_details.json", details)
    return ae, classifier


def evaluate_method(method_name, ae, classifier, cache, output, device, batch) -> dict:
    rows = []
    for index, name in enumerate(TEST_NAMES, start=1):
        payload = torch.load(cache / f"test_{name}.pt", map_location="cpu", weights_only=False)
        x = payload["features"]
        y = payload["labels"].long().numpy()
        score = predict(ae, classifier, x, batch, device)
        result = {"domain": name, "images": len(y), **metrics(y, score)}
        rows.append(result)
        pd.DataFrame({"path": payload["paths"], "label": y,
                      "domain": payload["domains"], "fake_score": score}).to_csv(
            output / f"{name}_predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(output / "metrics_by_domain.csv", index=False)
        atomic_json(output / "progress.json", {"status": "evaluating", "method": method_name,
                                                "domains_complete": index,
                                                "domains_total": len(TEST_NAMES),
                                                "current_domain": name, **result})
        print(f"[{method_name}] {name}: AUC={result['auc']:.4f}", flush=True)
    frame = pd.DataFrame(rows)
    macro = {"domain": "Macro average", "images": int(frame["images"].sum()),
             "auc": float(frame["auc"].mean()), "ap": float(frame["ap"].mean()),
             "accuracy": float(frame["accuracy"].mean())}
    frame = pd.concat([frame, pd.DataFrame([macro])], ignore_index=True)
    frame.to_csv(output / "metrics_by_domain.csv", index=False)
    summary = {"status": "complete", "method": method_name,
               "protocol": "fixed ten domains, 1000 real + 1000 fake per domain",
               "macro_auc": macro["auc"], "macro_ap": macro["ap"],
               "macro_accuracy": macro["accuracy"]}
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "progress.json", {**summary, "domains_complete": 10,
                                            "domains_total": 10})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--method", choices=["gfre", "attribution", "both"], default="both")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--test-cache-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--train-protocol", required=True,
                        help="Exact source split used for this run; saved in the result summary.")
    parser.add_argument("--expected-train", type=int, default=None)
    parser.add_argument("--expected-val", type=int, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result_root = (args.output_root or root / "results" / "reviewer3_baselines").resolve()
    cache = (args.cache_dir or result_root / "clip_l14_feature_cache").resolve()
    test_cache = (args.test_cache_dir or cache).resolve()
    train = torch.load(cache / "sdv5_train.pt", map_location="cpu", weights_only=False)
    val = torch.load(cache / "sdv5_val.pt", map_location="cpu", weights_only=False)
    train_x, train_y = train["features"], train["labels"].long()
    val_x, val_y = val["features"], val["labels"].long()
    if args.expected_train is not None and len(train_y) != args.expected_train:
        raise ValueError(f"Expected {args.expected_train} training images, found {len(train_y)}")
    if args.expected_val is not None and len(val_y) != args.expected_val:
        raise ValueError(f"Expected {args.expected_val} validation images, found {len(val_y)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected = ["gfre", "attribution"] if args.method == "both" else [args.method]
    for method in selected:
        if method == "gfre":
            output = result_root / "gfre_reimplementation"
            display = "GFRE reimplementation"
            eval_batch = 1024
        else:
            output = result_root / "single_class_attribution_reimplementation"
            display = "Single-class attribution reimplementation"
            eval_batch = 1024
        started = time.time()
        ae, classifier = fit_method(method, train_x, train_y, val_x, val_y, output,
                                    device, args.train_protocol)
        summary = evaluate_method(display, ae, classifier, test_cache, output,
                                  device, eval_batch)
        summary["elapsed_seconds"] = round(time.time() - started, 1)
        summary["training_protocol"] = args.train_protocol
        summary["training_images"] = len(train_y)
        summary["validation_images"] = len(val_y)
        summary["training_feature_cache"] = str(cache)
        summary["test_feature_cache"] = str(test_cache)
        atomic_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
