"""Summarize ten-domain per-image predictions with a fixed 0.50 threshold.

This script does not train or tune a detector. The confidence interval is a
descriptive Student-t interval across domain-level metrics, not an image-level
bootstrap interval. Keep the prediction CSVs and checkpoint provenance with
the reported result.
"""

from __future__ import annotations

import argparse
import csv
import json
from math import sqrt
from pathlib import Path
from statistics import mean, stdev

import numpy as np
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


def equal_error_rate(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(labels, scores)
    false_negative = 1.0 - true_positive
    difference = false_positive - false_negative
    crossings = np.flatnonzero(np.diff(np.sign(difference)) != 0)
    if len(crossings) == 0:
        return float(false_positive[np.argmin(np.abs(difference))])
    index = int(crossings[0])
    denominator = difference[index + 1] - difference[index]
    fraction = 0.0 if denominator == 0 else -difference[index] / denominator
    return float(false_positive[index] + fraction *
                 (false_positive[index + 1] - false_positive[index]))


def summarize_file(path: Path, label_column: str, score_column: str) -> dict:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {label_column, score_column}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing label or score column in {path}")
        rows = list(reader)
    labels = np.asarray([int(row[label_column]) for row in rows], dtype=np.int64)
    scores = np.asarray([float(row[score_column]) for row in rows], dtype=np.float64)
    if len(labels) == 0 or set(labels.tolist()) != {0, 1}:
        raise ValueError(f"Both classes are required in {path}")
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError(f"Scores must be finite fake-class probabilities in [0,1]: {path}")
    predicted = (scores >= 0.50).astype(np.int64)
    return {
        "domain": path.stem.removesuffix("_predictions"),
        "n": int(len(labels)),
        "real": int(np.sum(labels == 0)),
        "fake": int(np.sum(labels == 1)),
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "acc": float(accuracy_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "eer": equal_error_rate(labels, scores),
    }


def t_interval(values: list[float]) -> list[float]:
    if len(values) < 2:
        raise ValueError("At least two domains are needed for a confidence interval")
    center = mean(values)
    half_width = float(t.ppf(0.975, len(values) - 1)) * stdev(values) / sqrt(len(values))
    return [center - half_width, center + half_width]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--pattern", default="*_predictions.csv")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--score-column", default="prob_full")
    parser.add_argument("--expected-domains", type=int, default=10)
    parser.add_argument("--expected-per-domain", type=int, default=2000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    files = sorted(args.prediction_dir.glob(args.pattern))
    if len(files) != args.expected_domains:
        raise ValueError(f"Expected {args.expected_domains} domain files, found {len(files)}")
    domains = [summarize_file(path, args.label_column, args.score_column) for path in files]
    if len({row["domain"] for row in domains}) != len(domains):
        raise ValueError("Duplicate domain names")
    for row in domains:
        if row["n"] != args.expected_per_domain:
            raise ValueError(f"{row['domain']}: expected {args.expected_per_domain} images, found {row['n']}")
        if row["real"] != row["fake"]:
            raise ValueError(f"{row['domain']}: expected balanced real/fake classes")

    metrics = ("auc", "ap", "acc", "f1", "eer")
    report = {
        "threshold_protocol": "Fixed fake-class probability >= 0.50 for ACC and F1; AUC, AP, and EER use continuous scores",
        "interval_method": "Descriptive two-sided 95% Student-t interval across domain-level metrics",
        "domains": domains,
        "macro": {
            name: {"mean": mean([row[name] for row in domains]),
                   "ci95": t_interval([row[name] for row in domains])}
            for name in metrics
        },
    }
    payload = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
