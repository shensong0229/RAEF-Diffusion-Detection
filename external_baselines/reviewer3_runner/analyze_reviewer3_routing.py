"""Analyze per-image routing, uncertainty, and evaluation uncertainty.

Run only after reviewer3_routing_eval.py has completed all ten domains.
The analysis uses a fixed 0.5 classification threshold and paired, stratified
within-domain/class bootstrap resampling. EER is a diagnostic, not a tuned
deployment threshold. Both inputs must come from the same checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score, roc_curve

from reviewer3_test_time_ablation import DOMAINS, PROJECT, write_json


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def eer(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    fnr = 1.0 - tpr
    cross = np.where(np.diff(np.sign(fpr - fnr)) != 0)[0]
    if not len(cross):
        return float(fpr[np.argmin(np.abs(fpr - fnr))])
    i = int(cross[0])
    x0, x1 = fpr[i] - fnr[i], fpr[i + 1] - fnr[i + 1]
    alpha = 0.0 if x0 == x1 else -x0 / (x1 - x0)
    return float(fpr[i] + alpha * (fpr[i + 1] - fpr[i]))


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    predicted = (p >= 0.5).astype(np.int64)
    return {
        "auc": float(roc_auc_score(y, p)),
        "ap": float(average_precision_score(y, p)),
        "acc": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "eer": eer(y, p),
    }


def mean_and_interval(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {"n": 0, "mean": None, "median": None}
    return {"n": int(len(values)), "mean": float(np.mean(values)), "median": float(np.median(values))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing-dir", type=Path, default=PROJECT / "results" / "reviewer3_routing_evaluation")
    parser.add_argument("--ablation-dir", type=Path, default=PROJECT / "results" / "reviewer3_test_time_ablation")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    routing_provenance = json.loads((args.routing_dir / "provenance.json").read_text(encoding="utf-8"))
    ablation_provenance = json.loads((args.ablation_dir / "provenance.json").read_text(encoding="utf-8"))
    for key in ("checkpoint_sha256", "checkpoint_protocol", "csv_dir", "data_root", "domains"):
        if routing_provenance.get(key) != ablation_provenance.get(key):
            raise ValueError(f"Routing and ablation provenance disagree on {key}")
    progress = json.loads((args.routing_dir / "progress.json").read_text(encoding="utf-8"))
    if progress.get("status") != "complete" or progress.get("domains_complete") != 10:
        raise RuntimeError("Ten-domain per-image extraction is not complete")

    records = []
    per_domain = []
    for domain in DOMAINS:
        routing = read_csv(args.routing_dir / f"{domain}_predictions.csv")
        ablation = read_csv(args.ablation_dir / f"{domain}_predictions.csv")
        if len(routing) != 2000 or len(ablation) != 2000:
            raise ValueError(f"Expected 2,000 matching rows in {domain}")
        for r, a in zip(routing, ablation):
            if (r["domain"], r["path"], r["label"]) != (a["domain"], a["path"], a["label"]):
                raise ValueError(f"Routing/ablation alignment failed in {domain}: {r['path']}")
            records.append({
                "domain": domain, "y": int(r["label"]),
                "full": float(r["prob_full"]),
                "equal": float(a["prob_no_reliability_routing"]),
                "previous_full": float(a["prob_full"]),
                "spatial": float(r["prob_spatial"]),
                "fire": float(r["prob_fire"]),
                "ws": float(r["weight_spatial"]),
                "wf": float(r["weight_fire"]),
                "hs": float(r["entropy_spatial"]),
                "hf": float(r["entropy_fire"]),
            })
        subset = records[-2000:]
        y = np.asarray([v["y"] for v in subset])
        p = np.asarray([v["full"] for v in subset])
        per_domain.append({"domain": domain, "n": 2000, **metrics(y, p)})

    y = np.asarray([r["y"] for r in records], dtype=np.int64)
    full = np.asarray([r["full"] for r in records], dtype=np.float64)
    equal = np.asarray([r["equal"] for r in records], dtype=np.float64)
    old_full = np.asarray([r["previous_full"] for r in records], dtype=np.float64)
    ps = np.asarray([r["spatial"] for r in records], dtype=np.float64)
    pf = np.asarray([r["fire"] for r in records], dtype=np.float64)
    ws = np.asarray([r["ws"] for r in records], dtype=np.float64)
    wf = np.asarray([r["wf"] for r in records], dtype=np.float64)
    hs = np.asarray([r["hs"] for r in records], dtype=np.float64)
    hf = np.asarray([r["hf"] for r in records], dtype=np.float64)
    if not np.isfinite(np.stack([full, equal, ps, pf, ws, wf, hs, hf])).all():
        raise ValueError("Nonfinite values in extracted evidence")
    if np.max(np.abs(ws + wf - 1)) > 1e-4:
        raise ValueError("Router weights do not sum to 1")

    cs = (ps >= 0.5) == y
    cf = (pf >= 0.5) == y
    masks = {
        "both_correct": cs & cf,
        "only_spatial_correct": cs & ~cf,
        "only_fire_correct": ~cs & cf,
        "both_wrong": ~cs & ~cf,
    }
    groups = {}
    for name, mask in masks.items():
        groups[name] = {
            "weight_spatial": mean_and_interval(ws[mask]),
            "full_accuracy": float(np.mean((full[mask] >= 0.5) == y[mask])) if mask.any() else None,
            "equal_accuracy": float(np.mean((equal[mask] >= 0.5) == y[mask])) if mask.any() else None,
        }
    one_correct = cs ^ cf
    correct_weight = np.where(cs[one_correct], ws[one_correct], wf[one_correct])
    groups["exactly_one_correct"] = {
        "weight_on_correct_branch": mean_and_interval(correct_weight),
        "correct_branch_given_more_weight_rate": float(np.mean(correct_weight > 0.5)) if len(correct_weight) else None,
        "tie_rate": float(np.mean(correct_weight == 0.5)) if len(correct_weight) else None,
    }
    confidence_s = np.maximum(ps, 1 - ps)
    confidence_f = np.maximum(pf, 1 - pf)
    correlations = {
        "weight_spatial_vs_spatial_minus_fire_confidence": float(spearmanr(ws, confidence_s - confidence_f).statistic),
        "weight_spatial_vs_fire_minus_spatial_entropy": float(spearmanr(ws, hf - hs).statistic),
        "weight_spatial_vs_spatial_correct_minus_fire_correct": float(spearmanr(ws, cs.astype(int) - cf.astype(int)).statistic),
    }
    # Mean branch entropy is a prediction-based difficulty proxy, not a
    # ground-truth difficulty label. Quartiles are descriptive only.
    difficulty = (hs + hf) / 2
    edges = np.quantile(difficulty, [0, 0.25, 0.5, 0.75, 1])
    quartiles = []
    for i in range(4):
        mask = (difficulty >= edges[i]) & ((difficulty <= edges[i + 1]) if i == 3 else (difficulty < edges[i + 1]))
        quartiles.append({
            "quartile": i + 1, "n": int(mask.sum()),
            "difficulty_range": [float(edges[i]), float(edges[i + 1])],
            "full_accuracy": float(np.mean((full[mask] >= 0.5) == y[mask])) if mask.any() else None,
            "equal_accuracy": float(np.mean((equal[mask] >= 0.5) == y[mask])) if mask.any() else None,
        })

    # Resample each domain and class independently, preserving the exact
    # 1,000-real/1,000-fake balance. Use the same sampled rows for full and
    # equal routing to obtain a paired difference interval.
    rng = np.random.default_rng(args.seed)
    strata = []
    for domain in DOMAINS:
        for label in (0, 1):
            idx = np.asarray([i for i, row in enumerate(records) if row["domain"] == domain and row["y"] == label])
            if len(idx) != 1000:
                raise ValueError(f"Expected 1,000 class-{label} images in {domain}")
            strata.append(idx)
    names = ("auc", "ap", "acc", "f1", "eer")
    point_full = {name: float(np.mean([row[name] for row in per_domain])) for name in names}
    point_equal = {}
    for domain_index, domain in enumerate(DOMAINS):
        idx = np.arange(domain_index * 2000, (domain_index + 1) * 2000)
        point_equal[domain] = metrics(y[idx], equal[idx])
    point_equal_macro = {name: float(np.mean([point_equal[d][name] for d in DOMAINS])) for name in names}
    samples_full = {name: [] for name in names}
    samples_equal = {name: [] for name in names}
    for rep in range(args.bootstrap):
        sampled = [rng.choice(index, size=len(index), replace=True) for index in strata]
        for name, dest, scores in (("full", samples_full, full), ("equal", samples_equal, equal)):
            per_domain_sample = []
            for i in range(10):
                index = np.concatenate((sampled[2 * i], sampled[2 * i + 1]))
                per_domain_sample.append(metrics(y[index], scores[index]))
            for key in names:
                dest[key].append(float(np.mean([m[key] for m in per_domain_sample])))
        if (rep + 1) % 100 == 0:
            print(f"[BOOTSTRAP] {rep + 1}/{args.bootstrap}", flush=True)
    macro = {}
    for key in names:
        f = np.asarray(samples_full[key])
        e = np.asarray(samples_equal[key])
        delta = f - e
        macro[key] = {
            "full": point_full[key], "full_ci95": np.quantile(f, [0.025, 0.975]).tolist(),
            "equal": point_equal_macro[key], "equal_ci95": np.quantile(e, [0.025, 0.975]).tolist(),
            "full_minus_equal": point_full[key] - point_equal_macro[key],
            "difference_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        }

    summary = {
        "analysis": "Inference-only routing analysis; 10 fixed unseen domains, 20,000 images, no retraining",
        "checkpoint_protocol": routing_provenance["checkpoint_protocol"],
        "checkpoint_sha256": routing_provenance["checkpoint_sha256"],
        "threshold_protocol": "Fixed 0.5 fake-probability threshold for ACC/F1 and branch correctness; never tuned on target test data",
        "eer_note": "EER is threshold-free diagnostic from target-domain ROC, not a selected deployment threshold",
        "confidence_interval": f"95% percentile paired bootstrap, {args.bootstrap} replicates, seed {args.seed}, stratified within each domain/class",
        "full_probability_repeat_max_abs_difference": float(np.max(np.abs(full - old_full))),
        "full_probability_repeat_prediction_disagreement": int(np.sum((full >= 0.5) != (old_full >= 0.5))),
        "macro_metrics": macro,
        "router_correctness_groups": groups,
        "router_correlations": correlations,
        "difficulty_proxy": "Mean binary entropy of the two branch posterior probabilities",
        "difficulty_quartiles": quartiles,
    }
    write_json(args.routing_dir / "routing_analysis_summary.json", summary)
    with (args.routing_dir / "routing_metrics_by_domain.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("domain", "n", *names))
        writer.writeheader()
        writer.writerows(per_domain)
    print("[COMPLETE] routing and uncertainty analysis", flush=True)


if __name__ == "__main__":
    main()
