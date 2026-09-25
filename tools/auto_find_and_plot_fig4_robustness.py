import csv
import json
import math
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

ROOT = Path(r"F:\FakeImageDetect")
OUT = ROOT / "paper_assets" / "fig4_robustness"
OUT.mkdir(parents=True, exist_ok=True)

REP_DOMAINS = [
    "Flash_PixArt",
    "Flash_SD3",
    "JuggernautXL",
    "Lumina",
    "Flux_1",
    "PixArt_Alpha",
    "SDXL",
    "SDXL_Lightning",
    "Kolors",
    "SSD_1B",
]

SEARCH_ROOTS = [
    ROOT / "results",
    ROOT / "downloads",
    ROOT / "paper_assets",
]

# 如果你想把 UnivFD 的 AP 也画进主图，把这里改成 True。
# 默认 False，因为主图建议用 AUC。
INCLUDE_UNIVFD_AP_IN_MAIN_PLOT = False


def find_dir_by_name(name):
    hits = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob(name):
            if p.is_dir():
                hits.append(p)
    hits = sorted(set(hits), key=lambda x: (len(str(x)), str(x)))
    return hits[0] if hits else None


def try_float(x):
    try:
        if x is None:
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def normalize_auc(v):
    if v is None:
        return None
    return v / 100.0 if v > 1.5 else v


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return []
        return list(reader)


def avg_auc_from_domain_csv(path):
    rows = read_csv_rows(path)
    vals = []
    for r in rows:
        d = r.get("domain") or r.get("Domain") or r.get("dataset") or r.get("TestSet")
        if d is None:
            continue
        d = str(d).strip()
        if d not in REP_DOMAINS:
            continue
        v = r.get("auc") or r.get("AUC") or r.get("auroc") or r.get("AUROC")
        v = normalize_auc(try_float(v))
        if v is not None:
            vals.append(v)
    return mean(vals) if vals else None


def avg_metric_from_domain_txt(root, tag, metric_name):
    # UnivFD style: root/tag/domain/ap.txt 或 acc0.txt
    vals = []
    for d in REP_DOMAINS:
        p = root / tag / d / f"{metric_name}.txt"
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8").strip()
        except Exception:
            text = p.read_text(errors="ignore").strip()
        v = try_float(text.split()[0])
        if v is not None:
            vals.append(normalize_auc(v))
    return mean(vals) if vals else None


def rank_auc(labels, scores):
    pairs = [(float(s), int(y)) for y, s in zip(labels, scores)]
    n_pos = sum(1 for _, y in pairs if y == 1)
    n_neg = sum(1 for _, y in pairs if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    pairs.sort(key=lambda x: x[0])
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    rank_sum_pos = sum(ranks[i] for i, (_, y) in enumerate(pairs) if y == 1)
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def average_precision(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0], reverse=True)
    n_pos = sum(1 for _, y in pairs if int(y) == 1)
    if n_pos == 0:
        return None
    tp = 0
    precisions = []
    for i, (_, y) in enumerate(pairs, start=1):
        if int(y) == 1:
            tp += 1
            precisions.append(tp / i)
    return sum(precisions) / n_pos if precisions else None


def aer_domain_auc_from_distances(tag_dir, domain):
    # AER robust style:
    # tag/domain/nature/distances.csv
    # tag/domain/ai/distances.csv
    labels, scores = [], []
    for cls, label in [("nature", 0), ("ai", 1)]:
        p = tag_dir / domain / cls / "distances.csv"
        if not p.exists():
            return None, None
        rows = read_csv_rows(p)
        for r in rows:
            s = try_float(r.get("distance"))
            if s is None:
                continue
            labels.append(label)
            scores.append(s)
    if len(labels) == 0:
        return None, None
    return rank_auc(labels, scores), average_precision(labels, scores)


def avg_aer_auc(root, tag):
    vals = []
    for d in REP_DOMAINS:
        auc, _ = aer_domain_auc_from_distances(root / tag, d)
        if auc is not None:
            vals.append(auc)
    return mean(vals) if vals else None


def clean_from_table1(method):
    # 尝试从你已经生成的表1里读取 clean Avg
    candidates = [
        ROOT / "paper_assets" / "table1_final_draft" / "table1_main_10domains_percent.csv",
        ROOT / "paper_assets" / "table1_merged_strict" / "table1_main_auc_percent.csv",
        ROOT / "paper_assets" / "table1_merged" / "table1_main_auc_percent.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        rows = read_csv_rows(p)
        for r in rows:
            m = (r.get("Method") or "").strip()
            if m.lower() == method.lower():
                v = r.get("Avg") or r.get("Avg.10") or r.get("avg") or r.get("avg.10")
                v = try_float(v)
                if v is not None:
                    return v / 100.0 if v > 1.5 else v
    return None


def clean_from_source_csv(method):
    if method == "Ours":
        p = ROOT / "results" / "sfire_dragon25_clean.csv"
        return avg_auc_from_domain_csv(p) if p.exists() else None
    if method == "FIRE":
        p = ROOT / "results" / "fire_dragon25_best.csv"
        return avg_auc_from_domain_csv(p) if p.exists() else None
    if method == "AEROBLADE":
        p = ROOT / "results" / "aeroblade" / "dragon25_sd11_lpipsvgg2_resize512" / "aeroblade_all.csv"
        return avg_auc_from_domain_csv(p) if p.exists() else None
    return None


def tag_file_candidates(root, base):
    # 兼容 ours: jpeg_95.csv, fire: fire_jpeg_95.csv
    return [
        root / f"{base}.csv",
        root / f"fire_{base}.csv",
        root / f"ours_{base}.csv",
    ]


def get_auc_from_tag_csv(root, tag):
    for p in tag_file_candidates(root, tag):
        if p.exists():
            return avg_auc_from_domain_csv(p)
    return None


def discover():
    dirs = {
        "Ours": find_dir_by_name("robustness_ours_10domains"),
        "FIRE": find_dir_by_name("robustness_fire_10domains"),
        "UnivFD": find_dir_by_name("robustness_univfd_10domains"),
        "AEROBLADE": find_dir_by_name("robustness_aeroblade_10domains"),
    }
    return dirs


def collect_data():
    dirs = discover()
    report_lines = []
    report_lines.append("Discovered robustness directories:")
    for k, v in dirs.items():
        report_lines.append(f"{k}: {v}")

    records = []

    # Ours / FIRE: AUC CSV
    for method in ["Ours", "FIRE"]:
        root = dirs.get(method)
        if root is None:
            report_lines.append(f"[MISS] {method} robustness directory not found.")
            continue

        clean = get_auc_from_tag_csv(root, "clean")
        if clean is None:
            clean = clean_from_source_csv(method)
        if clean is None:
            clean = clean_from_table1(method)

        if clean is not None:
            records.append([method, "clean", "clean", "AUC", clean * 100])

        for tag in ["jpeg_95", "jpeg_80", "jpeg_60", "jpeg_40",
                    "resize_0p8", "resize_0p6", "resize_0p5",
                    "crop_0p9", "crop_0p8", "crop_0p6",
                    "blur_1", "blur_2", "blur_3", "blur_4"]:
            auc = get_auc_from_tag_csv(root, tag)
            if auc is None:
                report_lines.append(f"[MISS] {method} {tag} AUC not found.")
                continue

            ptype, level = parse_tag(tag)
            records.append([method, ptype, level, "AUC", auc * 100])

    # UnivFD: usually AP/ACC only
    root = dirs.get("UnivFD")
    if root is not None:
        clean_ap = clean_from_table1("UnivFD")
        # 表1 clean 的 UnivFD 是 AUC；但鲁棒性 validate.py 通常只保存 AP/ACC。
        # 这里不把 clean_ap 混进 AP 曲线，只记录 AP 从鲁棒性文件来。
        for tag in ["jpeg_95", "jpeg_80", "jpeg_60", "jpeg_40",
                    "resize_0p8", "resize_0p6", "resize_0p5",
                    "crop_0p9", "crop_0p8", "crop_0p6",
                    "blur_1", "blur_2", "blur_3", "blur_4"]:
            ap = avg_metric_from_domain_txt(root, tag, "ap")
            acc = avg_metric_from_domain_txt(root, tag, "acc0")
            ptype, level = parse_tag(tag)
            if ap is not None:
                records.append(["UnivFD", ptype, level, "AP", ap * 100])
            else:
                report_lines.append(f"[MISS] UnivFD {tag} AP not found.")
            if acc is not None:
                records.append(["UnivFD", ptype, level, "ACC", acc * 100])
    else:
        report_lines.append("[MISS] UnivFD robustness directory not found.")

    # AEROBLADE: compute AUC from distances.csv
    root = dirs.get("AEROBLADE")
    if root is not None:
        clean = clean_from_source_csv("AEROBLADE") or clean_from_table1("AEROBLADE")
        if clean is not None:
            records.append(["AEROBLADE", "clean", "clean", "AUC", clean * 100])

        for tag in ["jpeg_95", "jpeg_80", "jpeg_60", "jpeg_40",
                    "resize_0p8", "resize_0p6", "resize_0p5",
                    "crop_0p9", "crop_0p8", "crop_0p6",
                    "blur_1", "blur_2", "blur_3", "blur_4"]:
            auc = avg_aer_auc(root, tag)
            ptype, level = parse_tag(tag)
            if auc is not None:
                records.append(["AEROBLADE", ptype, level, "AUC", auc * 100])
            else:
                report_lines.append(f"[MISS] AEROBLADE {tag} AUC not found or incomplete.")
    else:
        report_lines.append("[MISS] AEROBLADE robustness directory not found.")

    return records, report_lines


def parse_tag(tag):
    if tag == "clean":
        return "clean", "clean"
    if tag.startswith("jpeg_"):
        return "jpeg", tag.split("_", 1)[1]
    if tag.startswith("resize_"):
        return "resize", tag.split("_", 1)[1].replace("p", ".")
    if tag.startswith("crop_"):
        return "crop", tag.split("_", 1)[1].replace("p", ".")
    if tag.startswith("blur_"):
        return "blur", tag.split("_", 1)[1]
    return "unknown", tag


def write_records(records):
    out_csv = OUT / "robustness_curve_data.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "perturb_type", "level", "metric", "value"])
        writer.writerows(records)
    return out_csv


def plot(records):
    # 主图只画 AUC。UnivFD 如果只有 AP/ACC，会不进入主图。
    auc_records = [r for r in records if r[3].upper() == "AUC"]
    if not auc_records:
        raise RuntimeError("No AUC robustness data found. Cannot plot AUC curves.")

    order = {
        "jpeg": ["clean", "95", "80", "60", "40"],
        "resize": ["clean", "0.8", "0.6", "0.5"],
        "crop": ["clean", "0.9", "0.8", "0.6"],
        "blur": ["clean", "1", "2", "3", "4"],
    }
    titles = {
        "jpeg": "(a) JPEG",
        "resize": "(b) Resize",
        "crop": "(c) Crop",
        "blur": "(d) Blur",
    }

    method_order = ["Ours", "FIRE", "AEROBLADE", "CNNDet", "UnivFD"]
    colors = {
        "Ours": "red",
        "FIRE": "limegreen",
        "AEROBLADE": "purple",
        "CNNDet": "tab:blue",
        "UnivFD": "blue",
    }

    data = {}
    for method, ptype, level, metric, value in auc_records:
        data.setdefault((method, ptype), {})[str(level)] = float(value)

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    axes = axes.flatten()

    for ax, ptype in zip(axes, ["jpeg", "resize", "crop", "blur"]):
        ax.set_facecolor("#f4f4f4")

        for method in method_order:
            vals = []
            has_any = False
            for lv in order[ptype]:
                v = data.get((method, ptype), {}).get(lv)
                if v is None and lv == "clean":
                    v = data.get((method, "clean"), {}).get("clean")
                vals.append(v)
                if v is not None:
                    has_any = True

            if not has_any:
                continue

            xs = list(range(len(order[ptype])))
            if method == "Ours":
                ax.plot(xs, vals, marker="s", linewidth=2.2, markersize=4.5, label=method, color=colors.get(method))
            else:
                ax.plot(xs, vals, marker="s", linewidth=1.5, markersize=4, label=method, color=colors.get(method))

        ax.set_xticks(range(len(order[ptype])))
        ax.set_xticklabels(["w/o" if x == "clean" else x for x in order[ptype]], fontsize=9)
        ax.set_ylim(40, 100)
        ax.grid(True, color="#cccccc", linewidth=0.8)
        ax.set_ylabel("AUC", fontsize=11)
        if ptype == "jpeg":
            ax.set_xlabel("$q$", fontsize=11)
        elif ptype == "resize":
            ax.set_xlabel("$f$", fontsize=11)
        else:
            ax.set_xlabel("$\\sigma$" if ptype == "blur" else "$r$", fontsize=11)
        ax.set_title(titles[ptype], y=-0.32, fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", fontsize=8, frameon=True)

    plt.subplots_adjust(wspace=0.35, hspace=0.45)

    png = OUT / "fig4_robustness_curves_auc.png"
    pdf = OUT / "fig4_robustness_curves_auc.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()

    return png, pdf


records, report = collect_data()
data_csv = write_records(records)
report_path = OUT / "robustness_discovery_report.txt"
report_path.write_text("\n".join(report), encoding="utf-8")

print("Saved data:", data_csv)
print("Saved report:", report_path)

try:
    png, pdf = plot(records)
    print("Saved figure:", png)
    print("Saved figure:", pdf)
except Exception as e:
    print("[PLOT_FAILED]", repr(e))
