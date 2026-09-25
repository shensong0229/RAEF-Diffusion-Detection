import csv
from pathlib import Path
import matplotlib.pyplot as plt

IN_CSV = Path(r"F:\FakeImageDetect\paper_assets\table1_final_draft\table1_main_10domains_percent.csv")
OUT_DIR = Path(r"F:\FakeImageDetect\paper_assets\fig3_performance_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)

domains = [
    "F-PixArt",
    "F-SD3",
    "JuggXL",
    "Lumina",
    "Flux",
    "PixArt-A",
    "SDXL",
    "SDXL-L",
    "Kolors",
    "SSD",
]

# DIRE 当前是 ACC，不和 AUC 画在同一张图里
exclude_methods = {"DIRE (ACC)"}

rows = []
with open(IN_CSV, "r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        method = r["Method"].strip()
        if method in exclude_methods:
            continue
        values = []
        ok = True
        for d in domains:
            try:
                values.append(float(r[d]))
            except Exception:
                ok = False
                break
        if ok:
            rows.append((method, values))

plt.figure(figsize=(10.5, 4.8))

for method, values in rows:
    if method.lower() == "ours":
        plt.plot(domains, values, marker="o", linewidth=3.0, markersize=6, label=method)
    else:
        plt.plot(domains, values, marker="o", linewidth=1.8, markersize=5, label=method)

plt.ylabel("AUC (%)", fontsize=12)
plt.xlabel("Representative unseen generator domains", fontsize=12)
plt.ylim(20, 100)
plt.xticks(rotation=35, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=10)
plt.tight_layout()

plt.savefig(OUT_DIR / "fig3_unseen_domain_auc_comparison.png", dpi=300, bbox_inches="tight")
plt.savefig(OUT_DIR / "fig3_unseen_domain_auc_comparison.pdf", bbox_inches="tight")

print("Saved:")
print(OUT_DIR / "fig3_unseen_domain_auc_comparison.png")
print(OUT_DIR / "fig3_unseen_domain_auc_comparison.pdf")
