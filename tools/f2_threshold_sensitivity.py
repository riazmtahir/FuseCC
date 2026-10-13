"""Threshold sensitivity analysis from saved OOF probabilities.

For every available OOF source, sweep the decision threshold over [0.01, 0.99],
write the full F1/P/R curve (CSV + PNG) and report the F1 plateau width — the
threshold range where F1 stays within 1% of its maximum. A wide plateau means
the method is insensitive to the threshold choice.

Outputs under results/f2_threshold/.
Run:  python tools/f2_threshold_sensitivity.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

OUT = "results/f2_threshold"
os.makedirs(OUT, exist_ok=True)


def load_sources():
    src = {}
    d = np.load("results/gcj_148_ft/classification_enhanced/oof.npz")
    src["gcj_148ft_meta"] = (d["meta_oof"], d["y"].astype(int))
    d = np.load("results/gcj_148/classification_enhanced/oof.npz")
    src["gcj_148zs_meta"] = (d["meta_oof"], d["y"].astype(int))
    df = pd.read_csv("results/real_fusecc/classification/oof_predictions.csv")
    src["bcb_16feat_stack"] = (df["oof_prob"].values, df["label"].astype(int).values)
    return src


grid = np.linspace(0.01, 0.99, 99)
summary = {}
for name, (p, y) in load_sources().items():
    rows = [{"threshold": float(t),
             "f1": f1_score(y, (p >= t).astype(int)),
             "precision": precision_score(y, (p >= t).astype(int)),
             "recall": recall_score(y, (p >= t).astype(int))} for t in grid]
    curve = pd.DataFrame(rows)
    curve.to_csv(f"{OUT}/{name}_curve.csv", index=False)
    best = curve.loc[curve.f1.idxmax()]
    plateau = curve[curve.f1 >= 0.99 * best.f1]
    summary[name] = {
        "best_threshold": float(best.threshold), "best_f1": float(best.f1),
        "f1_at_0.5": float(curve.loc[(curve.threshold - 0.5).abs().idxmin(), "f1"]),
        "plateau_99pct": [float(plateau.threshold.min()), float(plateau.threshold.max())],
        "plateau_width": float(plateau.threshold.max() - plateau.threshold.min()),
    }
    plt.figure(figsize=(6, 4))
    for col in ("f1", "precision", "recall"):
        plt.plot(curve.threshold, curve[col], label=col)
    plt.axvspan(plateau.threshold.min(), plateau.threshold.max(), alpha=0.15,
                label="F1 ≥ 99% of max")
    plt.xlabel("decision threshold")
    plt.ylabel("score")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/{name}_curve.png", dpi=200)
    plt.close("all")
    print(name, json.dumps(summary[name]), flush=True)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("saved", f"{OUT}/summary.json")
