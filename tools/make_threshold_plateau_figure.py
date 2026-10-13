"""Rebuild the threshold figure as a plateau-interval chart.

Replaces the three near-horizontal F1-versus-threshold curves, which conveyed "nothing
happens" weakly and hid the plateau widths in the prose, with one horizontal bar per
configuration spanning the theta range where F1 stays within 1% of that configuration's own
maximum (FuseCC_Structural_Revision_Plan.md section 1.8).

Intervals are RECOMPUTED from the per-configuration sweep CSVs in results/f2_threshold/ and
cross-checked against the plateau_width recorded in that directory's summary.json; the script
fails if the two disagree. Nothing is estimated by eye.

Greyscale-safe by construction: bars are a single mid grey, the operating point is a black
marker, and every quantity is also printed as text, so no claim rests on colour.

Run: python tools/make_threshold_plateau_figure.py
Out: results/f2_threshold/fig_threshold_plateau.{pdf,png}  (300 dpi)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC = "results/f2_threshold"
OUT = os.path.join(SRC, "fig_threshold_plateau")

# label, csv key, human label for the figure
CONFIGS = [
    ("gcj_148ft_meta", "Google Code Jam, 148 feat.\n(fine-tuned encoder)"),
    ("gcj_148zs_meta", "Google Code Jam, 148 feat.\n(zero-shot encoder)"),
    ("bcb_16feat_stack", "BigCloneBench stack\n(16-feature configuration)"),
]

summary = json.load(open(os.path.join(SRC, "summary.json")))

rows = []
for key, label in CONFIGS:
    df = pd.read_csv(os.path.join(SRC, f"{key}_curve.csv"))
    f1 = df["f1"].values
    thr = df["threshold"].values
    fmax = f1.max()
    inside = f1 >= 0.99 * fmax          # within 1% of this configuration's own maximum
    lo, hi = thr[inside].min(), thr[inside].max()
    width = hi - lo
    best_thr = float(thr[int(np.argmax(f1))])

    # cross-check against the recorded summary
    rec = summary[key]
    assert abs(width - rec["plateau_width"]) < 1e-6, (
        f"{key}: recomputed width {width:.4f} != recorded {rec['plateau_width']:.4f}")
    assert abs(best_thr - rec["best_threshold"]) < 1e-9, (
        f"{key}: recomputed best threshold {best_thr} != recorded {rec['best_threshold']}")
    rows.append(dict(key=key, label=label, lo=lo, hi=hi, width=width,
                     best=best_thr, fmax=fmax))
    print(f"{key:18} plateau [{lo:.2f}, {hi:.2f}]  width {width:.2f}  "
          f"best thr {best_thr:.2f}  max F1 {fmax:.4f}")

fig, ax = plt.subplots(figsize=(7.0, 2.35))
GREY = "0.62"
y = np.arange(len(rows))[::-1]

for yi, r in zip(y, rows):
    ax.barh(yi, r["hi"] - r["lo"], left=r["lo"], height=0.34,
            color=GREY, edgecolor="black", linewidth=0.8, zorder=2)
    # interval end caps
    for x in (r["lo"], r["hi"]):
        ax.plot([x, x], [yi - 0.22, yi + 0.22], color="black", lw=1.1, zorder=3)
    # selected operating point
    ax.plot(r["best"], yi, marker="o", ms=6.5, color="black", zorder=4)
    # width printed at the right
    ax.text(1.035, yi, f"width {r['width']:.2f}", va="center", ha="left", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels([r["label"] for r in rows], fontsize=9)
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.6, len(rows) - 0.4)
ax.set_xlabel(r"decision threshold $\theta$", fontsize=10)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.tick_params(axis="x", labelsize=9)
ax.grid(axis="x", color="0.88", lw=0.7, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)

# legend, drawn in the same greys as the marks themselves
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
handles = [
    Line2D([], [], marker="o", ms=6.5, color="black", ls="none",
           label=r"$F_1$-maximising $\theta$"),
    Patch(facecolor=GREY, edgecolor="black",
          label=r"within 1% of that row's max $F_1$"),
]
ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.02, -0.60),
          ncol=2, frameon=False, fontsize=8.5, handlelength=1.6)

fig.subplots_adjust(left=0.30, right=0.86, top=0.97, bottom=0.34)
fig.savefig(OUT + ".pdf", dpi=300)
fig.savefig(OUT + ".png", dpi=300)
print("wrote", OUT + ".pdf", "and .png")
