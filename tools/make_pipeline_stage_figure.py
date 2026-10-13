#!/usr/bin/env python
"""Figure "Data at each pipeline stage" for the methodology section.

Four panels, one per pipeline stage of Section 3, every one drawn from a real
artifact -- no synthetic data, no schematic content:

  (a) 3.3 tokenisation      token length per function
                            <- dataset/BCB/function_token_counts.csv
  (b) 3.4 similarity        the nineteen metric scores, clone vs non-clone
                            <- results/bcb_e3/features.npz  X[:, 0:19], y
  (c) 3.5 frozen embedding  the embedding cosine, clone vs non-clone
                            <- results/bcb_e3/features.npz  X[:, 19]
  (d) 3.6 classification    classifier probability + the decision threshold
                            <- results/bcb_leakage_isolation/bcb_e3_oof_t123.npy

All four panels are BigCloneBench, so the figure follows one corpus through the
pipeline rather than mixing datasets.  Panel (d) is the T1-T3 regime, matching
the vector that produced it.

Greyscale-legible: the two classes differ by hatch as well as by fill, and the
colours are a light/dark pair rather than two hues.

Writes Figures/fig_pipeline_stages.{pdf,png} at 300 dpi.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "manuscript_v2", "Figures")

TOKENS_CSV = os.path.join(ROOT, "dataset", "BCB", "function_token_counts.csv")
FEATURES = os.path.join(ROOT, "results", "bcb_e3", "features.npz")
OOF = os.path.join(ROOT, "results", "bcb_leakage_isolation", "bcb_e3_oof_t123.npy")

# light fill / dark fill, distinguishable in greyscale; hatch carries the class
NONCLONE = dict(facecolor="#c9c9c9", edgecolor="#3a3a3a", hatch="///")
CLONE = dict(facecolor="#4c4c4c", edgecolor="#101010", hatch="")

T123 = {"T1", "T2", "VST3", "ST3", "MT3"}


def panel_tokens(ax):
    df = pd.read_csv(TOKENS_CSV)
    n = df["tokens"].to_numpy()
    n = n[n > 0]
    ax.hist(np.log10(n), bins=60, color="#6f6f6f", edgecolor="#202020", linewidth=0.3)
    med = np.median(n)
    ax.axvline(np.log10(med), color="black", linestyle="--", linewidth=1.1)
    ax.annotate(f"median {int(med)}", xy=(np.log10(med), ax.get_ylim()[1] * 0.86),
                xytext=(4, 0), textcoords="offset points", fontsize=9, va="top")
    ax.set_xlabel(r"token length $n_f$ (log$_{10}$)", fontsize=10)
    ax.set_ylabel("functions", fontsize=10)
    ax.set_title(r"(a) Stage 3.3: $T_f=\tau_{\mathrm{tok}}(f)$", fontsize=10.5)
    return f"{len(n):,} functions"


def panel_metrics(ax, X, y):
    """Median metric score per class, for the 16 measures bounded in [0,1].

    t5/t13/t14 are unbounded (Section 3.4) so they cannot share this axis; they
    are named in the panel rather than silently dropped.
    """
    # column k holds t_{k+1}; drop t5 (col 4), t13 (col 12), t14 (col 13)
    bounded = [k for k in range(19) if k not in (4, 12, 13)]
    labels = [f"$t_{{{k+1}}}$" for k in bounded]
    m_non = [np.median(X[y == 0, k]) for k in bounded]
    m_cl = [np.median(X[y == 1, k]) for k in bounded]
    idx = np.arange(len(bounded))
    w = 0.4
    ax.bar(idx - w / 2, m_non, w, label="non-clone", **NONCLONE)
    ax.bar(idx + w / 2, m_cl, w, label="clone", **CLONE)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("median score", fontsize=10)
    ax.set_title(r"(b) Stage 3.4: $s_{ij}\in\mathbb{R}^{19}$", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2,
              bbox_to_anchor=(0.0, 1.02))


def panel_cosine(ax, X, y):
    c_non, c_cl = X[y == 0, 19], X[y == 1, 19]
    bins = np.linspace(-0.4, 1.0, 70)
    ax.hist(c_non, bins=bins, density=True, label="non-clone", **NONCLONE)
    ax.hist(c_cl, bins=bins, density=True, alpha=0.85, label="clone", **CLONE)
    ax.set_xlabel(r"$\cos(e_i,e_j)$", fontsize=10)
    ax.set_ylabel("density", fontsize=10)
    ax.set_title(r"(c) Stage 3.5: frozen-encoder cosine", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")


def panel_scores(ax, p, y, thr=0.5):
    bins = np.linspace(0, 1, 60)
    ax.hist(p[y == 0], bins=bins, density=True, label="non-clone", **NONCLONE)
    ax.hist(p[y == 1], bins=bins, density=True, alpha=0.85, label="clone", **CLONE)
    ax.set_yscale("log")
    ax.axvline(thr, color="black", linestyle="--", linewidth=1.2)
    ax.annotate(r"$\theta$", xy=(thr, ax.get_ylim()[1] * 0.25),
                xytext=(5, 0), textcoords="offset points", fontsize=11)
    ax.set_xlabel(r"$g(x_{ij})$", fontsize=10)
    ax.set_ylabel("density (log)", fontsize=10)
    ax.set_title(r"(d) Stage 3.6: classifier output", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left",
              bbox_to_anchor=(0.12, 1.0))


def main():
    for p in (TOKENS_CSV, FEATURES, OOF):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")
    os.makedirs(OUT, exist_ok=True)

    d = np.load(FEATURES, allow_pickle=True)
    X, y, ct = d["X"], d["y"].astype(int), d["clone_type"]
    keep = np.array([c in T123 or c == "NC" for c in ct])
    Xk, yk = X[keep], y[keep]

    p = np.load(OOF)
    # the OOF vector is the T1-T3 regime of bcb_e3, in the same row order
    if p.shape[0] != keep.sum():
        sys.exit(f"OOF length {p.shape[0]} != T1-T3 rows {keep.sum()}")

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.4))
    note = panel_tokens(axes[0, 0])
    panel_metrics(axes[0, 1], Xk, yk)
    panel_cosine(axes[1, 0], Xk, yk)
    panel_scores(axes[1, 1], p, yk)
    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        f = os.path.join(OUT, f"fig_pipeline_stages.{ext}")
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print("wrote", f)
    print("panel (a):", note, "| panels (b)-(d): BCB T1-T3,", f"{keep.sum():,} pairs")


if __name__ == "__main__":
    main()
