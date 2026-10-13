#!/usr/bin/env python3
"""Figures for the rebuilt experimental section (FuseCC v2 manuscript).

Each figure reads directly from a results/ artefact; the source path is printed and also
noted in the LaTeX comment above the figure. Outputs -> results/exp_figures/ (300 dpi,
>=9 pt fonts, no in-figure title, colourblind-safe).
  fig_param_sensitivity  <- results/rq1_classifiers/{bcb,gcj}/stage1_grid.csv
  fig_family_importance  <- results/f1_analysis/{bcb,gcj_zs}/family_summary.json
  fig_threshold_sweep    <- results/f2_threshold/*_curve.csv
  fig_pareto             <- results/rq5_efficiency + rq1 fit_s (artefact) + DL train times (logs, flagged)
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams.update({"font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
                            "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
                            "pdf.fonttype": 42, "ps.fonttype": 42})
REPO = _REPO_ROOT
OUT = f"{REPO}/results/exp_figures"
os.makedirs(OUT, exist_ok=True)
# Okabe-Ito colourblind-safe palette
CB = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def read_grid(ds):
    fams = {}
    for r in csv.DictReader(open(f"{REPO}/results/rq1_classifiers/{ds}/stage1_grid.csv")):
        try:
            fams.setdefault(r["family"], []).append((float(r["param"]), float(r["mean_f1"])))
        except (ValueError, TypeError):
            continue   # families with no numeric depth param (e.g. GBDT)
    return fams


def param_sensitivity():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, ds, title in zip(axes, ["bcb", "gcj"], ["BigCloneBench (T1--T3)", "Google Code Jam"]):
        fams = read_grid(ds)
        # depth-parameterised tree families only (comparable x-axis = max depth)
        show = [f for f in ["XGBoost", "LightGBM", "CatBoost", "RF", "DT"] if f in fams]
        for i, f in enumerate(show):
            xy = sorted(fams[f])
            x = [a for a, _ in xy]; y = [b for _, b in xy]
            ax.plot(x, y, marker="o", ms=3, color=CB[i % len(CB)], label=f)
        ax.set_xlabel("max depth")
        ax.set_ylabel("mean $F_1$ (3-fold)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right", ncol=1)
    save(fig, "fig_param_sensitivity")


def family_importance():
    order = ["lexical_fingerprint", "semantic_metric", "embedding_cos",
             "embedding_diff", "embedding_prod"]
    pretty = {"lexical_fingerprint": "lexical/\nfingerprint", "semantic_metric": "semantic\nmetric",
              "embedding_cos": "emb.\ncosine", "embedding_diff": "emb.\n|diff|", "embedding_prod": "emb.\nprod"}
    datasets = [("bcb", "BCB (regime-all)"), ("gcj_zs", "GCJ (zero-shot)")]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    w = 0.38
    xs = np.arange(len(order))
    for k, (ds, lab) in enumerate(datasets):
        d = json.load(open(f"{REPO}/results/f1_analysis/{ds}/family_summary.json"))["family_perm_f1_drop"]
        vals = [max(d.get(f, 0.0), 0.0) for f in order]
        # hatch as well as colour: the two series must stay separable in greyscale
        ax.bar(xs + (k - 0.5) * w, vals, w, color=CB[k], label=lab,
               hatch=["", "///"][k], edgecolor="black", linewidth=0.7)
    ax.set_xticks(xs, [pretty[f] for f in order])
    ax.set_ylabel("permutation $F_1$ drop")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_family_importance")


def threshold_sweep():
    curves = [("gcj_148ft_meta_curve.csv", "GCJ 148 (fine-tuned)", CB[0]),
              ("gcj_148zs_meta_curve.csv", "GCJ 148 (zero-shot)", CB[1]),
              ("bcb_16feat_stack_curve.csv", "BCB stacking", CB[2])]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for fn, lab, c in curves:
        p = f"{REPO}/results/f2_threshold/{fn}"
        if not os.path.exists(p):
            continue
        t = [float(r["threshold"]) for r in csv.DictReader(open(p))]
        f1 = [float(r["f1"]) for r in csv.DictReader(open(p))]
        ax.plot(t, f1, color=c, label=lab)
    ax.set_xlabel("decision threshold $\\theta$")
    ax.set_ylabel("$F_1$")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower center")
    ax.grid(alpha=0.3)
    save(fig, "fig_threshold_sweep")


def pareto():
    # y = F1, x = training wall-clock (s, log). Artefact-backed times marked (A); DL times
    # from training logs marked (L) in the caption. Points on GCJ.
    pts = [
        # name, F1, seconds, source-tag, marker
        ("FuseCC (148, XGB)", 0.9998, 10.0, "A", "*"),      # rq5 xgb148_gpu fit 10s; F1 gcj random 148 (train time only)
        ("XGBoost-19", 0.8331, 26.8, "A", "o"),             # rq1 gcj fit_s
        ("LightGBM-19", 0.8225, 23.8, "A", "o"),
        ("CatBoost-19", 0.8090, 8.1, "A", "o"),
        ("RandomForest-19", 0.8031, 99.0, "A", "o"),
        ("AdaBoost-19", 0.7612, 1294.5, "A", "o"),
        ("ASTNN", 0.9975, 43200.0, "L", "s"),               # ~12 h train (logs)
        ("FA-AST+GMN", 0.9778, 162000.0, "L", "s"),         # ~4.5 h/epoch x10 (logs)
        ("MAGNET", 0.9765, 121020.0, "L", "s"),             # ~33.6 h (logs)
    ]
    # Per-point label offsets. MAGNET and FA-AST+GMN sit 0.13 apart on the
    # log axis and their labels overlapped; AdaBoost-19's label sat under the legend.
    # Values are untouched -- only where each label is drawn.
    off = {"MAGNET": (-6, -10, "right"), "FA-AST+GMN": (4, 4, "left"),
           "ASTNN": (-4, 5, "right"), "AdaBoost-19": (-4, 4, "right"),
           "RandomForest-19": (4, -8, "left"), "CatBoost-19": (4, -8, "left")}
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for name, f1, sec, tag, mk in pts:
        c = CB[0] if tag == "A" else CB[1]
        ax.scatter(np.log10(sec), f1, s=70 if mk == "*" else 40, marker=mk, color=c, zorder=3,
                   edgecolor="black", linewidth=0.4)
        dx, dy, ha = off.get(name, (4, 3, "left"))
        ax.annotate(name, (np.log10(sec), f1), fontsize=7, ha=ha,
                    xytext=(dx, dy), textcoords="offset points")
    ax.set_xlabel("training wall-clock $\\log_{10}$(s)")
    ax.set_ylabel("$F_1$ on Google Code Jam")
    ax.grid(alpha=0.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=CB[0],
                              markeredgecolor="black", label="feature+tree (artefact time)"),
                       Line2D([0], [0], marker="s", color="w", markerfacecolor=CB[1],
                              markeredgecolor="black", label="deep model (log time)")],
              loc="center left", bbox_to_anchor=(0.26, 0.55), framealpha=0.95)
    save(fig, "fig_pareto")


if __name__ == "__main__":
    param_sensitivity()
    family_importance()
    threshold_sweep()
    pareto()
    print("done ->", OUT)
