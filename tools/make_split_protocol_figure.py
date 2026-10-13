#!/usr/bin/env python
"""Split-protocol figure for the experimental-setup section.

Section 4.1 states the BigCloneBench split geometry, the pair retention and the
measured function recurrence in prose alone.  This figure plots those same
numbers.  Every value is read from an artifact at run time -- nothing is typed
into this script:

  (a) pair retention per split criterion
      <- results/bcb_leakage_isolation/pairlevel_controlled.json  ["splits"]
      <- results/bcb_function_disjoint/summary.json               ["splits"]
  (b) measured train/test recurrence
      <- results/bcb_leakage_isolation/pairlevel_controlled.json
         ["function_recurrence_across_the_split"]; zero by construction in the
         function-disjoint run, which is asserted here rather than assumed
  (c) clone rate per partition, which is why AUC is the stable cross-split
      comparison (Section 8.1)
      <- the same two "splits" blocks

Greyscale-legible: fills are a light-to-dark ramp and every segment is also
hatched; no information is carried by hue alone.

Writes Figures/fig_split_protocols.{pdf,png} at 300 dpi.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "manuscript_v2", "Figures")
PAIRLEVEL = os.path.join(ROOT, "results", "bcb_leakage_isolation", "pairlevel_controlled.json")
FUNCDISJ = os.path.join(ROOT, "results", "bcb_function_disjoint", "summary.json")

STYLE = {                       # split -> (facecolor, hatch)
    "train": ("#d8d8d8", "//"),
    "val": ("#8f8f8f", "xx"),
    "test": ("#4a4a4a", ""),
    "discarded": ("#ffffff", "...."),
}


def load():
    pl = json.load(open(PAIRLEVEL))["splits"]
    fd = json.load(open(FUNCDISJ))["splits"]
    rec = json.load(open(PAIRLEVEL))["function_recurrence_across_the_split"]
    return pl, fd, rec


def panel_retention(ax, pl, fd):
    total = sum(pl[s]["n_pairs"] for s in ("train", "val", "test"))
    fd_used = sum(fd[s]["n_pairs"] for s in ("train", "val", "test"))
    rows = [
        ("function-disjoint", [fd["train"]["n_pairs"], fd["val"]["n_pairs"],
                               fd["test"]["n_pairs"], total - fd_used]),
        ("pair-level", [pl["train"]["n_pairs"], pl["val"]["n_pairs"],
                        pl["test"]["n_pairs"], 0]),
    ]
    names = ["train", "val", "test", "discarded"]
    for r, (label, vals) in enumerate(rows):
        left = 0
        for name, v in zip(names, vals):
            if v == 0:
                continue
            fc, hz = STYLE[name]
            ax.barh(r, v, left=left, height=0.55, facecolor=fc, hatch=hz,
                    edgecolor="#111111", linewidth=0.7,
                    label=name if r == 0 else None)
            left += v
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.set_xlabel("candidate pairs", fontsize=10)
    ax.set_xlim(0, total * 1.02)
    ax.set_xticks([0, 1e5, 2e5, 3e5, 4e5])
    ax.set_xticklabels(["0", "100k", "200k", "300k", "400k"])
    disc_pct = 100.0 * (total - fd_used) / total
    ax.annotate(f"{disc_pct:.1f}% cross-partition, discarded",
                xy=(fd_used + (total - fd_used) / 2, 0), xytext=(0, 26),
                textcoords="offset points", ha="center", fontsize=8.5)
    ax.set_ylim(-0.55, 1.75)
    ax.set_title("(a) pair retention", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.30))
    return total, fd_used, disc_pct


def panel_recurrence(ax, rec):
    vals_pl = [rec["test_functions_also_in_train_pct"],
               rec["test_pairs_with_BOTH_endpoints_seen_in_train_pct"]]
    vals_fd = [0.0, 0.0]
    idx = np.arange(2)
    w = 0.36
    ax.bar(idx - w / 2, vals_pl, w, facecolor="#4a4a4a", edgecolor="#111111",
           linewidth=0.7, label="pair-level")
    ax.bar(idx + w / 2, vals_fd, w, facecolor="#d8d8d8", hatch="//",
           edgecolor="#111111", linewidth=0.7, label="function-disjoint")
    for x, v in zip(idx - w / 2, vals_pl):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    for x in idx + w / 2:
        ax.annotate("0", (x, 0), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8.5)
    ax.set_xticks(idx)
    ax.set_xticklabels(["test functions\nseen in train",
                        "test pairs, both\nendpoints seen"], fontsize=9)
    ax.set_ylim(0, 138)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel("per cent of the test split", fontsize=10)
    ax.set_title("(b) measured recurrence", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.02))


def panel_clonerate(ax, pl, fd):
    splits = ["train", "val", "test"]
    idx = np.arange(3)
    w = 0.36
    r_pl = [pl[s]["clone_rate"] for s in splits]
    r_fd = [fd[s]["clone_rate"] for s in splits]
    ax.bar(idx - w / 2, r_pl, w, facecolor="#4a4a4a", edgecolor="#111111",
           linewidth=0.7, label="pair-level")
    ax.bar(idx + w / 2, r_fd, w, facecolor="#d8d8d8", hatch="//",
           edgecolor="#111111", linewidth=0.7, label="function-disjoint")
    for x, v in list(zip(idx - w / 2, r_pl)) + list(zip(idx + w / 2, r_fd)):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=8)
    ax.set_xticks(idx)
    ax.set_xticklabels(splits, fontsize=9.5)
    ax.set_ylim(0, 0.82)
    ax.set_ylabel("clone rate", fontsize=10)
    ax.set_title("(c) class balance per partition", fontsize=10.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper left")


def main():
    for p in (PAIRLEVEL, FUNCDISJ):
        if not os.path.exists(p):
            sys.exit(f"missing artifact: {p}")
    os.makedirs(OUT, exist_ok=True)
    pl, fd, rec = load()

    # the function-disjoint run removes recurrence by construction; assert the
    # artifact still says so rather than drawing a zero we merely believe
    fdj = json.load(open(FUNCDISJ))
    assert "no function occurs in more than one split" in fdj["protocol"]["split"], \
        fdj["protocol"]["split"]

    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8,
                         "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.3))
    total, used, disc = panel_retention(axes[0], pl, fd)
    panel_recurrence(axes[1], rec)
    panel_clonerate(axes[2], pl, fd)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        f = os.path.join(OUT, f"fig_split_protocols.{ext}")
        fig.savefig(f, dpi=300, bbox_inches="tight")
        print("wrote", f)
    print(f"total {total:,} pairs; function-disjoint keeps {used:,} "
          f"({100-disc:.1f}%), discards {disc:.1f}%")


if __name__ == "__main__":
    main()
