"""Paper figures: 19-metric Spearman correlation heatmaps + confusion matrices.

Heatmap: signed Spearman correlation between the 19 similarity metrics
(diverging RdBu_r, neutral midpoint, range [-1,1]); computed on a 100k
stratified subsample per regime for speed (rank correlation is stable).
Confusion matrix: XGBoost(t1-t19) under the RQ2 protocol (stratified 60/20/20,
threshold chosen on val at best F1, evaluated on test); counts + row-%.

Regimes: bcb_all, bcb_t13, gcj. Outputs 300-dpi PDF+PNG per figure plus
confusion counts JSON -> results/rq_figures/.
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 12345
REPO = _REPO_ROOT
OUT = f"{REPO}/results/rq_figures"
TNAMES = [f"t{i}" for i in range(1, 20)]
REG_TITLES = {"bcb_all": "BCB (all pairs)", "bcb_t13": "BCB (T1–T3)",
              "gcj": "GCJ"}

plt.rcParams.update({"font.size": 8, "font.family": "serif",
                     "axes.linewidth": 0.6})


def load(regime):
    if regime.startswith("bcb"):
        d = np.load(f"{REPO}/results/bcb_e3/features.npz", allow_pickle=True)
        X = np.nan_to_num(d["X"][:, :19].astype(np.float64), nan=0.0)
        y, ct = d["y"].astype(int), d["clone_type"]
        if regime == "bcb_t13":
            keep = ct != "WT3/T4"
            X, y = X[keep], y[keep]
        return X, y
    d = np.load(f"{REPO}/results/gcj_148/features.npz", allow_pickle=True)
    return (np.nan_to_num(d["X"][:, :19].astype(np.float64), nan=0.0),
            d["y"].astype(int))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png", flush=True)


def heatmap(X, regime):
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(X), min(100_000, len(X)), replace=False)
    rho = spearmanr(X[idx]).statistic
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(19), TNAMES, rotation=90)
    ax.set_yticks(range(19), TNAMES)
    ax.set_title(f"Spearman correlation of similarity metrics — "
                 f"{REG_TITLES[regime]}", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(labelsize=7)
    ax.tick_params(length=0)
    save(fig, f"heatmap_spearman_{regime}")
    return rho


def confusion(X, y, regime):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp],
                              random_state=SEED)
    clf = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                        tree_method="hist", device="cuda", n_jobs=8,
                        random_state=SEED, eval_metric="logloss")
    clf.fit(X[tr], y[tr])
    pv = clf.predict_proba(X[va])[:, 1]
    # best-F1 threshold on val (B2), same grid style as rq2
    qs = np.unique(np.quantile(pv, np.linspace(0.001, 0.999, 999)))
    f1s = [precision_recall_fscore_support(y[va], (pv >= t).astype(int),
                                           average="binary",
                                           zero_division=0)[2] for t in qs]
    thr = float(qs[int(np.argmax(f1s))])
    pt = clf.predict_proba(X[te])[:, 1]
    pred = (pt >= thr).astype(int)
    cm = confusion_matrix(y[te], pred)  # rows=true [nc, clone]
    p, r, f1, _ = precision_recall_fscore_support(y[te], pred, average="binary",
                                                  zero_division=0)

    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    im = ax.imshow(cm, cmap="Blues",
                   vmax=cm.max() * 1.15)  # headroom so text stays legible
    labels = ["Non-clone", "Clone"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{REG_TITLES[regime]} — XGBoost (t1–t19)\n"
                 f"thr={thr:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}",
                 fontsize=7.5)
    row_sum = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / row_sum[i, 0]
            color = "white" if cm[i, j] > 0.55 * cm.max() else "#1a1a1a"
            ax.text(j, i, f"{cm[i, j]:,}\n({frac:.1%})", ha="center",
                    va="center", fontsize=8, color=color)
    ax.tick_params(length=0)
    save(fig, f"confusion_{regime}")
    return {"threshold_on_val": thr, "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
            "precision": round(float(p), 4), "recall": round(float(r), 4),
            "f1": round(float(f1), 4), "n_test": int(len(te))}


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for regime in ["bcb_all", "bcb_t13", "gcj"]:
        print(f"[{regime}]", flush=True)
        X, y = load(regime)
        rho = heatmap(X, regime)
        tri = np.abs(rho[np.triu_indices(19, 1)])
        meta[regime] = {"mean_abs_offdiag_spearman": round(float(tri.mean()), 3),
                        "confusion": confusion(X, y, regime)}
    with open(os.path.join(OUT, "figures_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(OUT)
    except Exception as e:
        print(f"run_meta skipped: {e!r}")
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == "__main__":
    main()
