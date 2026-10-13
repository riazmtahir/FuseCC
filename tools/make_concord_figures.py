#!/usr/bin/env python3
"""Manuscript figures + stats for the ConCord datasets (POJ-104 / Java250).

Fills the gap noted in the writeup audit: BCB/GCJ already have confusion matrices and
Spearman heatmaps (results/rq_figures/), the ConCord datasets did not. This produces the
SAME figure set on the same style so the four datasets are presented consistently, plus a
paired-bootstrap significance test for the feature-family ablation.

Outputs -> results/concord/figures/:
  confusion_concord_<ds>.{pdf,png}   FuseCC-148 (UniXcoder) on the problem-disjoint test,
                                     threshold picked on valid (B2), from the trained model.
  heatmap_spearman_concord_<ds>.{pdf,png}  Spearman correlation of the 19 similarity metrics.
  ablation_bootstrap.json            paired-bootstrap 95% CIs + p-values for
                                     148 vs embeddings-only and 148 vs metrics-only.

All numbers come from the already-trained artifacts (xgb_148.joblib, pca.joblib,
summary.json) on the genuine test split — nothing is re-fit or fabricated.
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

sys.path.insert(0, _REPO_ROOT)
from tools.train_concord import build_features   # identical 148-feature builder

# Journal-quality defaults: >=9 pt fonts, no in-figure titles (captions carry them),
# 300 dpi, colourblind-safe palettes. Addresses Review B (figures too small/low-res).
matplotlib.rcParams.update({
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

REPO = _REPO_ROOT
DATA = f"{REPO}/dataset/concord"
OUTR = f"{REPO}/results/concord"
OUT = f"{OUTR}/figures"
SEED = 12345
TNAMES = [f"t{i}" for i in range(1, 20)]
TITLES = {"java250": "CodeNet-Java250", "poj104": "POJ-104"}


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png", flush=True)


def heatmap(X19, ds):
    # Large enough that at ~0.49\textwidth (figure*) the 19 tick labels stay >=9 pt.
    # RdBu_r is a ColorBrewer colourblind-safe diverging map, apt for [-1,1] correlation.
    rho = spearmanr(X19).statistic
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(19), TNAMES, rotation=90)
    ax.set_yticks(range(19), TNAMES)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Spearman $\\rho$", fontsize=10)
    cb.ax.tick_params(labelsize=9)
    ax.tick_params(length=0)
    # Each of the two panels is a different benchmark and the caption
    # cannot say which is which, so the panel names itself.
    ax.set_title(TITLES[ds], fontsize=10.5, pad=6)
    save(fig, f"heatmap_{ds}")
    tri = np.abs(rho[np.triu_indices(19, k=1)])
    return float(tri.mean())


def confusion(cm, thr, p, r, f1, ds):
    fig, ax = plt.subplots(figsize=(3.4, 3.1))
    im = ax.imshow(cm, cmap="Blues", vmax=cm.max() * 1.15)  # sequential, colourblind-safe
    labels = ["Non-clone", "Clone"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    row_sum = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / row_sum[i, 0]
            color = "white" if cm[i, j] > 0.55 * cm.max() else "#1a1a1a"
            ax.text(j, i, f"{cm[i, j]:,}\n({frac:.1%})", ha="center", va="center",
                    fontsize=11, color=color)
    ax.tick_params(length=0)
    # As above -- the panel names its own benchmark; P/R/F1 stay in the caption.
    ax.set_title(TITLES[ds], fontsize=10.5, pad=6)
    save(fig, f"confusion_{ds}")


def paired_bootstrap(y, prob_a, prob_b, thr_a, thr_b, n_boot=2000):
    """95% CI + two-sided p on F1(a)-F1(b), resampling test pairs with replacement."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        fa = f1_score(yb, (prob_a[idx] >= thr_a).astype(int), zero_division=0)
        fb = f1_score(yb, (prob_b[idx] >= thr_b).astype(int), zero_division=0)
        diffs[i] = fa - fb
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples on the opposite side of 0
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"mean_f1_delta": round(float(diffs.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_two_sided": round(float(min(p, 1.0)), 4)}


def main():
    os.makedirs(OUT, exist_ok=True)
    import joblib
    meta = {}
    ablation = {}
    for ds in ("java250", "poj104"):
        d = os.path.join(DATA, ds)
        f = np.load(os.path.join(d, "features_test.npz"), allow_pickle=True)
        X19 = np.nan_to_num(f["X"][:, :19].astype(np.float64), nan=0.0)
        y = f["y"].astype(int)

        mean_off = heatmap(X19, ds)

        # rebuild the 148 features + predict with the trained FuseCC model
        clf = joblib.load(os.path.join(OUTR, ds, "xgb_148.joblib"))
        pca, mean_vec = joblib.load(os.path.join(OUTR, ds, "pca.joblib"))
        Xte, yte = build_features(d, "test", pca, mean_vec, "")
        summ = json.load(open(os.path.join(OUTR, ds, "summary.json")))["results"]
        thr = summ["xgb_19+emb148"]["test"]["threshold"]
        prob = clf.predict_proba(Xte)[:, 1]
        pred = (prob >= thr).astype(int)
        cm = confusion_matrix(yte, pred)
        tp, fp = int(cm[1, 1]), int(cm[0, 1])
        fn, tn = int(cm[1, 0]), int(cm[0, 0])
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        confusion(cm, thr, p, r, f1, ds)

        # paired-bootstrap significance for the ablation (metrics-only, emb-only vs 148)
        import xgboost as xgb
        def fit_predict(cols):
            m = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1,
                                  subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                                  device="cuda", random_state=SEED, eval_metric="logloss")
            Xtr, ytr = build_features(d, "train", pca, mean_vec, "")
            Xva, yva = build_features(d, "valid", pca, mean_vec, "")
            m.fit(Xtr[:, cols], ytr)
            pv = m.predict_proba(Xva[:, cols])[:, 1]
            pt = m.predict_proba(Xte[:, cols])[:, 1]
            grid = np.linspace(0.05, 0.95, 181)
            t = max(grid, key=lambda z: f1_score(yva, (pv >= z).astype(int), zero_division=0))
            return pt, t
        cols_148 = list(range(148))
        cols_19 = list(range(19))
        cols_emb = list(range(19, 148))
        p148, t148 = prob, thr
        p19, t19 = fit_predict(cols_19)
        pemb, temb = fit_predict(cols_emb)
        ablation[ds] = {
            "f1": {"metrics_only": round(f1_score(yte, (p19 >= t19).astype(int)), 4),
                   "embeddings_only": round(f1_score(yte, (pemb >= temb).astype(int)), 4),
                   "fusecc_148": round(f1, 4)},
            "auc": {"metrics_only": round(roc_auc_score(yte, p19), 4),
                    "embeddings_only": round(roc_auc_score(yte, pemb), 4),
                    "fusecc_148": round(roc_auc_score(yte, p148), 4)},
            "148_vs_embeddings_only": paired_bootstrap(yte, p148, pemb, t148, temb),
            "148_vs_metrics_only": paired_bootstrap(yte, p148, p19, t148, t19),
        }
        meta[ds] = {"mean_abs_offdiag_spearman": round(mean_off, 3),
                    "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp,
                                  "threshold_on_val": round(float(thr), 4),
                                  "precision": round(p, 4), "recall": round(r, 4),
                                  "f1": round(f1, 4), "n_test": int(len(yte))}}
        print(f"[{ds}] mean|off-diag Spearman|={mean_off:.3f}  "
              f"148 F1={f1:.4f} vs emb {ablation[ds]['f1']['embeddings_only']} "
              f"vs metrics {ablation[ds]['f1']['metrics_only']}")

    json.dump({"figures": meta}, open(os.path.join(OUT, "figures_meta.json"), "w"), indent=2)
    json.dump(ablation, open(os.path.join(OUT, "ablation_bootstrap.json"), "w"), indent=2)
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
