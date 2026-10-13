"""RQ1 rerun (paper section "Enhancing Clone Detection with Tokenization and ML"),
new pipeline: t1-t19 hand-crafted metrics, new tokenizer, new classifier set.

Stage 1 (parameter study, feeds the parameter figure): stratified 100k subsample,
3-fold CV, per-family hyperparameter grid (tree depth / k / C).
Stage 2 (final comparison, feeds the table + significance): best config per family,
10-fold shared StratifiedKFold on the FULL data, P/R/F1@0.5 + AUC per fold,
Wilcoxon signed-rank on F1 folds vs the best system (10 folds -> min two-sided
p = 0.00195, same standard as tools/f4_wilcoxon.py). Fit/predict wall-clock per
fold is recorded (feeds the efficiency table).

BCB uses the Krinke-compliant T1-T3 regime (WT3/T4 excluded; regime-all coverage
lives in results/bcb_e3). GCJ uses all pairs, random-split protocol (classifier
selection is relative; honest protocols for the final system are in gcj_148_ft).
KNN trains on a 100k stratified subsample per fold (full-corpus KNN is
memory/time-prohibitive; noted in the caption).

Usage: rq1_classifier_comparison.py --dataset bcb|gcj [--sample N] [--quick]
Outputs: results/rq1_classifiers/<dataset>/{summary.json,stage1_grid.csv,stage2_folds.csv}
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 12345
REPO = _REPO_ROOT
KNN_TRAIN_CAP = 100_000


def load(dataset):
    if dataset == "bcb":
        d = np.load(f"{REPO}/results/bcb_e3/features.npz", allow_pickle=True)
        keep = d["clone_type"] != "WT3/T4"
        return (np.nan_to_num(d["X"][keep, :19].astype(np.float32), nan=0.0),
                d["y"][keep].astype(int))
    d = np.load(f"{REPO}/results/gcj_148/features.npz", allow_pickle=True)
    return (np.nan_to_num(d["X"][:, :19].astype(np.float32), nan=0.0),
            d["y"].astype(int))


def make_model(family, p):
    if family == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=p,
                             tree_method="hist", device="cuda", n_jobs=-1,
                             random_state=SEED, eval_metric="logloss")
    if family == "LightGBM":
        from lightgbm import LGBMClassifier
        leaves = 255 if p in (-1, 12, 16) else 2 ** p - 1
        return LGBMClassifier(n_estimators=400, learning_rate=0.1, max_depth=p,
                              num_leaves=leaves, n_jobs=-1, random_state=SEED,
                              verbose=-1)
    if family == "CatBoost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=800, learning_rate=0.1, depth=p,
                                  task_type="GPU", devices="0", verbose=0,
                                  random_seed=SEED, allow_writing_files=False)
    if family == "RF":
        return RandomForestClassifier(n_estimators=200, max_depth=p, n_jobs=-1,
                                      random_state=SEED)
    if family == "DT":
        return DecisionTreeClassifier(max_depth=p, random_state=SEED)
    if family == "AdaBoost":
        return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=p),
                                  n_estimators=200, random_state=SEED)
    if family == "GBDT":
        return HistGradientBoostingClassifier(max_iter=300, max_depth=p,
                                              random_state=SEED)
    if family == "KNN":
        return KNeighborsClassifier(n_neighbors=p, n_jobs=-1)
    if family == "LR":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=p, max_iter=2000,
                                                random_state=SEED))
    raise ValueError(family)


GRIDS = {
    "XGBoost": [4, 6, 8, 12, 16],
    "LightGBM": [-1, 4, 8, 12, 16],
    "CatBoost": [4, 6, 8],
    "RF": [8, 16, 32, None],
    "DT": [8, 16, 32, 64, None],
    "AdaBoost": [1, 2, 3],
    "GBDT": [None, 8, 16],
    "KNN": [1, 3, 5],
    "LR": [0.1, 1.0, 10.0],
}


def fit_eval(family, p, Xtr, ytr, Xte, yte):
    m = make_model(family, p)
    if family == "KNN" and len(ytr) > KNN_TRAIN_CAP:
        rng = np.random.RandomState(SEED)
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        f = KNN_TRAIN_CAP / len(ytr)
        sub = np.concatenate([rng.choice(pos, int(len(pos) * f), replace=False),
                              rng.choice(neg, int(len(neg) * f), replace=False)])
        Xtr, ytr = Xtr[sub], ytr[sub]
    t0 = time.time()
    m.fit(Xtr, ytr)
    t_fit = time.time() - t0
    t0 = time.time()
    prob = m.predict_proba(Xte)[:, 1]
    t_pred = time.time() - t0
    pred = (prob >= 0.5).astype(int)
    p_, r_, f1_, _ = precision_recall_fscore_support(yte, pred, average="binary",
                                                     zero_division=0)
    return {"precision": float(p_), "recall": float(r_), "f1": float(f1_),
            "auc": float(roc_auc_score(yte, prob)),
            "fit_s": round(t_fit, 2), "pred_s": round(t_pred, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bcb", "gcj"], required=True)
    ap.add_argument("--sample", type=int, default=0, help="debug subsample")
    ap.add_argument("--quick", action="store_true",
                    help="stage1 20k/2-fold, stage2 3-fold (smoke test)")
    args = ap.parse_args()

    X, y = load(args.dataset)
    rng = np.random.RandomState(SEED)
    if args.sample:
        idx = rng.choice(len(y), args.sample, replace=False)
        X, y = X[idx], y[idx]
    out = f"{REPO}/results/rq1_classifiers/{args.dataset}"
    os.makedirs(out, exist_ok=True)
    print(f"[{args.dataset}] X {X.shape} clone-rate {y.mean():.3f}", flush=True)

    # ---------- stage 1: hyperparameter study on a subsample ----------
    n_sub = 20_000 if args.quick else 100_000
    n_f1 = 2 if args.quick else 3
    if len(y) > n_sub:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        f = n_sub / len(y)
        sub = np.concatenate([rng.choice(pos, int(len(pos) * f), replace=False),
                              rng.choice(neg, int(len(neg) * f), replace=False)])
    else:
        sub = np.arange(len(y))
    Xs, ys = X[sub], y[sub]
    folds1 = list(StratifiedKFold(n_f1, shuffle=True, random_state=SEED).split(Xs, ys))

    grid_rows, best = [], {}
    for family, grid in GRIDS.items():
        for p in grid:
            f1s = []
            for tr, te in folds1:
                r = fit_eval(family, p, Xs[tr], ys[tr], Xs[te], ys[te])
                f1s.append(r["f1"])
            mf1 = float(np.mean(f1s))
            grid_rows.append({"family": family, "param": str(p), "mean_f1": mf1})
            if family not in best or mf1 > best[family][1]:
                best[family] = (p, mf1)
            print(f"stage1 {family:9s} param={p!s:5s} F1 {mf1:.4f}", flush=True)
    pd.DataFrame(grid_rows).to_csv(os.path.join(out, "stage1_grid.csv"), index=False)

    # ---------- stage 2: full-data 10-fold shared-fold comparison ----------
    n_f2 = 3 if args.quick else 10
    folds2 = list(StratifiedKFold(n_f2, shuffle=True, random_state=SEED).split(X, y))
    fold_rows = []
    for family, (p, _) in best.items():
        for k, (tr, te) in enumerate(folds2):
            r = fit_eval(family, p, X[tr], y[tr], X[te], y[te])
            r.update({"family": family, "param": str(p), "fold": k})
            fold_rows.append(r)
            print(f"stage2 {family:9s} fold {k}: F1 {r['f1']:.4f} "
                  f"AUC {r['auc']:.4f} fit {r['fit_s']}s", flush=True)
    df = pd.DataFrame(fold_rows)
    df.to_csv(os.path.join(out, "stage2_folds.csv"), index=False)

    # ---------- aggregate + Wilcoxon vs the best family ----------
    agg = df.groupby("family").agg(
        param=("param", "first"), f1=("f1", "mean"), f1_std=("f1", "std"),
        precision=("precision", "mean"), recall=("recall", "mean"),
        auc=("auc", "mean"), fit_s=("fit_s", "mean"),
        pred_s=("pred_s", "mean")).sort_values("f1", ascending=False)
    top = agg.index[0]
    top_f1 = df[df.family == top].sort_values("fold")["f1"].values
    pvals = {}
    for fam in agg.index[1:]:
        other = df[df.family == fam].sort_values("fold")["f1"].values
        try:
            pvals[fam] = float(wilcoxon(top_f1, other).pvalue)
        except ValueError:
            pvals[fam] = 1.0  # identical folds
    print(agg.round(4).to_string(), flush=True)
    print("wilcoxon vs", top, {k: round(v, 5) for k, v in pvals.items()}, flush=True)

    summary = {
        "dataset": args.dataset,
        "regime": "T1-T3 (WT3/T4 excluded)" if args.dataset == "bcb"
                  else "all pairs, random-split",
        "features": "t1-t19 (new tokenizer)", "seed": SEED,
        "n_pairs": int(len(y)),
        "stage1": {"subsample": int(len(sub)), "folds": n_f1,
                   "best_params": {k: str(v[0]) for k, v in best.items()}},
        "stage2": {"folds": n_f2, "threshold": 0.5,
                   "knn_train_cap": KNN_TRAIN_CAP,
                   "table": json.loads(agg.round(6).to_json(orient="index")),
                   "wilcoxon_f1_vs_" + top: pvals},
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(out)
    except Exception as e:
        print(f"run_meta skipped: {e!r}")
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
