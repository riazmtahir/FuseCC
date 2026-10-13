"""Wilcoxon signed-rank tests across CV folds.

Six systems evaluated on the SAME 10 StratifiedKFold folds of the same GCJ pair
rows (the results/gcj_20/results.csv dropna order that underlies both feature
matrices), XGB(GPU) with fixed params, F1@0.5 per fold:

  ours_ft_148     19 metrics + fine-tuned UniXcoder interactions
  ours_zs_148     19 metrics + zero-shot UniXcoder interactions
  handcrafted_19  t1..t19 only
  unixcoder_only / graphcodebert_only / codebert_only
                  cosine + 64-dim PCA |a-b|, a*b (129 feats, F4 baseline recipe)

Wilcoxon (two-sided) on paired fold-F1 vectors: ours vs every baseline.
Writes results/f4_wilcoxon/summary.json.

Run:  nohup python tools/f4_wilcoxon.py > logs/f4_wilcoxon.log 2>&1 &
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
N_SPLITS = 10
OUT_DIR = "results/f4_wilcoxon"

xgb_params = dict(n_estimators=500, max_depth=13, learning_rate=0.15,
                  subsample=0.8, colsample_bytree=1.0,
                  tree_method="hist", device="cuda",
                  random_state=RANDOM_STATE, eval_metric="logloss")


def embed_only_features(tag, path_a, path_b):
    with open(f"dataset/GCJ/embeddings_{tag}.pkl", "rb") as f:
        emb = pickle.load(f)
    paths = sorted(emb.keys())
    mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
    mean_vec = mat.mean(axis=0, keepdims=True)
    c = mat - mean_vec
    c = c / np.linalg.norm(c, axis=1, keepdims=True)
    pc = PCA(n_components=64, random_state=RANDOM_STATE).fit_transform(c)
    idx = {p: i for i, p in enumerate(paths)}
    ia = pd.Series(path_a).map(idx).values.astype(int)
    ib = pd.Series(path_b).map(idx).values.astype(int)
    A, B = pc[ia].astype(np.float32), pc[ib].astype(np.float32)
    cos = (c[ia] * c[ib]).sum(axis=1).astype(np.float32)
    return np.hstack([cos[:, None], np.abs(A - B), A * B])


ft = np.load("results/gcj_148_ft/features.npz", allow_pickle=True)
zs = np.load("results/gcj_148/features.npz", allow_pickle=True)
y = ft["y"].astype(int)
assert np.array_equal(y, zs["y"].astype(int)), "row order mismatch between feature files"
path_a, path_b = ft["path_a"], ft["path_b"]

systems = {
    "ours_ft_148": ft["X"],
    "ours_zs_148": zs["X"],
    "handcrafted_19": zs["X"][:, :19],
    "unixcoder_only": embed_only_features("unixcoder", path_a, path_b),
    "graphcodebert_only": embed_only_features("graphcodebert", path_a, path_b),
    "codebert_only": embed_only_features("codebert", path_a, path_b),
}

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
folds = list(cv.split(np.zeros(len(y)), y))
fold_f1 = {name: [] for name in systems}
for name, X in systems.items():
    for k, (tr, va) in enumerate(folds):
        m = xgb.XGBClassifier(**xgb_params)
        m.fit(X[tr], y[tr])
        f1 = f1_score(y[va], (m.predict_proba(X[va])[:, 1] >= 0.5).astype(int))
        fold_f1[name].append(float(f1))
    print(f"{name:20s} mean F1 {np.mean(fold_f1[name]):.4f} "
          f"(folds: {[round(v, 4) for v in fold_f1[name]]})", flush=True)

tests = {}
for ours in ("ours_ft_148", "ours_zs_148"):
    for base in systems:
        if base == ours or base.startswith("ours"):
            continue
        stat, p = wilcoxon(fold_f1[ours], fold_f1[base])
        tests[f"{ours}_vs_{base}"] = {"statistic": float(stat), "p_value": float(p),
                                      "mean_delta": float(np.mean(fold_f1[ours])
                                                          - np.mean(fold_f1[base]))}
stat, p = wilcoxon(fold_f1["ours_ft_148"], fold_f1["ours_zs_148"])
tests["ours_ft_148_vs_ours_zs_148"] = {"statistic": float(stat), "p_value": float(p),
                                       "mean_delta": float(np.mean(fold_f1["ours_ft_148"])
                                                           - np.mean(fold_f1["ours_zs_148"]))}

os.makedirs(OUT_DIR, exist_ok=True)
with open(f"{OUT_DIR}/summary.json", "w") as f:
    json.dump({"n_splits": N_SPLITS, "fold_f1": fold_f1, "wilcoxon": tests}, f, indent=2)
print(json.dumps(tests, indent=2), flush=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402
save_run_meta(OUT_DIR)
print("saved", f"{OUT_DIR}/summary.json")
