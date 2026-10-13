"""Embedding-only GCJ baselines re-run in our unified environment.

For a given frozen encoder (codebert / graphcodebert / unixcoder), build
embedding-only pair features (centered cosine + |a-b| and a*b in 64-dim PCA
space = 129 features, the same recipe as our full pipeline but WITHOUT the 19
hand-crafted metrics) and evaluate with XGBoost under both protocols:

  random-split : 3-fold StratifiedKFold OOF (comparable to literature baselines)
  file-disjoint: test = models/unixcoder_gcj_ft/val_files.txt (same split as our
                 honest protocol), threshold picked on a val carve-out (B2)

Writes results/gcj_baselines/<tag>.json.
Run:  python tools/baseline_embed_only_gcj.py --model codebert
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/gcj_baselines"
VAL_FILES = "models/unixcoder_gcj_ft/val_files.txt"

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True,
                    choices=["codebert", "graphcodebert", "unixcoder"])
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()

with open(f"dataset/GCJ/embeddings_{args.model}.pkl", "rb") as f:
    emb = pickle.load(f)
paths = sorted(emb.keys())
mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
mean_vec = mat.mean(axis=0, keepdims=True)
c = mat - mean_vec
c = c / np.linalg.norm(c, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=RANDOM_STATE)
pc = pca.fit_transform(c)
idx = {p: i for i, p in enumerate(paths)}

df = pd.read_csv("dataset/GCJ/pairs_mapped.csv")
if args.sample:
    df = df.sample(args.sample, random_state=RANDOM_STATE).reset_index(drop=True)
y = df["label"].astype(int).values
ia = df["path_a"].map(idx).values.astype(int)
ib = df["path_b"].map(idx).values.astype(int)
A, B = pc[ia].astype(np.float32), pc[ib].astype(np.float32)
cos = (c[ia] * c[ib]).sum(axis=1).astype(np.float32)
X = np.hstack([cos[:, None], np.abs(A - B), A * B])
path_a, path_b = df["path_a"].values, df["path_b"].values
print(f"{args.model}: X {X.shape}", flush=True)

xgb_params = dict(n_estimators=500, max_depth=13, learning_rate=0.15,
                  subsample=0.8, colsample_bytree=1.0,
                  tree_method="hist", device="cuda",
                  random_state=RANDOM_STATE, eval_metric="logloss")


def metrics(y_true, proba, thr):
    pred = (proba >= thr).astype(int)
    return {"threshold": float(thr),
            "f1": float(f1_score(y_true, pred)),
            "precision": float(precision_score(y_true, pred)),
            "recall": float(recall_score(y_true, pred)),
            "auc": float(roc_auc_score(y_true, proba))}


results = {"model": args.model, "features": int(X.shape[1])}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y))
for k, (tr, va) in enumerate(cv.split(X, y)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    print(f"random-split fold {k+1}/3 done", flush=True)
results["random_split"] = metrics(y, oof, 0.5)
print("random-split:", results["random_split"], flush=True)

with open(VAL_FILES) as f:
    test_files = set(line.strip() for line in f if line.strip())
in_test = np.isin(path_a, list(test_files)) | np.isin(path_b, list(test_files))
both_test = np.isin(path_a, list(test_files)) & np.isin(path_b, list(test_files))
train_files = sorted((set(path_a) | set(path_b)) - test_files)
rng = np.random.RandomState(RANDOM_STATE)
thr_files = set(rng.choice(train_files, size=int(0.15 * len(train_files)), replace=False))
a_thr, b_thr = np.isin(path_a, list(thr_files)), np.isin(path_b, list(thr_files))
fit_mask = ~in_test & ~a_thr & ~b_thr
thr_mask = ~in_test & a_thr & b_thr

m = xgb.XGBClassifier(**xgb_params)
m.fit(X[fit_mask], y[fit_mask])
p_thr = m.predict_proba(X[thr_mask])[:, 1]
grid = np.linspace(0.01, 0.99, 99)
t_star = float(grid[int(np.argmax([f1_score(y[thr_mask], (p_thr >= t).astype(int))
                                   for t in grid]))])
p_test = m.predict_proba(X[both_test])[:, 1]
results["file_disjoint"] = metrics(y[both_test], p_test, t_star)
results["file_disjoint"]["n_test"] = int(both_test.sum())
print("file-disjoint:", results["file_disjoint"], flush=True)

if not args.sample:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{args.model}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved", os.path.join(OUT_DIR, f"{args.model}.json"))
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_meta import save_run_meta
    save_run_meta(OUT_DIR)
