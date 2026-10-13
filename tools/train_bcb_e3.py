"""BCB retrain on canonical pairs with 19 metrics + UniXcoder features.

Inputs:
  results/bcb_19_canonical/results.csv   t1..t19 over dataset/BCB/pairs_canonical.csv
  dataset/BCB/embeddings_unixcoder.pkl   zero-shot UniXcoder, 73,135 files
  dataset/BCB/pairs_canonical.csv        canonical de-duplicated pairs + clone_type

Features (148): score_t1..t19, uxc cosine (centered), |a-b| and a*b in 64-dim PCA
space (PCA on all files — zero-shot encoder, unsupervised; labels never leak).

Protocol: mirrors tools/eval_bcb_per_type.py (XGB + LGB base -> LR meta, 5-fold StratifiedKFold OOF,
threshold 0.5) in the same two regimes:
  all  : train/eval on everything (comparability with the 16-feat 0.9742)
  t123 : WT3/T4 excluded from train+headline eval, scored separately as probe

Writes results/bcb_e3/summary.json (+ features.npz for reuse in the analysis scripts).
Run:  nohup python tools/train_bcb_e3.py > logs/train_bcb_e3.log 2>&1 &
      (--sample 14000 for a smoke test)
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/bcb_e3"
TYPES = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()

res = pd.read_csv("results/bcb_19_canonical/results.csv").dropna().reset_index(drop=True)
canon = pd.read_csv("dataset/BCB/pairs_canonical.csv")
res = res.merge(canon[["path_a", "path_b", "clone_type"]], on=["path_a", "path_b"], how="left")
assert res["clone_type"].notna().all(), "pairs missing clone_type after join"
print(f"rows {len(res)}; types {res.clone_type.value_counts().to_dict()}", flush=True)

with open("dataset/BCB/embeddings_unixcoder.pkl", "rb") as f:
    emb = pickle.load(f)
paths = sorted(emb.keys())
mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
mean_vec = mat.mean(axis=0, keepdims=True)
c = (mat - mean_vec)
c = c / np.linalg.norm(c, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=RANDOM_STATE)
pc = pca.fit_transform(c)
idx = {p: i for i, p in enumerate(paths)}

if args.sample:
    res = res.groupby("clone_type", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.sample // 7), random_state=RANDOM_STATE)
    ).reset_index(drop=True)

ia = res["path_a"].map(idx).values.astype(int)
ib = res["path_b"].map(idx).values.astype(int)
A, B = pc[ia].astype(np.float32), pc[ib].astype(np.float32)
cos = (c[ia] * c[ib]).sum(axis=1).astype(np.float32)
score_cols = [f"score_t{i}" for i in range(1, 20)]
X = np.hstack([res[score_cols].values.astype(np.float32), cos[:, None],
               np.abs(A - B), A * B])
y = res["label"].astype(int).values
print(f"X {X.shape}", flush=True)

if not args.sample:
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, "features.npz"),
        X=X, y=y.astype(np.int8),
        path_a=res["path_a"].values.astype(str), path_b=res["path_b"].values.astype(str),
        clone_type=res["clone_type"].values.astype(str),
        feature_names=np.array(score_cols + ["uxc_cos"]
                               + [f"uxc_diff_{i}" for i in range(64)]
                               + [f"uxc_prod_{i}" for i in range(64)]),
    )


def xgb_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=RANDOM_STATE, eval_metric="logloss")


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def run_regime(mask, probe_mask=None):
    Xs, ys = X[mask], y[mask]
    oof = np.zeros(mask.sum())
    probe_probs = np.zeros(probe_mask.sum()) if probe_mask is not None else None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for k, (tr, va) in enumerate(cv.split(Xs, ys)):
        xm, lm = xgb_model(), lgb_model()
        xm.fit(Xs[tr], ys[tr])
        lm.fit(Xs[tr], ys[tr])
        meta_tr = np.column_stack([xm.predict_proba(Xs[tr])[:, 1], lm.predict_proba(Xs[tr])[:, 1]])
        meta_va = np.column_stack([xm.predict_proba(Xs[va])[:, 1], lm.predict_proba(Xs[va])[:, 1]])
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(meta_tr, ys[tr])
        oof[va] = lr.predict_proba(meta_va)[:, 1]
        if probe_mask is not None:
            meta_p = np.column_stack([xm.predict_proba(X[probe_mask])[:, 1],
                                      lm.predict_proba(X[probe_mask])[:, 1]])
            probe_probs += lr.predict_proba(meta_p)[:, 1] / 5
        print(f"  fold {k+1}/5 done", flush=True)
    return oof, probe_probs


def report(frame_types, y_true, probs, thr=0.5):
    pred = (probs >= thr).astype(int)
    out = {"f1": float(f1_score(y_true, pred)), "precision": float(precision_score(y_true, pred)),
           "recall": float(recall_score(y_true, pred)), "n": int(len(y_true))}
    out["per_type_recall"] = {
        t: {"recall": float(pred[(frame_types == t)].mean()), "n": int((frame_types == t).sum())}
        for t in TYPES if (frame_types == t).any()
    }
    return out

results = {}
ct = res["clone_type"].values
X_full = X
SUBSETS = {"t16": 16, "t19": 19, "full148": X_full.shape[1]}

for sub_name, n_cols in SUBSETS.items():
    X = X_full[:, :n_cols]
    results[sub_name] = {}

    print(f"=== [{sub_name}] regime ALL ===", flush=True)
    all_mask = np.ones(len(y), dtype=bool)
    oof_all, _ = run_regime(all_mask)
    results[sub_name]["all"] = report(ct, y, oof_all)
    print(json.dumps(results[sub_name]["all"], indent=2), flush=True)

    print(f"=== [{sub_name}] regime T1-T3 (WT3/T4 probe) ===", flush=True)
    keep = ct != "WT3/T4"
    oof_sub, probe_probs = run_regime(keep, probe_mask=~keep)
    results[sub_name]["t123"] = report(ct[keep], y[keep], oof_sub)
    results[sub_name]["t123"]["wt3t4_probe"] = {
        "recall_at_0.5": float((probe_probs >= 0.5).mean()), "n": int((~keep).sum()),
        "note": "model never trained on WT3/T4",
    }
    print(json.dumps(results[sub_name]["t123"], indent=2), flush=True)

if not args.sample:
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved", os.path.join(OUT_DIR, "summary.json"))
    sys.path.insert(0, os.path.dirname(__file__))
    from run_meta import save_run_meta
    save_run_meta(OUT_DIR)
