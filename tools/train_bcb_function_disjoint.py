"""MAJ-1: function-disjoint BCB split for the T1-T3 headline.

WHY THIS EXISTS
`tools/train_bcb_e3.py` evaluates BCB with StratifiedKFold over PAIRS (its line ~105), so a
function may appear in training and test folds simultaneously. That is standard BCB practice
and it is what produced the headline F1 0.9986, but it is the only benchmark in the paper
without a disjointness control, and the threats section does not disclose it
(FuseCC_Verification_Audit.md, MAJ-1). This script answers the question the audit raises:
what is the number when functions cannot recur across splits?

WHAT IS DIFFERENT FROM bcb_e3 (everything else is held identical)
1. Split: function IDs are partitioned 60/20/20, then a pair is KEPT only if BOTH endpoints
   fall in the same partition. Cross-partition pairs are discarded and counted.
2. Encoder statistics: the centring vector and the PCA basis are fitted on TRAINING functions
   only. bcb_e3 fits them on all 73,135 files, which is unsupervised but still lets test-set
   geometry inform the projection.
3. Threshold: selected on the validation split by best F1, then applied unchanged to test.
   bcb_e3 uses a fixed 0.5 on out-of-fold predictions.
Feature vector, models and hyper-parameters are byte-identical to bcb_e3: 148 features
[t1..t19 | UniXcoder cosine | |a-b| PCA64 | a*b PCA64], XGB + LGB base -> LR meta.

The meta-learner is fitted on inner 5-fold out-of-fold predictions over the TRAINING split, so
the validation split is reserved purely for threshold selection and never trains anything.

Writes results/bcb_function_disjoint/{summary.json,run_meta.json}. Does not touch results/bcb_e3.

Run: nohup python tools/train_bcb_function_disjoint.py > logs/bcb_function_disjoint.log 2>&1 &
     --sample N   smoke test on N pairs
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/bcb_function_disjoint"
TYPES = ["T1", "T2", "VST3", "ST3", "MT3"]          # T1-T3 regime; WT3/T4 excluded
PROPS = (0.60, 0.20, 0.20)
SCORE_COLS = [f"score_t{i}" for i in range(1, 20)]

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=0, help="smoke-test on N pairs")
args = ap.parse_args()
t_start = time.time()


def log(m):
    print(f"[{time.time()-t_start:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- data
log("loading canonical pairs")
canon = pd.read_csv("dataset/BCB/pairs_canonical.csv",
                    usecols=["id_a", "id_b", "label", "path_a", "path_b", "clone_type"])
canon = canon[canon.clone_type != "WT3/T4"].reset_index(drop=True)
log(f"T1-T3 canonical pairs: {len(canon):,}")

log("loading t1-t19 scores")
res = pd.read_csv("results/bcb_19_canonical/results.csv",
                  usecols=["path_a", "path_b", "label"] + SCORE_COLS).dropna()
res = res.merge(canon[["path_a", "path_b", "id_a", "id_b", "clone_type"]],
                on=["path_a", "path_b"], how="inner")
log(f"pairs with features: {len(res):,}")

if args.sample:
    res = res.sample(min(args.sample, len(res)), random_state=RANDOM_STATE).reset_index(drop=True)
    log(f"SMOKE TEST on {len(res):,} pairs")

# ---------------------------------------------------------------- function-disjoint split
fn_ids = pd.unique(np.concatenate([res.id_a.values, res.id_b.values]))
rng = np.random.default_rng(RANDOM_STATE)
perm = rng.permutation(len(fn_ids))
n_tr = int(PROPS[0] * len(fn_ids))
n_va = n_tr + int(PROPS[1] * len(fn_ids))
part = np.empty(len(fn_ids), dtype=np.int8)
part[perm[:n_tr]] = 0
part[perm[n_tr:n_va]] = 1
part[perm[n_va:]] = 2
fn_part = dict(zip(fn_ids, part))
log(f"functions partitioned: train {int((part==0).sum()):,} / val {int((part==1).sum()):,} "
    f"/ test {int((part==2).sum()):,} (total {len(fn_ids):,})")

pa = res.id_a.map(fn_part).values
pb = res.id_b.map(fn_part).values
keep = pa == pb                      # both endpoints in the same partition
n_disc = int((~keep).sum())
log(f"pairs kept {int(keep.sum()):,}/{len(res):,} = {keep.mean():.2%} | "
    f"discarded {n_disc:,} = {(~keep).mean():.2%} (cross-partition)")

res = res[keep].reset_index(drop=True)
split = pa[keep]                     # 0 train, 1 val, 2 test

split_stats = {}
for k, name in [(0, "train"), (1, "val"), (2, "test")]:
    m = split == k
    lab = res.label.values[m]
    split_stats[name] = {
        "n_functions": int((part == k).sum()),
        "n_pairs": int(m.sum()),
        "n_clone": int(lab.sum()),
        "n_nonclone": int((lab == 0).sum()),
        "clone_rate": round(float(lab.mean()), 4),
    }
    log(f"  {name:5s}: {m.sum():7,} pairs, clone rate {lab.mean():.2%}")

# ---------------------------------------------------------------- encoder features (train-only fit)
log("loading UniXcoder embeddings")
with open("dataset/BCB/embeddings_unixcoder.pkl", "rb") as f:
    emb = pickle.load(f)

paths = sorted(emb.keys())
pidx = {p: i for i, p in enumerate(paths)}
mat = np.vstack([emb[p] for p in paths]).astype(np.float64)
del emb

# which embedding rows belong to TRAINING functions
path_of_fn = {}
for c_id, c_path in [("id_a", "path_a"), ("id_b", "path_b")]:
    for fid, fp in zip(res[c_id].values, res[c_path].values):
        path_of_fn.setdefault(fid, fp)
train_rows = np.array(sorted({pidx[path_of_fn[f]] for f in fn_ids
                              if fn_part[f] == 0 and f in path_of_fn and path_of_fn[f] in pidx}))
log(f"fitting centring vector + PCA on {len(train_rows):,} TRAINING function embeddings only")

mean_vec = mat[train_rows].mean(axis=0, keepdims=True)          # train-only centring
cen = mat - mean_vec
cen /= np.linalg.norm(cen, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=RANDOM_STATE).fit(cen[train_rows])   # train-only basis
pc = pca.transform(cen).astype(np.float32)
log(f"PCA explained variance (train fit): {pca.explained_variance_ratio_.sum():.4f}")

ia = res.path_a.map(pidx).values.astype(int)
ib = res.path_b.map(pidx).values.astype(int)
A, B = pc[ia], pc[ib]
cos = (cen[ia] * cen[ib]).sum(axis=1).astype(np.float32)
X = np.hstack([res[SCORE_COLS].values.astype(np.float32), cos[:, None],
               np.abs(A - B), A * B]).astype(np.float32)
y = res.label.values.astype(int)
ct = res.clone_type.values
del mat, cen, pc, A, B
log(f"X {X.shape} (148 features, identical layout to bcb_e3)")

tr, va, te = split == 0, split == 1, split == 2


# ---------------------------------------------------------------- models (identical to bcb_e3)
def xgb_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=RANDOM_STATE, eval_metric="logloss")


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


log("fitting XGB and LGB on the training split")
xm, lm = xgb_model(), lgb_model()
xm.fit(X[tr], y[tr])
lm.fit(X[tr], y[tr])

p_xgb = {s: xm.predict_proba(X[m])[:, 1] for s, m in [("val", va), ("test", te)]}
p_lgb = {s: lm.predict_proba(X[m])[:, 1] for s, m in [("val", va), ("test", te)]}

# meta-learner on inner OOF over TRAIN only, so val stays reserved for the threshold
log("building inner 5-fold OOF meta-features on the training split")
oof = np.zeros((int(tr.sum()), 2))
Xtr, ytr = X[tr], y[tr]
for k, (i_tr, i_va) in enumerate(StratifiedKFold(5, shuffle=True,
                                                 random_state=RANDOM_STATE).split(Xtr, ytr)):
    xk, lk = xgb_model(), lgb_model()
    xk.fit(Xtr[i_tr], ytr[i_tr])
    lk.fit(Xtr[i_tr], ytr[i_tr])
    oof[i_va, 0] = xk.predict_proba(Xtr[i_va])[:, 1]
    oof[i_va, 1] = lk.predict_proba(Xtr[i_va])[:, 1]
    log(f"  inner fold {k+1}/5")
meta = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(oof, ytr)
p_stack = {s: meta.predict_proba(np.column_stack([p_xgb[s], p_lgb[s]]))[:, 1]
           for s in ("val", "test")}


# ---------------------------------------------------------------- threshold on val, report on test
def pick_threshold(y_true, prob):
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def report(y_true, prob, thr, types=None):
    pred = (prob >= thr).astype(int)
    out = {"threshold": round(thr, 4),
           "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
           "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
           "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
           "auc": round(float(roc_auc_score(y_true, prob)), 4),
           "n": int(len(y_true))}
    if types is not None:
        out["per_type_recall"] = {
            t: {"recall": round(float(pred[(types == t) & (y_true == 1)].mean()), 4),
                "n": int(((types == t) & (y_true == 1)).sum())}
            for t in TYPES if ((types == t) & (y_true == 1)).any()}
    return out


results = {}
for name, pv, pt in [("xgb_148", p_xgb, p_xgb), ("stacked_xgb_lgb_lr", p_stack, p_stack)]:
    thr = pick_threshold(y[va], pv["val"])
    results[name] = {
        "val": report(y[va], pv["val"], thr),
        "test": report(y[te], pt["test"], thr, ct[te]),
        "test_at_0.5": report(y[te], pt["test"], 0.5, ct[te]),
    }
    log(f"{name}: thr(val)={thr:.3f}  TEST F1 {results[name]['test']['f1']:.4f} "
        f"AUC {results[name]['test']['auc']:.4f}")

# ---------------------------------------------------------------- delta vs the pair-level result
pair_level = None
try:
    e3 = json.load(open("results/bcb_e3/summary.json"))["full148"]["t123"]
    pair_level = {"f1": round(e3["f1"], 4), "precision": round(e3["precision"], 4),
                  "recall": round(e3["recall"], 4), "n": e3["n"],
                  "source": "results/bcb_e3/summary.json full148.t123 (pair-level 5-fold OOF, thr 0.5)"}
except Exception as exc:                                    # pragma: no cover
    log(f"could not read bcb_e3 for the delta: {exc}")

summary = {
    "purpose": "MAJ-1: function-disjoint BCB T1-T3 evaluation (audit FuseCC_Verification_Audit.md)",
    "protocol": {
        "split": "function IDs partitioned 60/20/20; a pair is kept only if BOTH endpoints are "
                 "in the same partition, so no function occurs in more than one split",
        "seed": RANDOM_STATE,
        "regime": "T1-T3 (WT3/T4 excluded, per Krinke)",
        "encoder_stats": "centring vector and PCA-64 basis fitted on TRAINING functions only",
        "threshold": "best F1 on the validation split, applied unchanged to test",
        "features": "148 = t1..t19 + UniXcoder cosine + |a-b| PCA64 + a*b PCA64 (identical to bcb_e3)",
        "models": "XGB + LGB (bcb_e3 hyper-parameters) -> LR meta on inner 5-fold OOF over train",
    },
    "splits": split_stats,
    "pairs": {
        "candidate_T1_T3_pairs": int(len(keep)),
        "kept": int(keep.sum()),
        "discarded_cross_partition": n_disc,
        "discarded_fraction": round(float((~keep).mean()), 4),
    },
    "results": results,
    "pair_level_reference": pair_level,
}
if pair_level:
    for name in results:
        summary["results"][name]["delta_vs_pair_level"] = {
            "f1": round(results[name]["test"]["f1"] - pair_level["f1"], 4),
            "precision": round(results[name]["test"]["precision"] - pair_level["precision"], 4),
            "recall": round(results[name]["test"]["recall"] - pair_level["recall"], 4),
        }

if not args.sample:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {OUT_DIR}/summary.json")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_meta import save_run_meta
    save_run_meta(OUT_DIR)
    log(f"wrote {OUT_DIR}/run_meta.json")
else:
    log("SMOKE TEST — nothing written")
print(json.dumps(summary, indent=2), flush=True)
log("done")
