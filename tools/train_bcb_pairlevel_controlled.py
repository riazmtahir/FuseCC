"""MAJ-1 follow-up, item 2: the pair-level BCB protocol re-run with the controls the paper claims.

WHY THIS EXISTS
The published pair-level headline (results/bcb_e3, F1 0.9986) was produced with two properties
the manuscript's threats section does not admit to:
  * the centring vector and the PCA-64 basis were fitted on ALL 73,135 embedded files, so test
    geometry informs the projection (train_bcb_e3.py lines 49-58);
  * the decision threshold was fixed at 0.5 on the out-of-fold predictions, never selected on a
    held-out split (train_bcb_e3.py report(..., thr=0.5)).
That makes 0.9986 unreportable as it stands. This script re-runs the SAME pair-level protocol
(random split over pairs, so functions recur across train and test) with both corrections, and
therefore differs from results/bcb_function_disjoint in exactly one variable: the split.

    results/bcb_function_disjoint   function-disjoint 60/20/20, train-only PCA, threshold on val
    this script                     PAIR-level     60/20/20, train-only PCA, threshold on val
                                                   ^^^^^^^^ the only difference

Everything else is held identical to train_bcb_function_disjoint.py and is quoted from it
verbatim: the 148-feature vector [t1..t19 | UniXcoder cosine | |a-b| PCA64 | a*b PCA64], the XGB
and LGB hyper-parameters, the LR meta fitted on inner 5-fold OOF over the training split (so the
validation split only ever picks the threshold), the T1-T3 regime and seed 12345.

The split is a StratifiedKFold(5, shuffle, seed 12345) over pairs, with folds 0-2 as train, fold
3 as validation and fold 4 as test. That is the original pair-level sampling scheme (functions
recur freely) reshaped into the 60/20/20 geometry of the function-disjoint run so the two are
directly comparable.

SUPPLEMENTARY, clearly labelled: the function-disjoint split has to discard 51.8% of pairs as
cross-partition, so its training set is 185,644 pairs against 263,392 here. A second variant
subsamples this training set to exactly 185,644 pairs, removing training-set size as a
confound in the split comparison.

Writes results/bcb_leakage_isolation/pairlevel_controlled.json. Does not touch results/bcb_e3
or results/bcb_function_disjoint.

Run: nohup python tools/train_bcb_pairlevel_controlled.py > logs/bcb_pairlevel_controlled.log 2>&1 &
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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/bcb_leakage_isolation"
TYPES = ["T1", "T2", "VST3", "ST3", "MT3"]          # T1-T3 regime; WT3/T4 excluded
SCORE_COLS = [f"score_t{i}" for i in range(1, 20)]
FD_TRAIN_PAIRS = 185644                             # training-set size of the fd run

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=0, help="smoke-test on N pairs")
args = ap.parse_args()
t_start = time.time()


def log(m):
    print(f"[{time.time()-t_start:7.1f}s] {m}", flush=True)


# ---------------------------------------------------------------- data (identical to the fd run)
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

# ---------------------------------------------------------------- PAIR-LEVEL split (the one change)
y_tmp = res.label.values.astype(int)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold = np.empty(len(res), dtype=np.int8)
for k, (_, va) in enumerate(skf.split(np.zeros(len(res)), y_tmp)):
    fold[va] = k
split = np.where(fold <= 2, 0, np.where(fold == 3, 1, 2)).astype(np.int8)   # 0 train, 1 val, 2 test
log("pair-level split: StratifiedKFold(5, seed 12345) over PAIRS -> folds 0-2 train / 3 val / 4 test")

# how badly do functions recur across this split? (the property the fd run removes)
fn_tr = set(res.id_a.values[split == 0]) | set(res.id_b.values[split == 0])
fn_te = set(res.id_a.values[split == 2]) | set(res.id_b.values[split == 2])
te_pairs = res[split == 2]
both_seen = np.isin(te_pairs.id_a.values, list(fn_tr)) & np.isin(te_pairs.id_b.values, list(fn_tr))
recurrence = {
    "test_functions": len(fn_te),
    "test_functions_also_in_train": len(fn_te & fn_tr),
    "test_functions_also_in_train_pct": round(100 * len(fn_te & fn_tr) / max(len(fn_te), 1), 2),
    "test_pairs_with_BOTH_endpoints_seen_in_train": int(both_seen.sum()),
    "test_pairs_with_BOTH_endpoints_seen_in_train_pct": round(100 * float(both_seen.mean()), 2),
    "note": "this is the leakage the function-disjoint split removes by construction; in the "
            "function-disjoint run all four of these numbers are 0",
}
log(f"function recurrence: {recurrence['test_functions_also_in_train_pct']}% of test functions "
    f"also appear in train; {recurrence['test_pairs_with_BOTH_endpoints_seen_in_train_pct']}% of "
    f"test pairs have BOTH endpoints seen in training")

split_stats = {}
for k, name in [(0, "train"), (1, "val"), (2, "test")]:
    m = split == k
    lab = res.label.values[m]
    split_stats[name] = {
        "n_pairs": int(m.sum()),
        "n_functions": int(len(set(res.id_a.values[m]) | set(res.id_b.values[m]))),
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

# CORRECTION 1: centring vector and PCA basis fitted on TRAINING pairs' functions only
train_paths = set(res.path_a.values[split == 0].tolist()) | set(res.path_b.values[split == 0].tolist())
train_rows = np.array(sorted({pidx[p] for p in train_paths if p in pidx}))
log(f"fitting centring vector + PCA on {len(train_rows):,} TRAINING function embeddings only "
    f"(bcb_e3 fitted on all {len(paths):,})")

mean_vec = mat[train_rows].mean(axis=0, keepdims=True)
cen = mat - mean_vec
cen /= np.linalg.norm(cen, axis=1, keepdims=True)
pca = PCA(n_components=64, random_state=RANDOM_STATE).fit(cen[train_rows])
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
log(f"X {X.shape} (148 features, identical layout to bcb_e3 and the fd run)")

tr, va, te = split == 0, split == 1, split == 2


# ---------------------------------------------------------------- models (identical to the fd run)
def xgb_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=RANDOM_STATE, eval_metric="logloss")


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def pick_threshold(y_true, prob):
    grid = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def report(y_true, prob, thr, types=None):
    pred = (prob >= thr).astype(int)
    out = {"threshold": round(float(thr), 4),
           "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
           "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
           "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
           "auc": round(float(roc_auc_score(y_true, prob)), 4),
           "n": int(len(y_true)),
           "clone_rate": round(float(np.mean(y_true)), 4)}
    if types is not None:
        out["per_type_recall"] = {
            t: {"recall": round(float(pred[(types == t) & (y_true == 1)].mean()), 4),
                "n": int(((types == t) & (y_true == 1)).sum())}
            for t in TYPES if ((types == t) & (y_true == 1)).any()}
    return out


def fit_and_score(train_idx, tag):
    """Fit XGB+LGB (+LR meta on inner 5-fold OOF over train) and score val/test."""
    log(f"[{tag}] fitting XGB and LGB on {len(train_idx):,} training pairs")
    Xtr, ytr = X[train_idx], y[train_idx]
    xm, lm = xgb_model(), lgb_model()
    xm.fit(Xtr, ytr)
    lm.fit(Xtr, ytr)
    p_xgb = {s: xm.predict_proba(X[m])[:, 1] for s, m in [("val", va), ("test", te)]}
    p_lgb = {s: lm.predict_proba(X[m])[:, 1] for s, m in [("val", va), ("test", te)]}

    log(f"[{tag}] building inner 5-fold OOF meta-features on the training split")
    oof = np.zeros((len(train_idx), 2))
    for k, (i_tr, i_va) in enumerate(StratifiedKFold(5, shuffle=True,
                                                     random_state=RANDOM_STATE).split(Xtr, ytr)):
        xk, lk = xgb_model(), lgb_model()
        xk.fit(Xtr[i_tr], ytr[i_tr])
        lk.fit(Xtr[i_tr], ytr[i_tr])
        oof[i_va, 0] = xk.predict_proba(Xtr[i_va])[:, 1]
        oof[i_va, 1] = lk.predict_proba(Xtr[i_va])[:, 1]
        log(f"  [{tag}] inner fold {k+1}/5")
    meta = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE).fit(oof, ytr)
    p_stack = {s: meta.predict_proba(np.column_stack([p_xgb[s], p_lgb[s]]))[:, 1]
               for s in ("val", "test")}

    out = {}
    for name, p in [("xgb_148", p_xgb), ("stacked_xgb_lgb_lr", p_stack)]:
        # CORRECTION 2: threshold selected on validation, applied unchanged to test
        thr = pick_threshold(y[va], p["val"])
        out[name] = {
            "val": report(y[va], p["val"], thr),
            "test": report(y[te], p["test"], thr, ct[te]),
            "test_at_0.5": report(y[te], p["test"], 0.5, ct[te]),
        }
        log(f"[{tag}] {name}: thr(val)={thr:.3f}  TEST F1 {out[name]['test']['f1']:.4f} "
            f"AUC {out[name]['test']['auc']:.4f}")
    return out


tr_idx = np.flatnonzero(tr)
results = {"primary": fit_and_score(tr_idx, "primary")}

# supplementary: match the fd run's training-set size so size is not a confound
if not args.sample and len(tr_idx) > FD_TRAIN_PAIRS:
    rs = np.random.RandomState(RANDOM_STATE)
    sub = np.sort(rs.choice(tr_idx, FD_TRAIN_PAIRS, replace=False))
    log(f"supplementary variant: training set subsampled {len(tr_idx):,} -> {len(sub):,} pairs "
        f"to match the function-disjoint run")
    results["train_size_matched_to_fd"] = fit_and_score(sub, "size-matched")

# ---------------------------------------------------------------- comparison
fd_sum = json.load(open("results/bcb_function_disjoint/summary.json"))
e3 = json.load(open("results/bcb_e3/summary.json"))["full148"]["t123"]
fd_stk = fd_sum["results"]["stacked_xgb_lgb_lr"]["test"]
ctrl_stk = results["primary"]["stacked_xgb_lgb_lr"]["test"]

summary = {
    "purpose": "MAJ-1 item 2: pair-level BCB protocol re-run with train-only PCA/centring and a "
               "validation-selected threshold, so it differs from the function-disjoint run in "
               "exactly one variable (the split)",
    "protocol": {
        "split": "StratifiedKFold(5, shuffle, seed 12345) over PAIRS; folds 0-2 train, fold 3 "
                 "validation, fold 4 test. Functions recur freely across splits.",
        "seed": RANDOM_STATE,
        "regime": "T1-T3 (WT3/T4 excluded, per Krinke)",
        "encoder_stats": "CORRECTED: centring vector and PCA-64 basis fitted on the functions of "
                         "TRAINING pairs only (bcb_e3 fitted on all 73,135 files)",
        "threshold": "CORRECTED: best F1 on the validation split, applied unchanged to test "
                     "(bcb_e3 fixed 0.5 on out-of-fold predictions)",
        "features": "148 = t1..t19 + UniXcoder cosine + |a-b| PCA64 + a*b PCA64 (identical)",
        "models": "XGB + LGB (bcb_e3 hyper-parameters) -> LR meta on inner 5-fold OOF over train",
    },
    "splits": split_stats,
    "function_recurrence_across_the_split": recurrence,
    "results": results,
    "comparison": {
        "controlled_pair_level_stacked_test": ctrl_stk,
        "function_disjoint_stacked_test": {**fd_stk, "source": "results/bcb_function_disjoint"},
        "uncontrolled_bcb_e3_headline": {
            "f1": round(e3["f1"], 4), "precision": round(e3["precision"], 4),
            "recall": round(e3["recall"], 4), "n": e3["n"],
            "source": "results/bcb_e3/summary.json full148.t123",
            "why_not_reportable": "all-files PCA/centring and a fixed 0.5 threshold",
        },
        "controlled_minus_function_disjoint": {
            "f1": round(ctrl_stk["f1"] - fd_stk["f1"], 4),
            "precision": round(ctrl_stk["precision"] - fd_stk["precision"], 4),
            "recall": round(ctrl_stk["recall"] - fd_stk["recall"], 4),
            "auc": round(ctrl_stk["auc"] - fd_stk["auc"], 4),
            "note": "the split is the only protocol difference between these two runs",
        },
        "controlled_minus_uncontrolled": {
            "f1": round(ctrl_stk["f1"] - round(e3["f1"], 4), 4),
            "note": "cost of applying the two controls at the same (pair-level) split",
        },
    },
}

if not args.sample:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "pairlevel_controlled.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote {OUT_DIR}/pairlevel_controlled.json")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_meta import save_run_meta  # noqa: E402
    prior = {}
    rm_path = os.path.join(OUT_DIR, "run_meta.json")
    if os.path.exists(rm_path):
        prior = json.load(open(rm_path))
    save_run_meta(OUT_DIR, extra={
        "component": "pairlevel_controlled (item 2)",
        "outputs": ["pairlevel_controlled.json"],
        "prior_runs": prior.get("prior_runs", []) +
                      ([{k: prior[k] for k in ("component", "timestamp_utc", "argv") if k in prior}]
                       if prior else [])})
    log(f"wrote {rm_path}")
else:
    log("SMOKE TEST - nothing written")
print(json.dumps(summary, indent=2), flush=True)
log("done")
