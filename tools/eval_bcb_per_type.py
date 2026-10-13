"""Krinke-compliant per-type BCB evaluation.

Mirrors the protocol that produced the BCB headline (tools/fast_classification_xgb_lgb.py:
XGB + LGB base -> LR meta, 5-fold StratifiedKFold OOF, threshold 0.5) and reports:

  regime "all"  : train/eval on ALL pairs (status quo, comparable to F1 0.9742)
  regime "t123" : WT3/T4 clones EXCLUDED from training and headline eval
                  (per Krinke & Ragkhitwetsagul IWSC'22 / arXiv:2505.04311);
                  WT3/T4 pairs are scored separately as a probe set by the
                  per-fold models (never trained on) and reported with a caveat.

Per-type recall uses OOF probabilities at threshold 0.5; precision comes from
the NC pairs in the same OOF predictions (BigCloneEval convention).

Writes results/bcb_per_type/summary.json.
Run:  nohup python tools/eval_bcb_per_type.py > logs/eval_bcb_per_type.log 2>&1 &
      (--sample 20000 for a smoke test)
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
OUT_DIR = "results/bcb_per_type"
TYPES = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()

res = pd.read_csv("results/real_fusecc/results.csv").dropna().reset_index(drop=True)
fid = lambda p: int(os.path.basename(p).split(".")[0])  # noqa: E731
res["id_a"], res["id_b"] = res["path_a"].map(fid), res["path_b"].map(fid)
res["lo"] = res[["id_a", "id_b"]].min(axis=1)
res["hi"] = res[["id_a", "id_b"]].max(axis=1)

types = pd.read_csv("dataset/BCB/pairs_with_types.csv")
types["lo"] = types[["id_a", "id_b"]].min(axis=1)
types["hi"] = types[["id_a", "id_b"]].max(axis=1)
res = res.merge(types[["lo", "hi", "clone_type"]].drop_duplicates(["lo", "hi"]),
                on=["lo", "hi"], how="left")
print("type coverage:", res["clone_type"].value_counts(dropna=False).to_dict(), flush=True)
assert res.loc[res.label == 1, "clone_type"].notna().all(), "unmapped clone rows"

if args.sample:
    res = res.groupby("clone_type", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.sample // 7), random_state=RANDOM_STATE)
    ).reset_index(drop=True)

feat_cols = [c for c in res.columns if c.startswith("score_")]


def xgb_model():
    return xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=-1)


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=200, num_leaves=31, max_depth=7, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)


def run_regime(frame, probe=None):
    """5-fold OOF stacking on `frame`; optionally score `probe` rows with the
    per-fold stacks (averaged). Returns (oof_probs, probe_probs)."""
    X, y = frame[feat_cols].values, frame["label"].astype(int).values
    Xp = probe[feat_cols].values if probe is not None else None
    oof = np.zeros(len(y))
    probe_probs = np.zeros(len(probe)) if probe is not None else None
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for k, (tr, va) in enumerate(cv.split(X, y)):
        xm, lm = xgb_model(), lgb_model()
        xm.fit(X[tr], y[tr])
        lm.fit(X[tr], y[tr])
        meta_tr = np.column_stack([xm.predict_proba(X[tr])[:, 1], lm.predict_proba(X[tr])[:, 1]])
        meta_va = np.column_stack([xm.predict_proba(X[va])[:, 1], lm.predict_proba(X[va])[:, 1]])
        lr = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)
        lr.fit(meta_tr, y[tr])
        oof[va] = lr.predict_proba(meta_va)[:, 1]
        if probe is not None:
            meta_p = np.column_stack([xm.predict_proba(Xp)[:, 1], lm.predict_proba(Xp)[:, 1]])
            probe_probs += lr.predict_proba(meta_p)[:, 1] / 5
        print(f"  fold {k+1}/5 done", flush=True)
    return oof, probe_probs


def report(frame, probs, thr=0.5):
    y = frame["label"].astype(int).values
    pred = (probs >= thr).astype(int)
    out = {"f1": float(f1_score(y, pred)), "precision": float(precision_score(y, pred)),
           "recall": float(recall_score(y, pred)), "n": int(len(y))}
    out["per_type_recall"] = {
        t: {"recall": float(pred[(frame.clone_type == t).values].mean()),
            "n": int((frame.clone_type == t).sum())}
        for t in TYPES if (frame.clone_type == t).any()
    }
    return out

results = {}

print("=== regime ALL (status quo, incl. WT3/T4) ===", flush=True)
oof_all, _ = run_regime(res)
results["all"] = report(res, oof_all)
print(json.dumps(results["all"], indent=2), flush=True)

print("=== regime T1-T3 (WT3/T4 excluded; scored as probe) ===", flush=True)
keep = res["clone_type"] != "WT3/T4"
sub, probe = res[keep].reset_index(drop=True), res[~keep].reset_index(drop=True)
oof_sub, probe_probs = run_regime(sub, probe)
results["t123"] = report(sub, oof_sub)
results["t123"]["wt3t4_probe"] = {
    "recall_at_0.5": float((probe_probs >= 0.5).mean()), "n": int(len(probe)),
    "note": "model never trained on WT3/T4; reported separately per Krinke critique",
}
print(json.dumps(results["t123"], indent=2), flush=True)

if not args.sample:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("saved", os.path.join(OUT_DIR, "summary.json"))
