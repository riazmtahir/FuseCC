"""Ensemble-design ablation: meta-learner and combination-rule
comparison on saved base-model OOF probabilities.

Sources: results/gcj_148{,_ft}/classification_enhanced/oof.npz
         (oof_cb, oof_xgb, oof_lgb, y from tools/train_enhanced_gcj_v2.py)

Compared (5-fold CV over the OOF matrix, F1 at 0.5 plus best-threshold F1):
  single best base model, soft voting (mean), LR meta, MLP(16,8) meta,
  XGB(50 trees, depth 3) meta.

Writes results/d2_meta_comparison/summary.json.
Run:  python tools/d2_meta_comparison.py
"""
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 12345
OUT = "results/d2_meta_comparison"
os.makedirs(OUT, exist_ok=True)

SOURCES = {
    "gcj_148_zeroshot": "results/gcj_148/classification_enhanced/oof.npz",
    "gcj_148_finetuned": "results/gcj_148_ft/classification_enhanced/oof.npz",
}


def f1_at(y, p, thr=0.5):
    return float(f1_score(y, (p >= thr).astype(int)))


def best_f1(y, p):
    grid = np.linspace(0.01, 0.99, 99)
    return float(max(f1_score(y, (p >= t).astype(int)) for t in grid))


def meta_oof(meta_factory, M, y):
    oof = np.zeros(len(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in cv.split(M, y):
        m = meta_factory()
        m.fit(M[tr], y[tr])
        oof[va] = m.predict_proba(M[va])[:, 1]
    return oof


summary = {}
for name, path in SOURCES.items():
    d = np.load(path)
    y = d["y"].astype(int)
    M = np.column_stack([d["oof_cb"], d["oof_xgb"], d["oof_lgb"]])
    res = {}
    singles = {k: f1_at(y, d[f"oof_{k}"]) for k in ("cb", "xgb", "lgb")}
    best_single = max(singles, key=singles.get)
    res["best_single"] = {"model": best_single, "f1@0.5": singles[best_single],
                          "all": singles}
    p_vote = M.mean(axis=1)
    res["soft_voting"] = {"f1@0.5": f1_at(y, p_vote), "best_f1": best_f1(y, p_vote)}
    metas = {
        "meta_LR": lambda: LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "meta_MLP": lambda: MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000,
                                          random_state=RANDOM_STATE, early_stopping=True,
                                          validation_fraction=0.1),
        "meta_XGB": lambda: xgb.XGBClassifier(n_estimators=50, max_depth=3,
                                              learning_rate=0.1, n_jobs=-1,
                                              random_state=RANDOM_STATE,
                                              eval_metric="logloss"),
    }
    for mname, factory in metas.items():
        p = meta_oof(factory, M, y)
        res[mname] = {"f1@0.5": f1_at(y, p), "best_f1": best_f1(y, p)}
    summary[name] = res
    print(name, json.dumps(res, indent=2), flush=True)

with open(f"{OUT}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("saved", f"{OUT}/summary.json")
