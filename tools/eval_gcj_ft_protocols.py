"""Evaluate the fine-tuned-encoder feature set (results/gcj_148_ft) under both
protocols:

1. random-split : 3-fold StratifiedKFold over pairs, XGBoost GPU.
   Comparable to the previous quick check on gcj_148 (F1 0.9433).
2. file-disjoint: test = pairs whose BOTH files are in the encoder's held-out
   set (models/unixcoder_gcj_ft/val_files.txt) — the encoder, PCA and
   classifier have never seen these files. Decision threshold is chosen on a
   file-disjoint validation split carved out of the training files (B2), then
   applied unchanged to the test pairs.
   Comparable to rep 0 of the previous file-disjoint check (F1 0.8486).

Writes results/gcj_148_ft/protocol_eval.json.

Run:  nohup python tools/eval_gcj_ft_protocols.py > logs/eval_gcj_ft.log 2>&1 &
      (add --sample 1000 for a smoke test)
"""
import argparse
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 12345
FEATURES = "results/gcj_148_ft/features.npz"
VAL_FILES = "models/unixcoder_gcj_ft/val_files.txt"
OUT = "results/gcj_148_ft/protocol_eval.json"

parser = argparse.ArgumentParser()
parser.add_argument("--sample", type=int, default=0, help="subsample pairs for smoke test")
args = parser.parse_args()

data = np.load(FEATURES, allow_pickle=True)
X, y = data["X"], data["y"].astype(int)
path_a, path_b = data["path_a"], data["path_b"]
if args.sample:
    rng = np.random.RandomState(RANDOM_STATE)
    keep = rng.choice(len(y), size=args.sample, replace=False)
    X, y, path_a, path_b = X[keep], y[keep], path_a[keep], path_b[keep]
print(f"X {X.shape}, positives {y.mean():.3f}")

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


def best_thr(y_true, proba):
    grid = np.linspace(0.01, 0.99, 99)
    f1s = [f1_score(y_true, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(f1s))])


results = {}

# ---- Protocol 1: random split (3-fold stratified) -------------------------
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
oof = np.zeros(len(y))
for fold, (tr, va) in enumerate(cv.split(X, y)):
    m = xgb.XGBClassifier(**xgb_params)
    m.fit(X[tr], y[tr])
    oof[va] = m.predict_proba(X[va])[:, 1]
    print(f"random-split fold {fold+1}/3 "
          f"F1@0.5={f1_score(y[va], (oof[va] >= 0.5).astype(int)):.4f}", flush=True)
results["random_split"] = metrics(y, oof, 0.5)
print("random-split:", results["random_split"], flush=True)

# ---- Protocol 2: file-disjoint, threshold picked on validation ------------
with open(VAL_FILES) as f:
    test_files = set(line.strip() for line in f if line.strip())
in_test = np.isin(path_a, list(test_files)) | np.isin(path_b, list(test_files))
both_test = np.isin(path_a, list(test_files)) & np.isin(path_b, list(test_files))

train_files = sorted((set(path_a) | set(path_b)) - test_files)
rng = np.random.RandomState(RANDOM_STATE)
thr_files = set(rng.choice(train_files, size=int(0.15 * len(train_files)), replace=False))
a_thr = np.isin(path_a, list(thr_files))
b_thr = np.isin(path_b, list(thr_files))

fit_mask = ~in_test & ~a_thr & ~b_thr          # pairs fully inside classifier-train files
thr_mask = ~in_test & a_thr & b_thr            # pairs fully inside threshold-val files
print(f"file-disjoint: fit {fit_mask.sum()}, thr-val {thr_mask.sum()}, "
      f"test {both_test.sum()} pairs", flush=True)

m = xgb.XGBClassifier(**xgb_params)
m.fit(X[fit_mask], y[fit_mask])
p_thr = m.predict_proba(X[thr_mask])[:, 1]
t_star = best_thr(y[thr_mask], p_thr)
p_test = m.predict_proba(X[both_test])[:, 1]
results["file_disjoint"] = {
    "n_fit": int(fit_mask.sum()), "n_thr_val": int(thr_mask.sum()),
    "n_test": int(both_test.sum()),
    "at_0.5": metrics(y[both_test], p_test, 0.5),
    "at_val_threshold": metrics(y[both_test], p_test, t_star),
}
print("file-disjoint @0.5:", results["file_disjoint"]["at_0.5"], flush=True)
print(f"file-disjoint @t*={t_star:.2f} (picked on val):",
      results["file_disjoint"]["at_val_threshold"], flush=True)

if not args.sample:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", OUT)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_meta import save_run_meta
    save_run_meta(os.path.dirname(OUT))
