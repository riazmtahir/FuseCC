"""Enhanced GCJ ensemble on the 148-feature matrix (20 similarity scores +
128 UniXcoder PCA interaction features).

Same protocol as tools/train_enhanced_gcj.py (5-fold StratifiedKFold OOF,
CatBoost + XGBoost + LightGBM base models, MLP meta-learner, threshold sweep)
so results are directly comparable with results/gcj_19 (F1 0.8397).
XGBoost runs on GPU.
"""
import os
import json
import argparse

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="results/gcj_148/features.npz")
parser.add_argument("--out-dir", default="results/gcj_148/classification_enhanced")
args = parser.parse_args()

OUT_DIR = args.out_dir
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 12345
N_SPLITS = 5

data = np.load(args.features, allow_pickle=True)
X, y = data["X"], data["y"].astype(int)
print(f"X shape: {X.shape}, y distribution: {np.bincount(y)}")

xgb_params = dict(n_estimators=500, max_depth=13, learning_rate=0.15,
                  subsample=0.8, colsample_bytree=1.0,
                  tree_method="hist", device="cuda",
                  random_state=RANDOM_STATE, eval_metric="logloss")
lgb_params = dict(n_estimators=700, num_leaves=255, max_depth=13, learning_rate=0.15,
                  subsample=1.0, colsample_bytree=0.7,
                  random_state=RANDOM_STATE, verbose=-1, n_jobs=-1)
cb_params = dict(iterations=300, depth=8, learning_rate=0.05,
                 random_seed=RANDOM_STATE, verbose=False, thread_count=-1)

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof = {m: np.zeros(len(y)) for m in ("cb", "xgb", "lgb")}

for fold, (tr, va) in enumerate(cv.split(X, y)):
    models = {
        "cb": cb.CatBoostClassifier(**cb_params),
        "xgb": xgb.XGBClassifier(**xgb_params),
        "lgb": lgb.LGBMClassifier(**lgb_params),
    }
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        oof[name][va] = m.predict_proba(X[va])[:, 1]
        print(f"fold {fold+1}/{N_SPLITS} {name}: "
              f"F1={f1_score(y[va], (oof[name][va] >= 0.5).astype(int)):.4f}", flush=True)

meta_X = np.column_stack([oof["cb"], oof["xgb"], oof["lgb"]])
meta_oof = np.zeros(len(y))
meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
for fold, (tr, va) in enumerate(meta_cv.split(meta_X, y)):
    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation="relu", max_iter=1000,
                        random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
    mlp.fit(meta_X[tr], y[tr])
    meta_oof[va] = mlp.predict_proba(meta_X[va])[:, 1]
    print(f"meta fold {fold+1}/5 done", flush=True)

np.savez_compressed(os.path.join(OUT_DIR, "oof.npz"),
                    oof_cb=oof["cb"], oof_xgb=oof["xgb"], oof_lgb=oof["lgb"],
                    meta_oof=meta_oof, y=y)

best = (0, 0, 0, 0)
for t in np.linspace(0.01, 0.99, 99):
    preds = (meta_oof >= t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best[1]:
        best = (t, f1, precision_score(y, preds), recall_score(y, preds))

print("=" * 60)
print(f"Best threshold: {best[0]:.3f} -> F1={best[1]:.4f}, Prec={best[2]:.4f}, Rec={best[3]:.4f}")
per_model = {}
for name in ("cb", "xgb", "lgb"):
    p = (oof[name] >= 0.5).astype(int)
    per_model[name] = {"f1": float(f1_score(y, p)),
                       "precision": float(precision_score(y, p)),
                       "recall": float(recall_score(y, p))}
    print(f"{name} @0.5: {per_model[name]}")

summary = {
    "features": int(X.shape[1]),
    "best_threshold": float(best[0]),
    "best_f1": float(best[1]),
    "best_precision": float(best[2]),
    "best_recall": float(best[3]),
    "base_models_at_0.5": per_model,
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved to", os.path.join(OUT_DIR, "summary.json"))
