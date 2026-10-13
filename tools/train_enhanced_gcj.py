import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# ---------- CONFIG ----------
OUT_DIR = 'results/gcj_19/classification_enhanced'
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 12345
N_SPLITS = 5

# Load data
print("Loading features (19 metrics)...")
df = pd.read_csv('results/gcj_19/results.csv').dropna()
y = df['label'].astype(int).values
feat_cols = [c for c in df.columns if c.startswith('score_')]
X = df[feat_cols].values
print(f"X shape: {X.shape}, y distribution: {np.bincount(y)}")

cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ---------- Base Models ----------
# CatBoost (light tuning)
cb_model = cb.CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    random_seed=RANDOM_STATE,
    verbose=False,
    thread_count=-1
)

# XGBoost (best parameters from your tuning)
xgb_params = {
    'n_estimators': 500,
    'max_depth': 13,
    'learning_rate': 0.15,
    'subsample': 0.8,
    'colsample_bytree': 1.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'n_jobs': -1
}
xgb_model = xgb.XGBClassifier(**xgb_params)

# LightGBM (best parameters from your tuning)
lgb_params = {
    'n_estimators': 700,
    'num_leaves': 255,
    'max_depth': 13,
    'learning_rate': 0.15,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'random_state': RANDOM_STATE,
    'verbose': -1,
    'n_jobs': -1
}
lgb_model = lgb.LGBMClassifier(**lgb_params)

# ---------- OOF Predictions ----------
print("\n=== Generating OOF for base models ===")
oof_probs_cb = np.zeros(len(y))
oof_probs_xgb = np.zeros(len(y))
oof_probs_lgb = np.zeros(len(y))

for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = y[train_idx]

    cb_fold = cb.CatBoostClassifier(iterations=300, depth=8, learning_rate=0.05,
                                    random_seed=RANDOM_STATE, verbose=False, thread_count=-1)
    xgb_fold = xgb.XGBClassifier(**xgb_params)
    lgb_fold = lgb.LGBMClassifier(**lgb_params)

    cb_fold.fit(X_train, y_train)
    xgb_fold.fit(X_train, y_train)
    lgb_fold.fit(X_train, y_train)

    oof_probs_cb[val_idx] = cb_fold.predict_proba(X_val)[:, 1]
    oof_probs_xgb[val_idx] = xgb_fold.predict_proba(X_val)[:, 1]
    oof_probs_lgb[val_idx] = lgb_fold.predict_proba(X_val)[:, 1]

    print(f"Fold {fold_idx+1}/{N_SPLITS} complete")

# Save base OOFs
oof_df = pd.DataFrame({
    'id': range(len(y)),
    'oof_cb': oof_probs_cb,
    'oof_xgb': oof_probs_xgb,
    'oof_lgb': oof_probs_lgb,
    'label': y
})
oof_df.to_csv(os.path.join(OUT_DIR, 'oof_base_predictions.csv'), index=False)

# ---------- Meta‑Learner: MLP ----------
print("\n=== Training MLP meta‑learner on OOF ===")
meta_X = np.column_stack([oof_probs_cb, oof_probs_xgb, oof_probs_lgb])
meta_y = y

meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
meta_oof = np.zeros(len(y))
for fold_idx, (train_idx, val_idx) in enumerate(meta_cv.split(meta_X, meta_y)):
    meta_X_train, meta_X_val = meta_X[train_idx], meta_X[val_idx]
    meta_y_train = meta_y[train_idx]

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', max_iter=1000,
                        random_state=RANDOM_STATE, early_stopping=True, validation_fraction=0.1)
    mlp.fit(meta_X_train, meta_y_train)
    meta_oof[val_idx] = mlp.predict_proba(meta_X_val)[:, 1]
    print(f"Meta fold {fold_idx+1}/5 complete")

# Save full OOF
oof_df['meta_oof'] = meta_oof
oof_df.to_csv(os.path.join(OUT_DIR, 'oof_full.csv'), index=False)

# ---------- Final Evaluation ----------
best = (0,0,0,0)
for t in np.linspace(0.01,0.99,99):
    preds = (meta_oof >= t).astype(int)
    f1 = f1_score(meta_y, preds)
    pr = precision_score(meta_y, preds)
    re = recall_score(meta_y, preds)
    if f1 > best[1]:
        best = (t, f1, pr, re)

print("\n" + "="*60)
print("ENHANCED GCJ RESULTS (MLP meta on 3 base models)")
print("="*60)
print(f"Best threshold: {best[0]:.3f} → F1={best[1]:.4f}, Prec={best[2]:.4f}, Rec={best[3]:.4f}")

summary = {
    'best_threshold': float(best[0]),
    'best_f1': float(best[1]),
    'best_precision': float(best[2]),
    'best_recall': float(best[3])
}
with open(os.path.join(OUT_DIR, 'gcj_enhanced_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print("\n✅ Summary saved to", os.path.join(OUT_DIR, 'gcj_enhanced_summary.json'))
