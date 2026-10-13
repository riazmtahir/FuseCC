"""PROBLEM-DISJOINT evaluation of the pd-fine-tuned encoder features.

Split by contest problem (googlejamN):
  fit      : pairs fully inside the encoder-train problems {1,5,6,8}
  thr-val  : pairs fully inside val problems {4,7,11,12} (also used for
             encoder monitoring) -> decision threshold picked here (B2)
  test     : pairs fully inside test problems {2,3,9,10} — never seen by the
             encoder, the PCA, the classifier, or the threshold selection.

Reads the pd feature matrix and the file lists saved by the fine-tune script.
Writes <features-dir>/protocol_eval.json.

Run:  nohup python tools/eval_gcj_pd.py > logs/eval_gcj_pd.log 2>&1 &
"""
import argparse
import json
import os

import numpy as np
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

RANDOM_STATE = 12345

parser = argparse.ArgumentParser()
parser.add_argument("--features", default="results/gcj_148_ft_pd/features.npz")
parser.add_argument("--model-dir", default="models/unixcoder_gcj_ft_pd")
parser.add_argument("--n-cols", type=int, default=0,
                    help="use only the first N feature columns (e.g. 19 = hand-crafted only)")
parser.add_argument("--tag", default="", help="suffix for the output json name")
parser.add_argument("--sample", type=int, default=0)
args = parser.parse_args()

data = np.load(args.features, allow_pickle=True)
X, y = data["X"], data["y"].astype(int)
if args.n_cols:
    X = X[:, :args.n_cols]
path_a, path_b = data["path_a"], data["path_b"]
if args.sample:
    rng = np.random.RandomState(RANDOM_STATE)
    keep = rng.choice(len(y), size=args.sample, replace=False)
    X, y, path_a, path_b = X[keep], y[keep], path_a[keep], path_b[keep]

def load_list(name):
    with open(os.path.join(args.model_dir, name)) as f:
        return set(line.strip() for line in f if line.strip())

val_files, test_files = load_list("val_files.txt"), load_list("test_files.txt")
a_val, b_val = np.isin(path_a, list(val_files)), np.isin(path_b, list(val_files))
a_tst, b_tst = np.isin(path_a, list(test_files)), np.isin(path_b, list(test_files))

fit_mask = ~(a_val | b_val | a_tst | b_tst)
thr_mask = a_val & b_val
test_mask = a_tst & b_tst
print(f"fit {fit_mask.sum()}, thr-val {thr_mask.sum()}, test {test_mask.sum()} "
      f"(test clone rate {y[test_mask].mean():.3f})", flush=True)

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


m = xgb.XGBClassifier(**xgb_params)
m.fit(X[fit_mask], y[fit_mask])
p_thr = m.predict_proba(X[thr_mask])[:, 1]
grid = np.linspace(0.01, 0.99, 99)
t_star = float(grid[int(np.argmax([f1_score(y[thr_mask], (p_thr >= t).astype(int))
                                   for t in grid]))])
p_test = m.predict_proba(X[test_mask])[:, 1]
results = {
    "n_fit": int(fit_mask.sum()), "n_thr_val": int(thr_mask.sum()),
    "n_test": int(test_mask.sum()), "test_clone_rate": float(y[test_mask].mean()),
    "at_0.5": metrics(y[test_mask], p_test, 0.5),
    "at_val_threshold": metrics(y[test_mask], p_test, t_star),
}
print("problem-disjoint @0.5:", results["at_0.5"], flush=True)
print(f"problem-disjoint @t*={t_star:.2f}:", results["at_val_threshold"], flush=True)

if not args.sample:
    name = f"protocol_eval{'_' + args.tag if args.tag else ''}.json"
    out = os.path.join(os.path.dirname(args.features), name)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", out)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from run_meta import save_run_meta
    save_run_meta(os.path.dirname(out))
