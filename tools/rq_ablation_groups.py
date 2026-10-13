"""Group-wise feature ablation over the similarity-metric families of the paper's
method section, on the new pipeline (t1-t19, new tokenizer).

Groups (method.tex taxonomy + the new semantic-normalization group):
  overlap          t1 Jaccard, t2 Dice
  edit_distance    t3 Jaro, t4 Jaro-Winkler, t5 Lev distance, t6 Lev ratio, t8 Fuzzy
  frequency        t7 BoW cosine, t11 TF-IDF
  sequence         t9 LCS, t10 N-gram, t14 RKR-GST
  fingerprint      t12 token-category, t13 rolling hash, t15 SeqRatio, t16 winnowing
  semantic_norm    t17 identifier abstraction, t18 AST norm, t19 control-flow (NEW)

For each dataset regime (bcb-all, bcb-T1-T3, gcj): XGBoost(GPU) 5-fold OOF with
the full 19 features (baseline), leave-one-group-out, and each group alone.
Reports F1@0.5 / AUC (+ per-clone-type recall on BCB) and, for the full model,
XGB gain importance + permutation-drop per feature (feeds the RQ4 table).

Usage: rq_ablation_groups.py [--only bcb_all|bcb_t13|gcj] [--sample N]
Outputs: results/rq_ablation_groups/<regime>/{summary.json,ablation_table.csv,
         feature_importance.csv}
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

SEED = 12345
REPO = _REPO_ROOT

GROUPS = {
    "overlap": [0, 1],                    # t1 t2
    "edit_distance": [2, 3, 4, 5, 7],     # t3 t4 t5 t6 t8
    "frequency": [6, 10],                 # t7 t11
    "sequence": [8, 9, 13],               # t9 t10 t14
    "fingerprint": [11, 12, 14, 15],      # t12 t13 t15 t16
    "semantic_norm": [16, 17, 18],        # t17 t18 t19
}
TNAMES = [f"t{i}" for i in range(1, 20)]


def load(regime):
    if regime.startswith("bcb"):
        d = np.load(f"{REPO}/results/bcb_e3/features.npz", allow_pickle=True)
        X = np.nan_to_num(d["X"][:, :19].astype(np.float32), nan=0.0)
        y, ct = d["y"].astype(int), d["clone_type"]
        if regime == "bcb_t13":
            keep = ct != "WT3/T4"
            return X[keep], y[keep], ct[keep]
        return X, y, ct
    d = np.load(f"{REPO}/results/gcj_148/features.npz", allow_pickle=True)
    return (np.nan_to_num(d["X"][:, :19].astype(np.float32), nan=0.0),
            d["y"].astype(int), None)


def xgb():
    return XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                         tree_method="hist", device="cuda", n_jobs=8,
                         random_state=SEED, eval_metric="logloss")


def cv_eval(X, y, cols, ct=None, collect_importance=False):
    """5-fold OOF probabilities with XGB on the given feature columns."""
    oof = np.zeros(len(y))
    imps = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y):
        m = xgb()
        m.fit(X[tr][:, cols], y[tr])
        oof[te] = m.predict_proba(X[te][:, cols])[:, 1]
        if collect_importance:
            imps.append(m.feature_importances_)
    pred = (oof >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                  zero_division=0)
    out = {"precision": round(float(p), 4), "recall": round(float(r), 4),
           "f1": round(float(f1), 6), "auc": round(float(roc_auc_score(y, oof)), 6)}
    if ct is not None:
        out["per_type_recall"] = {
            t: round(float(pred[(ct == t) & (y == 1)].mean()), 4)
            for t in ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]
            if ((ct == t) & (y == 1)).sum()}
    if collect_importance:
        out["_gain_importance"] = np.mean(imps, axis=0)
    return out


def permutation_drop(X, y, base_f1):
    """Single 80/20 split permutation importance per feature (fast, seed-fixed)."""
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y,
                              random_state=SEED)
    m = xgb()
    m.fit(X[tr], y[tr])
    prob = m.predict_proba(X[te])[:, 1]
    f1_ref = precision_recall_fscore_support(
        y[te], (prob >= 0.5).astype(int), average="binary", zero_division=0)[2]
    rng = np.random.RandomState(SEED)
    drops = []
    for j in range(X.shape[1]):
        Xp = X[te].copy()
        Xp[:, j] = Xp[rng.permutation(len(te)), j]
        pj = m.predict_proba(Xp)[:, 1]
        f1j = precision_recall_fscore_support(
            y[te], (pj >= 0.5).astype(int), average="binary", zero_division=0)[2]
        drops.append(f1_ref - f1j)
    return np.array(drops)


def run_regime(regime, sample):
    X, y, ct = load(regime)
    if sample:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y), sample, replace=False)
        X, y = X[idx], y[idx]
        ct = ct[idx] if ct is not None else None
    out = f"{REPO}/results/rq_ablation_groups/{regime}"
    os.makedirs(out, exist_ok=True)
    print(f"[{regime}] n={len(y)} clone-rate {y.mean():.3f}", flush=True)

    rows = []
    full = cv_eval(X, y, list(range(19)), ct, collect_importance=True)
    gain = full.pop("_gain_importance")
    full.update({"config": "full t1-t19", "features": 19,
                 "delta_f1_vs_full": 0.0})
    rows.append(full)
    print(f"  full: F1 {full['f1']:.4f} AUC {full['auc']:.4f}", flush=True)

    for gname, cols in GROUPS.items():
        rest = [c for c in range(19) if c not in cols]
        r = cv_eval(X, y, rest, ct)
        r.update({"config": f"without {gname}", "features": len(rest),
                  "delta_f1_vs_full": round(r["f1"] - full["f1"], 6)})
        rows.append(r)
        a = cv_eval(X, y, cols, ct)
        a.update({"config": f"{gname} alone", "features": len(cols),
                  "delta_f1_vs_full": round(a["f1"] - full["f1"], 6)})
        rows.append(a)
        print(f"  -{gname:14s}: F1 {r['f1']:.4f} (d {r['delta_f1_vs_full']:+.4f}) | "
              f"alone {a['f1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    lead = ["config", "features", "f1", "delta_f1_vs_full", "precision",
            "recall", "auc"]
    df[lead + [c for c in df.columns if c not in lead]].to_csv(
        os.path.join(out, "ablation_table.csv"), index=False)

    drops = permutation_drop(X, y, full["f1"])
    imp = pd.DataFrame({"feature": TNAMES, "xgb_gain": gain,
                        "perm_f1_drop": drops}).sort_values(
        "perm_f1_drop", ascending=False)
    imp.to_csv(os.path.join(out, "feature_importance.csv"), index=False)
    print(imp.head(6).round(4).to_string(index=False), flush=True)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump({"regime": regime, "n_pairs": int(len(y)), "seed": SEED,
                   "protocol": "XGB(GPU d8 n400) 5-fold OOF, F1@0.5; groups per "
                               "method-section taxonomy + semantic_norm (t17-t19)",
                   "groups": {k: [TNAMES[c] for c in v] for k, v in GROUPS.items()},
                   "table": rows}, f, indent=2)
    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(out)
    except Exception as e:
        print(f"run_meta skipped: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["bcb_all", "bcb_t13", "gcj"])
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    for regime in ([args.only] if args.only else ["bcb_all", "bcb_t13", "gcj"]):
        run_regime(regime, args.sample)
    print("all done", flush=True)


if __name__ == "__main__":
    main()
