"""RQ2 rerun (paper section "Effectiveness of Individual Components"), new pipeline.

For each of the 19 hand-crafted metrics (t1..t19) plus the zero-shot UniXcoder
cosine, use the metric alone as a threshold classifier:
  - stratified 60/20/20 train/val/test split (train unused for single metrics);
  - orientation from val AUC (distance-like metrics get negated);
  - threshold = best F1 on val (never picked on test);
  - report test AUC / P / R / F1 (+ per-type recall on BCB).
Reference rows: XGBoost trained on t1-t19 and on t1-t19+cosine under the same
split/threshold protocol. Replaces the old fixed {0.7,0.8,0.9} threshold tables.

Usage: rq2_individual_metrics.py --dataset bcb|gcj [--sample N]
Outputs: results/rq2_metrics/<dataset>/summary.json, metrics_table.csv,
         sweep curves per metric (f1 vs threshold on val) in sweeps.npz
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
from sklearn.model_selection import train_test_split

SEED = 12345
REPO = _REPO_ROOT

METRIC_NAMES = {
    "score_t1": "Jaccard", "score_t2": "Dice", "score_t3": "Jaro",
    "score_t4": "Jaro-Winkler", "score_t5": "Levenshtein distance",
    "score_t6": "Levenshtein ratio", "score_t7": "BoW cosine",
    "score_t8": "Fuzzy matching", "score_t9": "LCS",
    "score_t10": "N-gram", "score_t11": "TF-IDF",
    "score_t12": "Token-category (AST proxy)", "score_t13": "Rolling hash",
    "score_t14": "RKR-GST", "score_t15": "Semantic clone (SeqRatio)",
    "score_t16": "Winnowing", "score_t17": "Identifier abstraction",
    "score_t18": "AST normalization", "score_t19": "Control-flow stats",
    "uxc_cos": "UniXcoder cosine (zero-shot)",
    "score_t20": "UniXcoder cosine (zero-shot)",
}


def load(dataset):
    if dataset == "bcb":
        d = np.load(f"{REPO}/results/bcb_e3/features.npz", allow_pickle=True)
        names = [str(n) for n in d["feature_names"][:20]]
        return d["X"][:, :20], d["y"].astype(int), names, d["clone_type"]
    d = np.load(f"{REPO}/results/gcj_148/features.npz", allow_pickle=True)
    names = [str(n) for n in d["feature_names"][:20]]
    return d["X"][:, :20], d["y"].astype(int), names, None


def best_f1_threshold(scores, y, n_grid=999):
    qs = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, n_grid)))
    best_t, best_f1, curve = qs[0], -1.0, []
    for t in qs:
        pred = scores >= t
        tp = np.sum(pred & (y == 1))
        p = tp / max(pred.sum(), 1)
        r = tp / max((y == 1).sum(), 1)
        f1 = 2 * p * r / max(p + r, 1e-12)
        curve.append(f1)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, np.array(qs), np.array(curve)


def eval_at(scores, y, thr, clone_type=None):
    pred = (scores >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                  zero_division=0)
    out = {"precision": float(p), "recall": float(r), "f1": float(f1)}
    if clone_type is not None:
        per_type = {}
        for t in ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]:
            m = (clone_type == t) & (y == 1)
            if m.sum():
                per_type[t] = round(float(pred[m].mean()), 4)
        out["per_type_recall"] = per_type
    return out


def run_regime(X, y, names, clone_type, tag, out):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=y, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=SEED)
    print(f"[{tag}] train {len(tr)} val {len(va)} test {len(te)} "
          f"clone-rate {y.mean():.3f}", flush=True)

    rows, sweeps = [], {}
    for j, name in enumerate(names):
        s = np.nan_to_num(X[:, j].astype(float), nan=0.0)
        auc_va = roc_auc_score(y[va], s[va])
        direction = "higher=similar"
        if auc_va < 0.5:
            s, direction, auc_va = -s, "lower=similar", 1 - auc_va
        thr, grid, curve = best_f1_threshold(s[va], y[va])
        res = eval_at(s[te], y[te], thr, clone_type[te] if clone_type is not None else None)
        res.update({"metric": METRIC_NAMES.get(name, name), "col": name,
                    "direction": direction,
                    "val_auc": round(float(auc_va), 4),
                    "test_auc": round(float(roc_auc_score(y[te], s[te])), 4),
                    "threshold_on_val": float(thr)})
        rows.append(res)
        sweeps[name] = np.stack([grid, curve])
        print(f"  {name:10s} {res['metric']:32s} AUC {res['test_auc']:.3f} "
              f"F1 {res['f1']:.3f}", flush=True)

    # reference rows: trained classifier under the identical protocol
    from xgboost import XGBClassifier
    for label, cols in [("XGBoost (t1-t19)", slice(0, 19)),
                        ("XGBoost (t1-t19 + emb cosine)", slice(0, 20))]:
        Xf = np.nan_to_num(X[:, cols].astype(float), nan=0.0)
        clf = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                            tree_method="hist", device="cuda",
                            random_state=SEED, n_jobs=-1, eval_metric="logloss")
        clf.fit(Xf[tr], y[tr])
        pv = clf.predict_proba(Xf[va])[:, 1]
        pt = clf.predict_proba(Xf[te])[:, 1]
        thr, _, _ = best_f1_threshold(pv, y[va])
        res = eval_at(pt, y[te], thr, clone_type[te] if clone_type is not None else None)
        res.update({"metric": label, "col": "-", "direction": "higher=similar",
                    "val_auc": round(float(roc_auc_score(y[va], pv)), 4),
                    "test_auc": round(float(roc_auc_score(y[te], pt)), 4),
                    "threshold_on_val": float(thr)})
        rows.append(res)
        print(f"  {label}: AUC {res['test_auc']:.4f} F1 {res['f1']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    lead = ["metric", "direction", "test_auc", "threshold_on_val",
            "precision", "recall", "f1"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(os.path.join(out, f"metrics_table_{tag}.csv"), index=False)
    np.savez_compressed(os.path.join(out, f"sweeps_{tag}.npz"), **sweeps)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bcb", "gcj"], required=True)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    X, y, names, clone_type = load(args.dataset)
    if args.sample:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y), args.sample, replace=False)
        X, y = X[idx], y[idx]
        clone_type = clone_type[idx] if clone_type is not None else None

    out = f"{REPO}/results/rq2_metrics/{args.dataset}"
    os.makedirs(out, exist_ok=True)

    summary = {"dataset": args.dataset, "seed": SEED, "n_pairs": int(len(y)),
               "protocol": "stratified 60/20/20; orientation+threshold on val "
                           "(best F1), evaluated on test (B2)",
               "regimes": {}}
    if args.dataset == "bcb":
        summary["regimes"]["all"] = run_regime(X, y, names, clone_type, "all", out)
        keep = clone_type != "WT3/T4"
        summary["regimes"]["T1-T3"] = run_regime(
            X[keep], y[keep], names, clone_type[keep], "T1-T3", out)
    else:
        summary["regimes"]["random-split"] = run_regime(X, y, names, None,
                                                        "random-split", out)

    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    sys.path.insert(0, f"{REPO}/tools")
    try:
        from run_meta import save_run_meta
        save_run_meta(out)
    except Exception as e:
        print(f"run_meta skipped: {e!r}")
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
