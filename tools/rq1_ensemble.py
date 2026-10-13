"""RQ1 extension: ensemble of the top classifier families on the SAME 10 folds.

Reads results/rq1_classifiers/<ds>/summary.json (stage-2 table + best params),
takes the top-3 families by mean F1, and on each of the same shared folds:
  - inner 3-fold OOF probabilities of the 3 bases on the outer-train split;
  - soft voting (mean of base probabilities);
  - stacking with LR meta and MLP meta trained on the inner OOF probs;
  - bases retrained on the full outer-train, meta applied to their test probs.
Adds Friedman omnibus test over all systems (families + ensembles) and pairwise
Wilcoxon vs the best system. This is the fold-paired protocol that replaces the
old KS-test (independent-samples, wrong for shared folds).

Usage: rq1_ensemble.py --dataset bcb|gcj
Outputs: results/rq1_classifiers/<ds>/{ensemble_folds.csv,ensemble_summary.json}
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_classifier_comparison import SEED, REPO, load, make_model  # noqa: E402


def parse_param(family, s):
    if s == "None":
        return None
    if family == "LR":
        return float(s)
    return int(s)


def metas(seed):
    return {
        "Stacking-LR": LogisticRegression(max_iter=2000, random_state=seed),
        "Stacking-MLP": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500,
                                      random_state=seed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bcb", "gcj"], required=True)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--sample", type=int, default=0, help="debug subsample")
    args = ap.parse_args()

    out = f"{REPO}/results/rq1_classifiers/{args.dataset}"
    with open(os.path.join(out, "summary.json")) as f:
        summ = json.load(f)
    table = summ["stage2"]["table"]
    fams = sorted(table, key=lambda k: -table[k]["f1"])[:args.top]
    params = {f: parse_param(f, table[f]["param"]) for f in fams}
    print(f"[{args.dataset}] ensembling top-{args.top}: "
          f"{[(f, params[f]) for f in fams]}", flush=True)

    X, y = load(args.dataset)
    if args.sample:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(y), args.sample, replace=False)
        X, y = X[idx], y[idx]
    folds = list(StratifiedKFold(summ["stage2"]["folds"], shuffle=True,
                                 random_state=SEED).split(X, y))

    rows = []
    for k, (tr, te) in enumerate(folds):
        # inner OOF probs for meta training
        oof = np.zeros((len(tr), len(fams)))
        inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
        for j, fam in enumerate(fams):
            for itr, ite in inner.split(X[tr], y[tr]):
                m = make_model(fam, params[fam])
                m.fit(X[tr][itr], y[tr][itr])
                oof[ite, j] = m.predict_proba(X[tr][ite])[:, 1]
        # bases on full outer-train -> test probs
        test_probs = np.zeros((len(te), len(fams)))
        for j, fam in enumerate(fams):
            m = make_model(fam, params[fam])
            m.fit(X[tr], y[tr])
            test_probs[:, j] = m.predict_proba(X[te])[:, 1]

        systems = {"Voting (top3)": test_probs.mean(axis=1)}
        for name, meta in metas(SEED).items():
            meta.fit(oof, y[tr])
            systems[name] = meta.predict_proba(test_probs)[:, 1]
        for name, prob in systems.items():
            pred = (prob >= 0.5).astype(int)
            p, r, f1, _ = precision_recall_fscore_support(
                y[te], pred, average="binary", zero_division=0)
            rows.append({"family": name, "fold": k, "precision": float(p),
                         "recall": float(r), "f1": float(f1),
                         "auc": float(roc_auc_score(y[te], prob))})
            print(f"fold {k} {name:14s} F1 {f1:.4f}", flush=True)

    edf = pd.DataFrame(rows)
    edf.to_csv(os.path.join(out, "ensemble_folds.csv"), index=False)

    # combined significance over families + ensembles on the shared folds
    fdf = pd.read_csv(os.path.join(out, "stage2_folds.csv"))
    both = pd.concat([fdf[["family", "fold", "precision", "recall", "f1", "auc"]],
                      edf], ignore_index=True)
    piv = both.pivot(index="fold", columns="family", values="f1")
    fried = friedmanchisquare(*[piv[c].values for c in piv.columns])
    agg = both.groupby("family").agg(
        f1=("f1", "mean"), f1_std=("f1", "std"), precision=("precision", "mean"),
        recall=("recall", "mean"), auc=("auc", "mean")).sort_values(
        "f1", ascending=False)
    top = agg.index[0]
    pvals = {}
    for fam in agg.index[1:]:
        try:
            pvals[fam] = float(wilcoxon(piv[top].values, piv[fam].values).pvalue)
        except ValueError:
            pvals[fam] = 1.0
    print(agg.round(4).to_string(), flush=True)
    print(f"friedman chi2 {fried.statistic:.1f} p {fried.pvalue:.3g}; "
          f"wilcoxon vs {top}: " + str({k: round(v, 5) for k, v in pvals.items()}),
          flush=True)

    with open(os.path.join(out, "ensemble_summary.json"), "w") as f:
        json.dump({"dataset": args.dataset, "bases": {f: str(params[f]) for f in fams},
                   "metas": "voting / LR / MLP(32,16) on inner 3-fold OOF",
                   "friedman": {"chi2": float(fried.statistic),
                                "p": float(fried.pvalue)},
                   "table": json.loads(agg.round(6).to_json(orient="index")),
                   "wilcoxon_f1_vs_" + top: pvals}, f, indent=2)
    try:
        from run_meta import save_run_meta
        save_run_meta(out)
    except Exception as e:
        print(f"run_meta skipped: {e!r}")
    print("done ->", out, flush=True)


if __name__ == "__main__":
    main()
