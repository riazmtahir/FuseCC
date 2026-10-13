"""Complementarity analysis — SHAP, permutation importance,
feature-family correlation. Answers the questions "why these
measures / why does it work".

For each dataset (BCB = results/bcb_e3/features.npz, GCJ =
results/gcj_148_ft/features.npz):
  1. Train XGB (GPU) on a stratified 80% split, SHAP on 20k held-out rows.
  2. Permutation importance (F1 drop) on 50k held-out rows, grouped by
     feature FAMILY (lexical t1-t14/t16, semantic-metric t15+t17-t19,
     embedding cos/diff/prod) as well as per-feature top-20.
  3. Spearman correlation matrix of the similarity metrics (t1..t19/t20)
     + mean absolute inter-family vs intra-family correlation.

Outputs under results/f1_analysis/<dataset>/:
  shap_summary.png, shap_mean_abs.csv, perm_importance.csv,
  metric_correlation.csv, correlation_heatmap.png, family_summary.json

Run:  nohup python tools/f1_analysis.py > logs/f1_analysis.log 2>&1 &
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 12345
rng = np.random.RandomState(RANDOM_STATE)

DATASETS = {
    "bcb": "results/bcb_e3/features.npz",
    "gcj": "results/gcj_148_ft/features.npz",
    "gcj_zs": "results/gcj_148/features.npz",  # zero-shot encoder features
}

parser = argparse.ArgumentParser()
parser.add_argument("--only", choices=sorted(DATASETS), default=None)
args = parser.parse_args()
if args.only:
    DATASETS = {args.only: DATASETS[args.only]}


def family_of(name: str) -> str:
    if name.startswith("uxc"):
        return "embedding_cos" if "cos" in name else ("embedding_diff" if "diff" in name
                                                      else "embedding_prod")
    t = int(name.split("_t")[1])
    if t == 15 or t >= 17:
        return "semantic_metric"
    return "lexical_fingerprint"


for ds, path in DATASETS.items():
    out = f"results/f1_analysis/{ds}"
    os.makedirs(out, exist_ok=True)
    data = np.load(path, allow_pickle=True)
    X, y = data["X"], data["y"].astype(int)
    names = [str(n) for n in data["feature_names"]]
    fams = np.array([family_of(n) for n in names])
    print(f"== {ds}: X {X.shape}", flush=True)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=RANDOM_STATE)
    model = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                              subsample=0.8, tree_method="hist", device="cuda",
                              random_state=RANDOM_STATE, eval_metric="logloss")
    model.fit(Xtr, ytr)

    # --- SHAP (XGBoost-native TreeSHAP; shap.TreeExplainer can't parse
    # xgboost>=3 base_score strings) -----------------------------------------
    sh_idx = rng.choice(len(Xte), size=min(20000, len(Xte)), replace=False)
    booster = model.get_booster()
    booster.set_param({"device": "cpu"})
    contribs = booster.predict(xgb.DMatrix(Xte[sh_idx]), pred_contribs=True)
    sv = contribs[:, :-1]  # last column is the bias term
    mean_abs = np.abs(sv).mean(axis=0)
    pd.DataFrame({"feature": names, "family": fams, "mean_abs_shap": mean_abs}) \
        .sort_values("mean_abs_shap", ascending=False) \
        .to_csv(f"{out}/shap_mean_abs.csv", index=False)
    plt.figure()
    shap.summary_plot(sv, Xte[sh_idx], feature_names=names, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(f"{out}/shap_summary.png", dpi=200)
    plt.close("all")

    # --- permutation importance (per family + top single features) --------
    pi_idx = rng.choice(len(Xte), size=min(50000, len(Xte)), replace=False)
    Xpi, ypi = Xte[pi_idx].copy(), yte[pi_idx]
    base_f1 = f1_score(ypi, model.predict(Xpi))
    perm_rows = []
    for fam in sorted(set(fams)):
        cols = np.where(fams == fam)[0]
        Xp = Xpi.copy()
        for c_ in cols:
            Xp[:, c_] = Xp[rng.permutation(len(Xp)), c_]
        perm_rows.append({"group": fam, "kind": "family", "n_features": len(cols),
                          "f1_drop": float(base_f1 - f1_score(ypi, model.predict(Xp)))})
    top20 = np.argsort(-mean_abs)[:20]
    for c_ in top20:
        Xp = Xpi.copy()
        Xp[:, c_] = Xp[rng.permutation(len(Xp)), c_]
        perm_rows.append({"group": names[c_], "kind": "feature", "n_features": 1,
                          "f1_drop": float(base_f1 - f1_score(ypi, model.predict(Xp)))})
    pd.DataFrame(perm_rows).sort_values(["kind", "f1_drop"], ascending=[True, False]) \
        .to_csv(f"{out}/perm_importance.csv", index=False)
    print(f"{ds}: base F1 {base_f1:.4f}; family drops "
          f"{[(r['group'], round(r['f1_drop'], 4)) for r in perm_rows if r['kind'] == 'family']}",
          flush=True)

    # --- metric correlation (similarity scores only) -----------------------
    n_scores = sum(1 for n in names if not n.startswith("uxc") or "cos" in n)
    score_cols = [i for i, n in enumerate(names) if not n.startswith("uxc_diff")
                  and not n.startswith("uxc_prod")]
    corr_idx = rng.choice(len(X), size=min(100000, len(X)), replace=False)
    corr, _ = spearmanr(X[corr_idx][:, score_cols])
    labels = [names[i] for i in score_cols]
    pd.DataFrame(corr, index=labels, columns=labels).to_csv(f"{out}/metric_correlation.csv")
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    fig.colorbar(im)
    ax.set_title(f"{ds.upper()} similarity-metric Spearman correlation")
    plt.tight_layout()
    plt.savefig(f"{out}/correlation_heatmap.png", dpi=200)
    plt.close("all")

    fam_of_score = [family_of(labels[i]) for i in range(len(labels))]
    intra, inter = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (intra if fam_of_score[i] == fam_of_score[j] else inter).append(abs(corr[i, j]))
    summary = {
        "base_f1_holdout": float(base_f1),
        "family_perm_f1_drop": {r["group"]: r["f1_drop"] for r in perm_rows
                                if r["kind"] == "family"},
        "family_mean_abs_shap": {f: float(mean_abs[fams == f].sum())
                                 for f in sorted(set(fams))},
        "mean_abs_corr_intra_family": float(np.mean(intra)),
        "mean_abs_corr_inter_family": float(np.mean(inter)),
    }
    with open(f"{out}/family_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

print("F1 ANALYSIS DONE")
