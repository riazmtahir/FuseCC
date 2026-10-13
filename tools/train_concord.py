#!/usr/bin/env python3
"""Train + evaluate FuseCC on a ConCord benchmark (binary pair protocol).

Feature layout mirrors the GCJ 148-feature build so results are directly comparable:
    cols 0..18   = t1..t19 similarity metrics
    col  19      = UniXcoder cosine
    cols 20..83  = |a-b| in 64-dim PCA space
    cols 84..147 = a*b   in 64-dim PCA space

LEAKAGE CONTROL: the PCA (and the centering mean) are fitted on TRAIN-split samples
ONLY. The GCJ zero-shot run fitted PCA on all 1,664 files and had to carry an
"unsupervised-only leakage" caveat (GCJ problem-disjoint experiment); there is no reason to repeat
that here, and on a problem-disjoint benchmark it would be a much worse look.

Protocol: train on the train split, pick the decision threshold on VALID (B2), report on
TEST. Splits are problem-disjoint by the benchmark's own design, so test problems are
unseen. Pairs are balanced 50/50 -> the all-clone baseline is F1 0.667; AUC is reported
alongside F1 for that reason.

Usage:
    python tools/train_concord.py --dataset java250
    python tools/train_concord.py --dataset poj104 --ensemble
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, _REPO_ROOT)
RANDOM_STATE = 12345
DATA_ROOT = _REPO_ROOT + "/dataset/concord"
OUT_ROOT = _REPO_ROOT + "/results/concord"


def build_features(d, split, pca, mean_vec, emb_tag=""):
    f = np.load(os.path.join(d, f"features_{split}.npz"), allow_pickle=True)
    X19, y, pairs = f["X"], f["y"], f["pairs"]
    emb_file = f"embeddings_{split}.npy" if not emb_tag else f"embeddings_{emb_tag}_{split}.npy"
    emb = np.load(os.path.join(d, emb_file))

    a = emb[pairs[:, 0]] - mean_vec
    b = emb[pairs[:, 1]] - mean_vec
    # cosine on centered vectors (centering before cosine was worth +0.15 AUC on GCJ)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cos = (num / np.clip(den, 1e-9, None)).astype(np.float32)

    pa, pb = pca.transform(a), pca.transform(b)
    X = np.hstack([X19, cos[:, None], np.abs(pa - pb), pa * pb]).astype(np.float32)
    return X, y.astype(np.int8)


def metrics_at(y, prob, thr):
    pred = (prob >= thr).astype(np.int8)
    return {"threshold": float(thr),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "auc": float(roc_auc_score(y, prob))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ensemble", action="store_true",
                    help="CatBoost+XGB+LGB -> MLP meta, as in the GCJ headline run")
    ap.add_argument("--emb-tag", default="",
                    help="embedding backbone tag; '' = UniXcoder (embeddings_<split>.npy), "
                         "else reads embeddings_<tag>_<split>.npy (e.g. jinacode_sym)")
    ap.add_argument("--suffix", default="",
                    help="suffix for artifacts so a backbone run does not overwrite the "
                         "UniXcoder one (e.g. _jina -> summary_jina.json, xgb_148_jina.joblib)")
    args = ap.parse_args()

    d = os.path.join(DATA_ROOT, args.dataset)
    out = os.path.join(OUT_ROOT, args.dataset)
    os.makedirs(out, exist_ok=True)
    et = args.emb_tag

    # --- PCA fitted on TRAIN samples only -----------------------------------------
    emb_tr_file = "embeddings_train.npy" if not et else f"embeddings_{et}_train.npy"
    emb_tr = np.load(os.path.join(d, emb_tr_file))
    mean_vec = emb_tr.mean(0)
    pca = PCA(n_components=64, random_state=RANDOM_STATE).fit(emb_tr - mean_vec)
    print(f"PCA fitted on {emb_tr.shape[0]:,} TRAIN samples (backbone='{et or 'unixcoder'}', "
          f"explained var {pca.explained_variance_ratio_.sum():.3f})")

    data = {s: build_features(d, s, pca, mean_vec, et) for s in ("train", "valid", "test")}
    for s, (X, y) in data.items():
        print(f"{s:5s}: X={X.shape} pos_rate={y.mean():.3f}")

    Xtr, ytr = data["train"]
    Xva, yva = data["valid"]
    Xte, yte = data["test"]

    import xgboost as xgb
    t0 = time.time()
    clf = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method="hist", device="cuda",
                            random_state=RANDOM_STATE, eval_metric="logloss")
    clf.fit(Xtr, ytr)
    fit_s = time.time() - t0
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]

    # threshold chosen on VALID, applied to TEST
    grid = np.linspace(0.05, 0.95, 181)
    best_thr = max(grid, key=lambda t: f1_score(yva, (p_va >= t).astype(np.int8),
                                                zero_division=0))
    res = {"xgb_19+emb148": {
        "valid": metrics_at(yva, p_va, best_thr),
        "test": metrics_at(yte, p_te, best_thr),
        "test_at_0.5": metrics_at(yte, p_te, 0.5),
        "fit_seconds": round(fit_s, 1)}}
    print(f"\nXGB(148)  thr={best_thr:.3f} (picked on valid)")
    print(f"  TEST  P {res['xgb_19+emb148']['test']['precision']:.4f} "
          f"R {res['xgb_19+emb148']['test']['recall']:.4f} "
          f"F1 {res['xgb_19+emb148']['test']['f1']:.4f} "
          f"AUC {res['xgb_19+emb148']['test']['auc']:.4f}")

    # ablation: hand-crafted metrics only (no embeddings) -- the complementarity story
    clf19 = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1,
                              subsample=0.8, colsample_bytree=0.8,
                              tree_method="hist", device="cuda",
                              random_state=RANDOM_STATE, eval_metric="logloss")
    clf19.fit(Xtr[:, :19], ytr)
    q_va = clf19.predict_proba(Xva[:, :19])[:, 1]
    q_te = clf19.predict_proba(Xte[:, :19])[:, 1]
    thr19 = max(grid, key=lambda t: f1_score(yva, (q_va >= t).astype(np.int8),
                                             zero_division=0))
    res["xgb_t1_t19_only"] = {"valid": metrics_at(yva, q_va, thr19),
                              "test": metrics_at(yte, q_te, thr19)}
    print(f"XGB(t1-t19 only)  TEST F1 {res['xgb_t1_t19_only']['test']['f1']:.4f} "
          f"AUC {res['xgb_t1_t19_only']['test']['auc']:.4f}")

    # embedding-only reference (cosine + PCA interactions, no hand-crafted metrics)
    clfE = xgb.XGBClassifier(n_estimators=600, max_depth=8, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", device="cuda",
                             random_state=RANDOM_STATE, eval_metric="logloss")
    clfE.fit(Xtr[:, 19:], ytr)
    e_va = clfE.predict_proba(Xva[:, 19:])[:, 1]
    e_te = clfE.predict_proba(Xte[:, 19:])[:, 1]
    thrE = max(grid, key=lambda t: f1_score(yva, (e_va >= t).astype(np.int8),
                                            zero_division=0))
    res["xgb_embeddings_only"] = {"valid": metrics_at(yva, e_va, thrE),
                                  "test": metrics_at(yte, e_te, thrE)}
    print(f"XGB(embeddings only)  TEST F1 {res['xgb_embeddings_only']['test']['f1']:.4f} "
          f"AUC {res['xgb_embeddings_only']['test']['auc']:.4f}")

    if args.ensemble:
        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier
        from sklearn.neural_network import MLPClassifier
        bases = {
            "xgb": clf,
            "lgb": LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.1,
                                  random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
            "cb": CatBoostClassifier(iterations=600, depth=8, learning_rate=0.1,
                                     random_seed=RANDOM_STATE, verbose=0,
                                     task_type="GPU"),
        }
        Pva, Pte = [], []
        for name, m in bases.items():
            if name != "xgb":
                m.fit(Xtr, ytr)
            Pva.append(m.predict_proba(Xva)[:, 1])
            Pte.append(m.predict_proba(Xte)[:, 1])
        Pva, Pte = np.column_stack(Pva), np.column_stack(Pte)
        meta = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500,
                             random_state=RANDOM_STATE)
        # meta is fitted on VALID base-probabilities, then applied to TEST; valid is
        # already the threshold-selection split so no extra data is consumed
        meta.fit(Pva, yva)
        m_te = meta.predict_proba(Pte)[:, 1]
        res["stacking_mlp"] = {"test_at_0.5": metrics_at(yte, m_te, 0.5)}
        print(f"Stacking-MLP  TEST F1 {res['stacking_mlp']['test_at_0.5']['f1']:.4f} "
              f"AUC {res['stacking_mlp']['test_at_0.5']['auc']:.4f}")

    # persist model + PCA so tools/eval_concord_mapr.py can re-rank with the SAME model
    import joblib
    sfx = args.suffix
    joblib.dump(clf, os.path.join(out, f"xgb_148{sfx}.joblib"))
    joblib.dump((pca, mean_vec), os.path.join(out, f"pca{sfx}.joblib"))

    backbone = et or "unixcoder"
    meta_json = json.load(open(os.path.join(d, "meta.json")))
    summary = {
        "dataset": args.dataset,
        "language": meta_json["language"],
        "backbone": backbone,
        "protocol": meta_json["protocol"],
        "detokenization": meta_json["detokenization"],
        "feature_layout": f"0-18 t1-t19 | 19 {backbone} cosine | 20-83 |a-b| PCA64 | 84-147 a*b PCA64",
        "pca_fit": "TRAIN split samples only (no valid/test leakage)",
        "encoder": f"{backbone} embeddings, zero-shot (no encoder fine-tuning)",
        "threshold_selection": "best-F1 on VALID, applied to TEST (B2)",
        "baseline_note": "pairs are balanced 50/50 -> all-clone baseline F1 = 0.667; "
                         "judge by AUC and precision as well as F1",
        "comparability_warning": (
            "These are the OFFICIAL problem-disjoint CodeXGLUE/CONCORD splits, but the "
            "official metric for both benchmarks is MAP@R (R=499 POJ-104 / R=299 "
            "Java250), not binary pair F1 — see results/concord/*/mapr.json for the "
            "comparable numbers. Do NOT compare these F1 values with the OJClone F1 "
            "figures reported by ASTNN/CDLH: those use the first 15 POJ-104 problems "
            "with a random pair split, so problems are SHARED across train/test."),
        "results": res,
        "splits": meta_json["splits"],
    }
    with open(os.path.join(out, f"summary{sfx}.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    try:
        from tools.run_meta import save_run_meta
        save_run_meta(out, extra={"dataset": args.dataset, "backbone": backbone})
    except Exception as exc:
        print(f"(run_meta not written: {exc})")
    print("\nwrote", os.path.join(out, f"summary{sfx}.json"))


if __name__ == "__main__":
    main()
