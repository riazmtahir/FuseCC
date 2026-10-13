#!/usr/bin/env python
"""TriFusion-LLM re-run: final arbitrated headline on the full test set.

Merges the DeepSeek arbitration decisions (low-confidence pred in {2,3,4,5}
subset) into the fusion model's full-test predictions and recomputes the same
metrics as tset.py's test_evaluation_stats.json, so the no-arb and arbitrated
rows are directly comparable.

Output: results/baselines_rerun/trifusion_bcb/summary.json
"""
import os as _os
_REPO_ROOT = _os.environ.get("FUSECC_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_EXT_ROOT = _os.environ.get("FUSECC_EXTERNAL", _os.path.dirname(_REPO_ROOT))
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tools.run_meta import save_run_meta  # noqa: E402

PRED_DIR = _EXT_ROOT + "/baselines/TriFusion-LLM/my_predictions/run_seed42"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "baselines_rerun", "trifusion_bcb")


def metrics_block(y_true, y_pred):
    return {
        "total_samples": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_f1": [round(v, 4) for v in f1_score(y_true, y_pred, average=None, zero_division=0)],
        "per_class_recall": [round(v, 4) for v in recall_score(y_true, y_pred, average=None, zero_division=0)],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    probs = pd.read_csv(os.path.join(PRED_DIR, "test_probability_matrix_with_code.csv"),
                        usecols=["sample_id", "true_label", "pred_label"])
    arb_csvs = sorted(glob.glob(os.path.join(PRED_DIR, "deepseek_arbitration_results",
                                             "ds_arbitration_results_*.csv")))
    assert arb_csvs, "no arbitration results CSV found"
    arb = pd.read_csv(arb_csvs[-1])
    print(f"full test: {len(probs)} rows; arbitrated subset: {len(arb)} rows ({arb_csvs[-1]})")

    y_true = probs["true_label"].values
    y_noarb = probs["pred_label"].values.copy()

    # override the arbitrated sample_ids with DeepSeek's decision
    ds_map = dict(zip(arb["sample_id"].astype(int), arb["deepseek_prediction"].astype(int)))
    y_arb = y_noarb.copy()
    idx = probs["sample_id"].astype(int).map(ds_map)
    mask = idx.notna().values
    y_arb[mask] = idx[mask].astype(int).values
    changed = int((y_arb != y_noarb).sum())
    print(f"overridden: {mask.sum()} samples, of which {changed} predictions actually changed")

    no_arb = metrics_block(y_true, y_noarb)
    with_arb = metrics_block(y_true, y_arb)

    # subset-level improvement stats (matches deepseek.py's own report)
    sub = probs[mask]
    sub_true = sub["true_label"].values
    sub_orig = sub["pred_label"].values
    sub_ds = y_arb[mask.nonzero()[0]] if hasattr(mask, "nonzero") else y_arb[np.where(mask)[0]]
    improved = int(((sub_ds == sub_true) & (sub_orig != sub_true)).sum())
    worsened = int(((sub_ds != sub_true) & (sub_orig == sub_true)).sum())

    summary = {
        "system": "TriFusion-LLM (arXiv 2603.15004, 2026) re-run — ML(LightGBM) + CodeBERT fusion + DeepSeek arbitration",
        "dataset": "their BCB-derived 7-class dataset, split_dataset.py --seed 42 (train 78,688 / val 6,002 / test 934,258 = FULL test split; paper uses 934,677)",
        "arbitration": {
            "model": "deepseek-chat (V3.2, api.deepseek.com), temperature 0.1 — the paper's arbitrator",
            "filter": "confidence in [0,0.6) AND fusion pred in {2,3,4,5} (their defaults)",
            "n_arbitrated": int(mask.sum()),
            "n_changed": changed,
            "improved": improved,
            "worsened": worsened,
            "net_improvement": improved - worsened,
            "description": "deepseek-chat (V3.2) re-labelled the fusion model's low-confidence predictions (confidence < 0.6, predicted class in {2,3,4,5}); counts above compare those labels with the fusion predictions",
        },
        "no_arbitration": no_arb,
        "with_arbitration": with_arb,
        "notes": [
            "no_arbitration row reproduces tset.py's test_evaluation_stats.json exactly (cross-check).",
            "Fusion stage: reconstructed train.py (upstream ships it empty at git HEAD 51a091f), authors' components + config.py hyperparams, seed 42.",
            "ML stage setting: Optuna 5 trials instead of the argparse default of 15 (trial-0 log-loss was already 0.0794).",
            "Full 934,258-sample test used (the planned 100k cap was not in effect for the final chain) — matches the paper's full-test protocol.",
        ],
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    save_run_meta(OUT_DIR)
    print(json.dumps({k: summary[k] for k in ("arbitration",)}, indent=2))
    print(f"no-arb  : acc {no_arb['accuracy']:.4f}  f1_macro {no_arb['f1_macro']:.4f}  f1_weighted {no_arb['f1_weighted']:.4f}")
    print(f"with-arb: acc {with_arb['accuracy']:.4f}  f1_macro {with_arb['f1_macro']:.4f}  f1_weighted {with_arb['f1_weighted']:.4f}")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
