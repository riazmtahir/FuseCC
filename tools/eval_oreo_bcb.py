"""Evaluate Oreo's predictions against our canonical BCB pairs.

Oreo (Saini et al., ESEC/FSE 2018) was run in its original artifact environment
on our 73k wrapped single-function files (dataset/BCB/id2sourcecode_wrapped_oreo),
so each predicted pair maps to BCB function ids via the file names.

Metrics reported (against dataset/BCB/pairs_canonical.csv):
  - per-clone-type recall: fraction of our labelled clone pairs Oreo reports
    (BigCloneEval-style recall)
  - labelled precision: among Oreo predictions that fall inside our labelled
    pair universe, the fraction labelled clone (predictions outside the
    labelled universe are counted but not judged — standard caveat)

Usage: python tools/eval_oreo_bcb.py --predictions <concatenated predictions.txt>
Writes results/oreo_bcb/summary.json.
"""
import argparse
import json
import os
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--predictions", required=True)
args = parser.parse_args()

pred_pairs = set()
n_lines = 0
with open(args.predictions, errors="ignore") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < 8:
            continue
        n_lines += 1
        a = int(parts[1].replace(".java", ""))
        b = int(parts[5].replace(".java", ""))
        if a == b:
            continue
        pred_pairs.add((min(a, b), max(a, b)))
print(f"prediction lines {n_lines} -> {len(pred_pairs)} unique cross-file id pairs")

canon = pd.read_csv("dataset/BCB/pairs_canonical.csv")
canon["lo"] = canon[["id_a", "id_b"]].min(axis=1)
canon["hi"] = canon[["id_a", "id_b"]].max(axis=1)
canon["key"] = list(zip(canon["lo"], canon["hi"]))
canon["predicted"] = canon["key"].isin(pred_pairs)

types = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]
recall = {}
for t in types:
    sub = canon[canon.clone_type == t]
    recall[t] = {"recall": float(sub.predicted.mean()), "n": int(len(sub))}
clones = canon[canon.label == 1]
recall["T1-T3_overall"] = {
    "recall": float(clones[clones.clone_type != "WT3/T4"].predicted.mean()),
    "n": int((clones.clone_type != "WT3/T4").sum()),
}

labelled_keys = set(canon["key"])
in_universe = [p for p in pred_pairs if p in labelled_keys]
clone_keys = set(canon.loc[canon.label == 1, "key"])
tp = sum(1 for p in in_universe if p in clone_keys)
precision_labelled = tp / len(in_universe) if in_universe else float("nan")

summary = {
    "prediction_lines": n_lines,
    "unique_pred_pairs": len(pred_pairs),
    "pred_in_labelled_universe": len(in_universe),
    "labelled_precision": precision_labelled,
    "per_type_recall": recall,
    "note": "Oreo run on wrapped single-function files in its original artifact "
            "environment; recall = fraction of our labelled clone pairs reported; "
            "precision judged only on predictions inside the labelled pair universe.",
}
os.makedirs("results/oreo_bcb", exist_ok=True)
with open("results/oreo_bcb/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402
save_run_meta("results/oreo_bcb")
