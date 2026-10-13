"""F4 re-runs: evaluate predicted file-pair output (MSCCD, StoneDetector) on GCJ.

Input: CSV of two file basenames per line (extra columns ignored). A labelled
pair (dataset/GCJ/pairs_mapped.csv) is predicted clone iff its unordered
basename pair is in the predicted set. Writes
results/baselines_rerun/<name>/summary.json (+ run_meta.json).

Usage: python tools/eval_pairs_gcj.py --pairs pred.csv --name msccd_gcj [--sep ,]
"""
import argparse
import json
import os
import sys

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--sep", default=",")
args = parser.parse_args()

pred = set()
with open(args.pairs, errors="ignore") as f:
    for line in f:
        parts = line.strip().split(args.sep)
        if len(parts) < 2:
            continue
        a, b = os.path.basename(parts[0]), os.path.basename(parts[1])
        if a != b:
            pred.add(tuple(sorted((a, b))))
print(f"{len(pred)} unique predicted file pairs")

pairs = pd.read_csv("dataset/GCJ/pairs_mapped.csv")
pairs["a"] = pairs["path_a"].map(os.path.basename)
pairs["b"] = pairs["path_b"].map(os.path.basename)
pairs["key"] = [tuple(sorted(k)) for k in zip(pairs["a"], pairs["b"])]
pairs["predicted"] = pairs["key"].isin(pred).astype(int)

y, yhat = pairs["label"].values, pairs["predicted"].values
summary = {
    "n_pairs": int(len(pairs)),
    "predicted_positive": int(yhat.sum()),
    "precision": float(precision_score(y, yhat)),
    "recall": float(recall_score(y, yhat)),
    "f1": float(f1_score(y, yhat)),
}
out_dir = f"results/baselines_rerun/{args.name}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta
save_run_meta(out_dir)
