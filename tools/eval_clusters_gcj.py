"""F4 re-runs: evaluate cluster-based tool output (e.g. Deckard) on GCJ.

A file pair is predicted clone iff both files share at least one cluster
(blank-line-separated blocks, `FILE <path>` member lines). Metrics are
computed over our labelled GCJ pair set (dataset/GCJ/pairs_mapped.csv).

Usage: python tools/eval_clusters_gcj.py --clusters <file> --name deckard_gcj
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", required=True)
parser.add_argument("--name", required=True)
args = parser.parse_args()

file_clusters = defaultdict(set)
rx = re.compile(r"FILE\s+(\S+)\s")
block_id, block_files, n_blocks = 0, set(), 0
def flush(files, bid):
    global n_blocks
    if len(files) >= 2:
        n_blocks += 1
        for fn in files:
            file_clusters[fn].add(bid)
with open(args.clusters, errors="ignore") as f:
    for line in f:
        m = rx.search(line)
        if m:
            block_files.add(os.path.basename(m.group(1)))
        elif line.strip() == "" and block_files:
            flush(block_files, block_id)
            block_id += 1
            block_files = set()
flush(block_files, block_id)
print(f"{n_blocks} cross-file clusters, {len(file_clusters)} files in them")

df = pd.read_csv("dataset/GCJ/pairs_mapped.csv")
ba = df["path_a"].map(os.path.basename)
bb = df["path_b"].map(os.path.basename)

def shares(a, b):
    ca = file_clusters.get(a)
    if not ca:
        return False
    cb = file_clusters.get(b)
    return bool(cb) and not ca.isdisjoint(cb)

pred = [shares(a, b) for a, b in zip(ba, bb)]
y = df["label"].astype(int).values
summary = {
    "n_pairs": int(len(y)),
    "predicted_positive": int(sum(pred)),
    "precision": float(precision_score(y, pred)),
    "recall": float(recall_score(y, pred)),
    "f1": float(f1_score(y, pred)),
}
out_dir = f"results/baselines_rerun/{args.name}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402
save_run_meta(out_dir)
