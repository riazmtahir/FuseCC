"""F4 re-runs: evaluate Deckard clusters against canonical BCB pairs.

Deckard cluster file lines: `<cluster_id>\tdist:<d>\tFILE <path> LINE:a:b ...`
A pair is predicted clone iff both functions appear in at least one common
cluster (membership test — avoids enumerating O(n^2) pairs of mega-clusters).
Precision is therefore judged on the labelled pair universe only.

Usage: python tools/eval_deckard_bcb.py --clusters <file> --name deckard_bcb [--min-tokens 50]
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--min-tokens", type=int, default=0)
args = parser.parse_args()

fid_clusters = defaultdict(set)
rx = re.compile(r"FILE\s+(\S+)\s")
n_lines, block_id, block_fids, n_blocks = 0, 0, set(), 0
def flush(fids, bid):
    global n_blocks
    if len(fids) >= 2:  # cross-file clusters only
        n_blocks += 1
        for fid in fids:
            fid_clusters[fid].add(bid)
with open(args.clusters, errors="ignore") as f:
    for line in f:
        m = rx.search(line)
        if m:
            n_lines += 1
            base = os.path.basename(m.group(1))
            if base.endswith(".java"):
                block_fids.add(int(base[:-5]))
        elif line.strip() == "" and block_fids:
            flush(block_fids, block_id)
            block_id += 1
            block_fids = set()
flush(block_fids, block_id)
print(f"{n_lines} member lines, {n_blocks} cross-file clusters, "
      f"{len(fid_clusters)} distinct functions in them")

canon = pd.read_csv("dataset/BCB/pairs_canonical.csv")

def shares_cluster(row):
    ca = fid_clusters.get(row.id_a)
    if not ca:
        return False
    cb = fid_clusters.get(row.id_b)
    return bool(cb) and not ca.isdisjoint(cb)

canon["predicted"] = canon.apply(shares_cluster, axis=1)

types = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]
recall = {t: {"recall": float(canon[canon.clone_type == t].predicted.mean()),
              "n": int((canon.clone_type == t).sum())} for t in types}
clones = canon[canon.label == 1]
recall["T1-T3_overall"] = {
    "recall": float(clones[clones.clone_type != "WT3/T4"].predicted.mean()),
    "n": int((clones.clone_type != "WT3/T4").sum()),
}
pred = canon[canon.predicted]
summary = {
    "cluster_member_lines": n_lines,
    "labelled_pairs_predicted": int(len(pred)),
    "labelled_precision": float(pred.label.mean()) if len(pred) else float("nan"),
    "per_type_recall": recall,
    "note": "precision judged on labelled pair universe (shared-cluster membership test)",
}
if args.min_tokens:
    tok = pd.read_csv("dataset/BCB/function_token_counts.csv")
    sizes = dict(zip(tok.function_id, tok.tokens))
    big = canon[(canon.id_a.map(sizes).fillna(0) >= args.min_tokens)
                & (canon.id_b.map(sizes).fillna(0) >= args.min_tokens)]
    summary["within_limits"] = {
        "min_tokens": args.min_tokens,
        "per_type_recall": {t: {"recall": float(big[big.clone_type == t].predicted.mean()),
                                "n": int((big.clone_type == t).sum())}
                            for t in types if (big.clone_type == t).any()},
    }
out_dir = f"results/baselines_rerun/{args.name}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402
save_run_meta(out_dir)
