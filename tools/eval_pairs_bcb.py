"""F4 baseline re-runs: evaluate ANY predicted clone-pair set against our
canonical BCB pairs (generalizes tools/eval_oreo_bcb.py).

Input: a CSV/whitespace file of predicted pairs of BCB FUNCTION ids
(two integer columns; extra columns ignored), e.g. produced by a converter
from the tool's native output.

Reports per-type recall + labelled precision, writes
results/baselines_rerun/<name>/summary.json.

Usage: python tools/eval_pairs_bcb.py --pairs pred.csv --name scc_bcb [--sep ,]
"""
import argparse
import json
import os
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--pairs", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--sep", default=",")
parser.add_argument("--min-tokens", type=int, default=0,
                    help="also report within-limits recall on pairs where BOTH "
                         "functions have >= N tokens (BigCloneEval convention "
                         "for tools with a size floor)")
args = parser.parse_args()

pred_pairs = set()
with open(args.pairs, errors="ignore") as f:
    for line in f:
        parts = line.strip().split(args.sep)
        if len(parts) < 2:
            continue
        try:
            a, b = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if a != b:
            pred_pairs.add((min(a, b), max(a, b)))
print(f"{len(pred_pairs)} unique predicted id pairs")

canon = pd.read_csv("dataset/BCB/pairs_canonical.csv")
canon["lo"] = canon[["id_a", "id_b"]].min(axis=1)
canon["hi"] = canon[["id_a", "id_b"]].max(axis=1)
canon["key"] = list(zip(canon["lo"], canon["hi"]))
canon["predicted"] = canon["key"].isin(pred_pairs)

types = ["T1", "T2", "VST3", "ST3", "MT3", "WT3/T4"]
recall = {t: {"recall": float(canon[canon.clone_type == t].predicted.mean()),
              "n": int((canon.clone_type == t).sum())} for t in types}
clones = canon[canon.label == 1]
recall["T1-T3_overall"] = {
    "recall": float(clones[clones.clone_type != "WT3/T4"].predicted.mean()),
    "n": int((clones.clone_type != "WT3/T4").sum()),
}
labelled = set(canon["key"])
in_uni = [p for p in pred_pairs if p in labelled]
clone_keys = set(canon.loc[canon.label == 1, "key"])
tp = sum(1 for p in in_uni if p in clone_keys)

summary = {
    "unique_pred_pairs": len(pred_pairs),
    "pred_in_labelled_universe": len(in_uni),
    "labelled_precision": tp / len(in_uni) if in_uni else float("nan"),
    "per_type_recall": recall,
}

if args.min_tokens:
    tok = pd.read_csv("dataset/BCB/function_token_counts.csv")
    sizes = dict(zip(tok.function_id, tok.tokens))
    big = canon[(canon.id_a.map(sizes).fillna(0) >= args.min_tokens)
                & (canon.id_b.map(sizes).fillna(0) >= args.min_tokens)]
    wl = {t: {"recall": float(big[big.clone_type == t].predicted.mean()),
              "n": int((big.clone_type == t).sum())} for t in types
          if (big.clone_type == t).any()}
    summary["within_limits"] = {"min_tokens": args.min_tokens,
                                "per_type_recall": wl}
out_dir = f"results/baselines_rerun/{args.name}"
os.makedirs(out_dir, exist_ok=True)
with open(f"{out_dir}/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps(summary, indent=2))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_meta import save_run_meta  # noqa: E402
save_run_meta(out_dir)
