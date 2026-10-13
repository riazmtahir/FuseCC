"""Convert SourcererCC output pairs (project_id,block_id,project_id,block_id)
to BCB function-id pairs using Oreo's mlcc_input.file bookkeeping
(field 2 = <function_id>.java, last comma field before @#@ = block_id).

Usage: python tools/convert_scc_pairs.py --scc-output <concatenated clones file> \
           --bookkeeping <OREO_DIR>/python_scripts/1_metric_output/mlcc_input.file \
           --out results/baselines_rerun/scc_bcb/pred_pairs.csv
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scc-output", required=True)
parser.add_argument("--bookkeeping", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

block2fid = {}
with open(args.bookkeeping, errors="ignore") as f:
    for line in f:
        head = line.split("@#@", 1)[0]
        parts = head.split(",")
        if len(parts) < 3:
            continue
        fname = parts[1]
        block_id = parts[-1].strip()
        if fname.endswith(".java") and block_id.isdigit():
            block2fid[block_id] = fname[:-5]
print(f"bookkeeping: {len(block2fid)} block ids mapped")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
n_in, n_out, missing = 0, 0, 0
with open(args.scc_output, errors="ignore") as fin, open(args.out, "w") as fout:
    for line in fin:
        parts = line.strip().split(",")
        if len(parts) != 4:
            continue
        n_in += 1
        fa, fb = block2fid.get(parts[1]), block2fid.get(parts[3])
        if fa is None or fb is None:
            missing += 1
            continue
        fout.write(f"{fa},{fb}\n")
        n_out += 1
print(f"pairs in {n_in}, written {n_out}, unmapped {missing} -> {args.out}")
