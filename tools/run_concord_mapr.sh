#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# All MAP@R rows for POJ-104 + Java250, the official metric (comparable to CONCORD 91.5/86.5).
#
# Stage A: embedding-cosine MAP@R, EXACT (every test sample is a query). One row per
#          encoder — UniXcoder, CodeBERT, GraphCodeBERT. These are the encoders the
#          CodeXGLUE POJ-104 leaderboard reports MAP@R for.
# Stage B: FuseCC two-stage MAP@R — UniXcoder retrieves top-K, the trained 148-feature
#          classifier re-ranks. Query-sampled (the 19-metric scorer is ~125 ms/pair, so
#          all-pairs is infeasible: POJ-104 test alone is ~72M pairs). --queries 1500.
set -u
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

# idempotent: skip a row whose label is already present in mapr.json
have () { $PY -c "import json,sys;d=json.load(open('results/concord/$1/mapr.json'))if __import__('os').path.exists('results/concord/$1/mapr.json')else{};sys.exit(0 if '$2' in d else 1)"; }

for ds in java250 poj104; do
  echo "############## $ds ##############"
  # Stage A — exact embedding baselines
  for tag in unixcoder codebert graphcodebert; do
    if have $ds $tag; then echo "$ds/$tag already done, skip"; else
      $PY -u tools/eval_concord_mapr.py --dataset $ds --scorer embedding --tag $tag --label $tag
    fi
  done
  # Stage B — FuseCC re-rank (sampled queries)
  if have $ds fusecc_rerank; then echo "$ds/fusecc already done, skip"; else
    $PY -u tools/eval_concord_mapr.py --dataset $ds --scorer fusecc --tag unixcoder --queries 500 --workers 10 --label fusecc_rerank
  fi
done
echo "=== all MAP@R done $(date) ==="
