#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Measure the MAP@R gain from a larger stage-1 candidate pool (K=5R).
# The ceiling diagnosis showed the re-ranker is near-oracle, so lifting K should lift
# realised MAP@R. 300 queries keeps 300*5R pairs tractable at ~120 pairs/s.
set -u
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
# Java250: 5R = 1495 ; POJ-104: 5R = 2495
$PY -u tools/eval_concord_mapr.py --dataset java250 --scorer fusecc --tag unixcoder --queries 300 --topk 1495 --workers 10 --label fusecc_rerank_bigK
$PY -u tools/eval_concord_mapr.py --dataset poj104  --scorer fusecc --tag unixcoder --queries 300 --topk 2495 --workers 10 --label fusecc_rerank_bigK
echo "=== bigK MAP@R done $(date) ==="
