#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# MAP@R re-rank with jina-code as stage-1 (tests if metrics help on top of jina retrieval).
# stage-1 = jinacode_sym cosine; re-rank with the jina-trained 148 model (xgb_148_jina).
set -u
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
for spec in "java250 499" "poj104 699"; do
  set -- $spec; ds=$1; K=$2
  echo "=== $ds jina re-rank K=$K start $(date) ==="
  $PY -u tools/eval_concord_mapr.py --dataset $ds --scorer fusecc \
    --tag jinacode_sym --model-suffix _jina --queries 500 --topk $K --workers 10 \
    --label fusecc_jina_rerank 2>&1 | grep -E "MAP@R|scored|stage-1"
  echo "=== $ds done $(date) ==="
done
echo "=== JINA RERANK DONE $(date) ==="
