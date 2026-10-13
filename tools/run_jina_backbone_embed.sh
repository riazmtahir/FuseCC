#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Embed all splits with the SYMMETRIC jina backbone (code2code_document prompt) so
# FuseCC can be rebuilt on jina-code instead of frozen UniXcoder.
set -u
cd "$ROOT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
for ds in poj104 java250; do
  echo "=== $ds jinacode_sym train,valid,test start $(date) ==="
  $PY -u tools/embed_concord_st.py --dataset $ds --model jinacode_sym --splits train,valid,test 2>&1 | grep -viE "Batches|it/s|Loading weights"
  echo "=== $ds done $(date) ==="
done
echo "=== JINA BACKBONE EMBED DONE $(date) ==="
