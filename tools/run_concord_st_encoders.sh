#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Embed both ConCord test splits with the 2025/2026 code encoders (test split only;
# MAP@R needs test). Sequential so the 16 GB GPU isn't oversubscribed.
set -u
cd "$ROOT" || exit 1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python}"
for ds in poj104 java250; do
  for m in coderankembed jinacode qwen3emb; do
    echo "=== $ds / $m start $(date) ==="
    $PY -u tools/embed_concord_st.py --dataset $ds --model $m 2>&1 \
      | grep -viE "Batches|it/s|Loading weights"
    echo "=== $ds / $m done $(date) ==="
  done
done
echo "=== ALL ST ENCODERS DONE $(date) ==="
