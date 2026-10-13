#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Encoder baselines for POJ-104 / Java250: CodeBERT + GraphCodeBERT.
#
# These are the right baselines for these benchmarks — they are exactly the encoders the
# CodeXGLUE POJ-104 leaderboard reports MAP@R for, and both are multilingual so the same
# row covers Java250 (Java) and POJ-104 (C/C++). UniXcoder is not repeated here: the
# FuseCC pipeline already embeds with it, and that same file is its baseline row.
#
# Waits for any running embedding job first so the GPU is not oversubscribed.
set -u
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"

while pgrep -f "tools/embed_concord.py" >/dev/null; do sleep 120; done

for ds in java250 poj104; do
  for spec in "codebert microsoft/codebert-base" "graphcodebert microsoft/graphcodebert-base"; do
    set -- $spec
    tag=$1; model=$2
    if [ -f "dataset/concord/$ds/embeddings_${tag}_test.npy" ]; then
      echo "=== $ds/$tag already embedded, skipping ==="
      continue
    fi
    echo "=== $ds / $tag start $(date) ==="
    # only the test split is needed for MAP@R; train/valid would only be used if we
    # also trained a pair-classifier on this encoder, which we do not for baselines
    $PY -u tools/embed_concord.py --dataset "$ds" --model "$model" --tag "$tag" \
        --splits test
    echo "=== $ds / $tag done $(date) ==="
  done
done
echo "=== all encoder baselines embedded $(date) ==="
