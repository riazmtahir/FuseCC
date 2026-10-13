#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# RQ1 full runs, sequential (both stages share the GPU for XGB/CatBoost).
set -e
# GPU libs (XGB-cuda/CatBoost) leave spinning OpenMP threads that slow later
# sklearn fits up to 100x in the same process; PASSIVE wait fixes it.
export OMP_WAIT_POLICY=PASSIVE
PY="${PYTHON:-python}"
cd "$ROOT"
echo "=== RQ1 BCB start $(date) ==="
$PY tools/rq1_classifier_comparison.py --dataset bcb
echo "=== RQ1 GCJ start $(date) ==="
$PY tools/rq1_classifier_comparison.py --dataset gcj
echo "=== RQ1 all done $(date) ==="
