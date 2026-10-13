#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Extract t1-t19 for both ConCord pair sets, sequentially.
#
# Sequential on purpose, to bound CPU load (9 worker processes).
#
# poj104 gets --cpp-keywords (it is C/C++); java250 must NOT, so its features stay
# bit-identical in definition to the BCB/GCJ runs.
set -u
cd "$ROOT" || exit 1
PY="${PYTHON:-python}"
W=9

echo "=== java250 extraction start $(date) ==="
$PY -u tools/extract_concord_features.py --dataset java250 --workers $W
echo "=== java250 done $(date) ==="

echo "=== poj104 extraction start $(date) ==="
$PY -u tools/extract_concord_features.py --dataset poj104 --workers $W --cpp-keywords
echo "=== poj104 done $(date) ==="
