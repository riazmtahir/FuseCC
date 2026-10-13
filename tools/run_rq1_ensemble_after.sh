#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
# Waits for the RQ1 campaign to finish, then runs the top-3 ensemble on the
# same folds for both datasets.
set -e
export OMP_WAIT_POLICY=PASSIVE
PY="${PYTHON:-python}"
cd "$ROOT"
until grep -q "RQ1 all done" logs/rq1_full.log 2>/dev/null; do sleep 120; done
echo "=== RQ1 ensemble BCB start $(date) ==="
$PY tools/rq1_ensemble.py --dataset bcb
echo "=== RQ1 ensemble GCJ start $(date) ==="
$PY tools/rq1_ensemble.py --dataset gcj
echo "=== RQ1 ensemble all done $(date) ==="
