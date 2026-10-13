#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT="${FUSECC_EXTERNAL:-$(dirname "$ROOT")}"
# FA-AST+GMN BCB re-run.
#
# Pairs are IDENTICAL to MAGNET BCB's (BCB/{traindata,devdata,testdata}.txt
# regenerated from MAGNET's clone_labels.txt by tools/prep_faast_bcb_from_magnet.py;
# stock upstream files preserved as BCB/*_upstream_stock.txt, --data_setting 0
# reads ours). Config mirrors our FA-AST GCJ re-run: astandnext, 10 epochs,
# per-epoch dev+test dumps -> dev-selected epoch.
#
# Idempotent: exits if the final epoch dump exists or run_bcb.py is running.
# Launch: setsid nohup bash tools/run_faast_bcb.sh > logs/faast_bcb_launch.log 2>&1 &
set -uo pipefail
PY="${PYTHON:-python}"
REPO=$EXT/baselines/graphmatch_clone
LOG=$ROOT/logs/faast_bcb.log

[ -s "$REPO/gmnbcbresult/astandnext_epoch_10" ] && { echo "FA-AST BCB already complete"; exit 0; }
pgrep -f "$PY run_bcb.py" >/dev/null && { echo "run_bcb.py already running"; exit 0; }

cd "$REPO" || exit 1
mkdir -p gmnbcbresult
echo "=== FA-AST BCB: launching ($(date)) ==="
# NB do NOT pass --num_epochs: run_bcb.py declares it without type=int, so a
# command-line value arrives as a string and trange() dies with
# "TypeError: 'str' object cannot be interpreted as an integer".
# Its argparse default is already the int 10 = the budget we want (same as the
# GCJ re-run). Same trap applies to --batch_size / --num_layers / --lr.
$PY run_bcb.py --dataset bcb --graphmode astandnext --data_setting 0 >> "$LOG" 2>&1
echo "=== FA-AST BCB: finished ($(date)) ==="
