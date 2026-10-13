#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT="${FUSECC_EXTERNAL:-$(dirname "$ROOT")}"
# MAGNET BCB — resume an interrupted run from its epoch-7 checkpoint
# (model_7.pth) and train epochs 8 and 9.
#
# PROTOCOL CAVEAT: resuming restarts Adam's moment
# estimates at the epoch-8 boundary, because upstream checkpoints weights only.
# The learning rate is NOT affected — scheduler.step() is commented out upstream,
# so the LR is constant at init_lr for the whole run. The resumed epochs are
# therefore labelled "spliced" and the headline stays the dev-selected epoch,
# which sits at epoch 5 inside the continuous segment.
set -u
REPO=$EXT/baselines/Multigraph_match
PY="${MAGNET_PYTHON:-python}"
LOG=$ROOT/logs/magnet_bcb_resume.log
CKPT="$REPO/out/BCB/multi_graph/checkpoints/graph_match_nerual_network_JavaBigCloneBench_GPU0_19h12m18s_on_Jul_16_2026/model_7.pth"

[ -s "$CKPT" ] || { echo "FATAL: checkpoint missing: $CKPT" | tee -a "$LOG"; exit 1; }

# Idempotent: don't start a second resume if one is already running. Match the
# interpreter path AND the env marker, not bare args (pgrep -f would match this shell).
if pgrep -f "MAGNET_RESUME" >/dev/null 2>&1; then
  echo "resume already running" | tee -a "$LOG"; exit 0
fi

cd "$REPO" || exit 1
{
  echo "=== MAGNET BCB RESUME: epochs 8-9 from attempt 4's model_7.pth ($(date)) ==="
  echo "=== checkpoint: $CKPT"
  echo "=== NOTE: Adam state resets at the epoch-8 boundary (weights-only ckpt)"
} >> "$LOG"

MAGNET_RESUME_FROM="$CKPT" MAGNET_START_EPOCH=8 \
  setsid nohup "$PY" main.py --config configs/Q1/bcb.json --epochs 10 \
  >> "$LOG" 2>&1 &

echo "launched resume, pid $!" | tee -a "$LOG"
