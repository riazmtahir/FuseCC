#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT="${FUSECC_EXTERNAL:-$(dirname "$ROOT")}"
# TriFusion-LLM re-run, stages 4-5 (GPU): wait for stage23's ml_outputs AND for
# the MAGNET GCJ training (PID passed as $1, default 405187) to release the GPU,
# then train the CodeBERT fusion model (reconstructed train.py — upstream ships
# it empty) and generate test predictions + arbitration inputs via tset.py.
# The DeepSeek arbitration stage is NOT auto-run.
# Launch: setsid nohup bash tools/run_trifusion_stage45.sh > logs/trifusion_stage45.log 2>&1 &
set -euo pipefail
PY="${PYTHON:-python}"
REPO=$EXT/baselines/TriFusion-LLM
DATA=$REPO/experiment_data_42
MAGNET_PID=${1:-405187}
trap 'echo "=== STAGE45: FAILED at line $LINENO ($(date)) ==="' ERR

echo "=== STAGE45: waiting for ml_outputs ($(date)) ==="
until [ -s "$DATA/ml_outputs/test/test_with_ml_score.jsonl" ] \
   && [ -s "$DATA/ml_outputs/train/train_with_ml_score.jsonl" ] \
   && [ -s "$DATA/ml_outputs/val/val_with_ml_score.jsonl" ] \
   && ! pgrep -f "ML_clone_classifier.py" >/dev/null; do
  sleep 120
done
echo "=== STAGE45: ml_outputs ready ($(date)) ==="

while kill -0 "$MAGNET_PID" 2>/dev/null; do sleep 300; done
echo "=== STAGE45: GPU free, MAGNET exited ($(date)) ==="

cd "$REPO"
export PYTHONPATH="$REPO"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

$PY main.py \
  --train_file ./experiment_data_42/ml_outputs/train/train_with_ml_score.jsonl \
  --val_file ./experiment_data_42/ml_outputs/val/val_with_ml_score.jsonl \
  --test_file ./experiment_data_42/ml_outputs/test/test_with_ml_score.jsonl \
  --model_name_or_path microsoft/codebert-base \
  --seed 42
echo "=== STAGE45: fusion training done ($(date)) ==="

$PY tset.py \
  --model_path microsoft/codebert-base \
  --pth_file save_model/config/seed_42/best_model_seed_42.pth \
  --input_dir ./experiment_data_42/ml_outputs/test \
  --test_pattern test_with_ml_score.jsonl \
  --output_dir ./my_predictions/run_seed42 \
  --eval_batch_size 128 --n_jobs 24 --seed 42
echo "=== STAGE45: predictions done ($(date)) ==="
echo "=== STAGE45: ALL DONE ($(date)) ==="
