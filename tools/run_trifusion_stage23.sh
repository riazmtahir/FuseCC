#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT="${FUSECC_EXTERNAL:-$(dirname "$ROOT")}"
# TriFusion-LLM re-run, stages 2-3 (CPU): wait for split_dataset.py ->
# cap test_real at 100k (stratified, seed 12345; full set kept) ->
# feature extraction -> LightGBM ML classifier (ml_fusion_score vectors).
# Launch: setsid nohup bash tools/run_trifusion_stage23.sh > logs/trifusion_stage23.log 2>&1 &
set -euo pipefail
PY="${PYTHON:-python}"
REPO=$EXT/baselines/TriFusion-LLM
DATA=$REPO/experiment_data_42
TOOLS=$ROOT/tools
SPLITLOG=$ROOT/logs/trifusion_split.log
trap 'echo "=== STAGE23: FAILED at line $LINENO ($(date)) ==="' ERR

echo "=== STAGE23: waiting for split ($(date)) ==="
while pgrep -f "split_dataset.py" >/dev/null; do sleep 60; done
echo "=== STAGE23: split exited ($(date)) ==="

if ! grep -q "Done! Total time" "$SPLITLOG"; then
  echo "=== STAGE23: FAILED split log has no Done marker ==="
  exit 1
fi
for f in train.jsonl val.jsonl test_real.jsonl; do
  if [ ! -s "$DATA/$f" ]; then
    echo "=== STAGE23: FAILED missing/empty $DATA/$f ==="
    exit 1
  fi
done
wc -l "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/test_real.jsonl"

NTEST=$(wc -l < "$DATA/test_real.jsonl")
if [ "$NTEST" -gt 100000 ]; then
  echo "test_real has $NTEST lines -> capping at 100k (full set: test_real_full.jsonl)"
  mv "$DATA/test_real.jsonl" "$DATA/test_real_full.jsonl"
  $PY "$TOOLS/subsample_jsonl_stratified.py" \
    --input "$DATA/test_real_full.jsonl" --output "$DATA/test_real.jsonl" \
    --total 100000 --seed 12345
fi
echo "=== STAGE23: subsample done ($(date)) ==="

cd "$REPO/dataprocess"
$PY extract_features_for_machine_learning.py \
  --input_dir ../experiment_data_42 --output_dir ../experiment_data_42 \
  --train_pattern train.jsonl --val_pattern val.jsonl --test_pattern test_real.jsonl
echo "=== STAGE23: features done ($(date)) ==="

$PY ML_clone_classifier.py --mode train \
  --input_dir ../experiment_data_42 --output_dir ../experiment_data_42/ml_outputs/ \
  --train_pattern train.jsonl --val_pattern val.jsonl --test_pattern test_real.jsonl \
  --seed 42
echo "=== STAGE23: ML train done ($(date)) ==="
echo "=== STAGE23: ALL DONE ($(date)) ==="
