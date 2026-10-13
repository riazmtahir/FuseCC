#!/bin/bash
ROOT="${FUSECC_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
EXT="${FUSECC_EXTERNAL:-$(dirname "$ROOT")}"
# TriFusion-LLM re-run, final stage: DeepSeek arbitration on the fusion model's
# low-confidence rare-class predictions (their defaults: confidence in [0,0.6)
# AND pred in {2,3,4,5} -> 1,378 of 934,258 full-test samples).
# Model: deepseek-chat = V3.2 (the paper's arbitrator).
# Launch: setsid nohup bash tools/run_trifusion_arbitration.sh > logs/trifusion_arbitration.log 2>&1 &
set -euo pipefail
PY="${PYTHON:-python}"
REPO=$EXT/baselines/TriFusion-LLM
PRED=$REPO/my_predictions/run_seed42
trap 'echo "=== ARBITRATION: FAILED at line $LINENO ($(date)) ==="' ERR

echo "=== ARBITRATION: start ($(date)) ==="
cd "$PRED"
$PY "$REPO/my_predictions/deepseek.py" \
  --api_key "$(cat "${DEEPSEEK_API_KEY_FILE:-$HOME/.deepseek_api_key}")" \
  --model deepseek-chat \
  --base_url https://api.deepseek.com \
  --probs_file "$PRED/test_probability_matrix_with_code.csv" \
  --code_file "$PRED/test_original_code.jsonl" \
  --output_dir "$PRED/deepseek_arbitration_results" \
  --workers 12
echo "=== ARBITRATION: ALL DONE ($(date)) ==="
