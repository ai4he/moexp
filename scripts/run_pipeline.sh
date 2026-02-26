#!/bin/bash
# MathScy Full Pipeline Orchestrator
# Run on GPU machine after gpu_setup.sh
set -eo pipefail

BASE_DIR="/scratch/ctoxtli/moexp"
LOGS_DIR="$BASE_DIR/logs"
SCRIPTS_DIR="$BASE_DIR/scripts"
CONFIGS_DIR="$BASE_DIR/configs"
mkdir -p "$LOGS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_GPUS=${NUM_GPUS:-8}
DS_CONFIG="${DS_CONFIG:-$CONFIGS_DIR/deepspeed_zero2.json}"

echo "=== MathScy Pipeline Started at $(date) ==="
echo "GPUs: $NUM_GPUS, DeepSpeed config: $DS_CONFIG"

# Phase 1: MoE Domain Expert Training
echo ""
echo "=== Phase 1: MoE Domain Expert Training ==="
echo "Training all domain experts via Branch-Train-Mix"

deepspeed --num_gpus="$NUM_GPUS" "$SCRIPTS_DIR/train_moe.py" \
    --mode branch \
    --deepspeed "$DS_CONFIG" \
    2>&1 | tee "$LOGS_DIR/moe_training_${TIMESTAMP}.log"

echo "Phase 1 complete at $(date)"

# Phase 2: STP Conjecture Generation (requires notebook conversion for now)
echo ""
echo "=== Phase 2: Self-Play Theorem Proving ==="
echo "Note: STP requires GPU inference. Running notebook as script."

jupyter nbconvert --to script --stdout "$BASE_DIR/notebooks/04_conjecture_generation.ipynb" 2>/dev/null \
    | grep -v 'get_ipython\|%matplotlib\|IPython' \
    > "$SCRIPTS_DIR/_04_conjecture_generation.py"
python3 "$SCRIPTS_DIR/_04_conjecture_generation.py" \
    2>&1 | tee "$LOGS_DIR/stp_${TIMESTAMP}.log"

echo "Phase 2 complete at $(date)"

# Phase 3: Lean 4 Verification
echo ""
echo "=== Phase 3: Lean 4 Autoformalization & Verification ==="

jupyter nbconvert --to script --stdout "$BASE_DIR/notebooks/03_lean4_autoformalization.ipynb" 2>/dev/null \
    | grep -v 'get_ipython\|%matplotlib\|IPython' \
    > "$SCRIPTS_DIR/_03_lean4_autoformalization.py"
python3 "$SCRIPTS_DIR/_03_lean4_autoformalization.py" \
    2>&1 | tee "$LOGS_DIR/lean4_${TIMESTAMP}.log"

echo "Phase 3 complete at $(date)"

# Cleanup temp scripts
rm -f "$SCRIPTS_DIR/_04_conjecture_generation.py" "$SCRIPTS_DIR/_03_lean4_autoformalization.py"

echo ""
echo "=== MathScy Pipeline Complete ==="
echo "Results in: $BASE_DIR/results/"
echo "Logs in: $LOGS_DIR/"
