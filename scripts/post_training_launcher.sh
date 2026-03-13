#!/bin/bash
# Post-training launcher:
#   1. Waits for main v3 training (PID 4144812) to finish
#   2. Runs analysis rerun with lr=5e-6
#   3. Runs full skip-existing pass to rebuild complete registry
#   4. Runs router assembly
cd /scratch/ctoxtli/moexp
source activate moexp 2>/dev/null || conda activate moexp 2>/dev/null || true
export HF_HOME=/scratch/ctoxtli/cache

MAIN_PID=4144812
LOG_DIR=/scratch/ctoxtli/moexp/logs

echo "[$(date)] post_training_launcher started — watching PID $MAIN_PID"

# Step 1: Wait for main training
while kill -0 $MAIN_PID 2>/dev/null; do
    sleep 30
done
echo "[$(date)] Main training finished."

# Verify shared expert saved
if [ ! -d "models/v3_moe/expert_shared/final" ]; then
    echo "[$(date)] WARNING: expert_shared/final not found"
fi

# Step 2: Analysis rerun with lower LR
echo "[$(date)] Starting analysis rerun (lr=5e-6)..."
python3 -u scripts/train_moe_v3.py \
    --domains analysis \
    --no-skip \
    --lr 5e-6 \
    2>&1 | tee $LOG_DIR/train_v3_analysis_rerun.log
echo "[$(date)] Analysis rerun finished."

if [ ! -d "models/v3_moe/expert_analysis/final" ]; then
    echo "[$(date)] ERROR: analysis adapter not saved — check logs"
    exit 1
fi

# Step 3: Rebuild complete registry (skip-existing passes all 8, just saves registry)
echo "[$(date)] Rebuilding complete v3 registry..."
python3 -u scripts/train_moe_v3.py \
    --skip-existing \
    2>&1 | tee $LOG_DIR/train_v3_registry_rebuild.log
echo "[$(date)] Registry rebuilt."

# Step 4: Router assembly
echo "[$(date)] Starting router assembly..."
python3 -u scripts/assemble_moe.py \
    2>&1 | tee $LOG_DIR/assemble_v3_router.log
echo "[$(date)] All post-training steps done!"
