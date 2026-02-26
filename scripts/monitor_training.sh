#!/bin/bash
# Monitor MoE training status every 10 minutes
LOG_DIR="/scratch/ctoxtli/moexp/logs"
MODELS_DIR="/scratch/ctoxtli/moexp/models"

while true; do
    echo ""
    echo "=========================================="
    echo "MathScy Training Status - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # GPU status
    echo ""
    echo "--- GPU Status ---"
    nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null

    # Training progress from latest log
    echo ""
    echo "--- Latest Training Progress ---"
    LATEST_LOG=$(ls -t "$LOG_DIR"/train_branch_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_LOG" ]; then
        # Show last progress line
        grep -E "^\s+[0-9]+%|Expert|Starting|saved" "$LATEST_LOG" | tail -5
    fi

    # Expert completion status
    echo ""
    echo "--- Expert Status ---"
    for domain in algebraic_geometry discrete_math number_theory analysis algebra geometry_topology probability_statistics computational; do
        DIR="$MODELS_DIR/expert_${domain}"
        if [ -d "$DIR/final" ]; then
            echo "  $domain: COMPLETE"
        elif [ -d "$DIR" ]; then
            STEPS=$(ls "$DIR"/checkpoint-* 2>/dev/null | wc -l)
            echo "  $domain: IN PROGRESS (checkpoints: $STEPS)"
        else
            echo "  $domain: PENDING"
        fi
    done

    echo "=========================================="
    sleep 600  # 10 minutes
done
