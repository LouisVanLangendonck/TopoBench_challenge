#!/bin/bash

# =============================================================================
# Parallel downstream evaluation script
# =============================================================================
# This script runs downstream evaluations in parallel across multiple GPUs
# by launching separate Python processes on different devices.
#
# Usage:
#   bash scripts/run_downstream_eval_parallel.sh \
#       <wandb_project> \
#       [downstream_project] \
#       [max_runs]
#
# Example:
#   bash scripts/run_downstream_eval_parallel.sh \
#       "entity/graphmae_pretraining" \
#       "graphmae_pretraining_downstream_eval" \
#       10
# =============================================================================

# Arguments
WANDB_PROJECT=$1
DOWNSTREAM_PROJECT=${2:-"downstream_eval"}
MAX_RUNS=${3:-""}

if [ -z "$WANDB_PROJECT" ]; then
    echo "Error: Please provide wandb project path"
    echo "Usage: bash scripts/run_downstream_eval_parallel.sh <wandb_project> [downstream_project] [max_runs]"
    exit 1
fi

echo "=============================================================================="
echo "PARALLEL DOWNSTREAM EVALUATION"
echo "=============================================================================="
echo "Source project: $WANDB_PROJECT"
echo "Downstream project: $DOWNSTREAM_PROJECT"
echo "Max runs: ${MAX_RUNS:-all}"
echo "Devices: 0, 1, 2, 3"
echo "=============================================================================="

# Build the base command
CMD="python3 tutorials/run_downstream_eval_grid_v2.py \
    --wandb_project $WANDB_PROJECT \
    --downstream_project $DOWNSTREAM_PROJECT \
    --num_workers 1"

# Add max_runs if specified
if [ -n "$MAX_RUNS" ]; then
    CMD="$CMD --max_runs $MAX_RUNS"
fi

# Device 0
echo "Starting worker on GPU 0..."
$CMD --devices 0 --seed 42 > logs/downstream_eval_gpu0.log 2>&1 &
PID0=$!

sleep 5

# Device 1
echo "Starting worker on GPU 1..."
$CMD --devices 1 --seed 43 > logs/downstream_eval_gpu1.log 2>&1 &
PID1=$!

sleep 5

# Device 2
echo "Starting worker on GPU 2..."
$CMD --devices 2 --seed 44 > logs/downstream_eval_gpu2.log 2>&1 &
PID2=$!

sleep 5

# Device 3
echo "Starting worker on GPU 3..."
$CMD --devices 3 --seed 45 > logs/downstream_eval_gpu3.log 2>&1 &
PID3=$!

echo ""
echo "=============================================================================="
echo "All 4 workers started!"
echo "=============================================================================="
echo "GPU 0: PID $PID0 (log: logs/downstream_eval_gpu0.log)"
echo "GPU 1: PID $PID1 (log: logs/downstream_eval_gpu1.log)"
echo "GPU 2: PID $PID2 (log: logs/downstream_eval_gpu2.log)"
echo "GPU 3: PID $PID3 (log: logs/downstream_eval_gpu3.log)"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/downstream_eval_gpu*.log"
echo ""
echo "Waiting for all workers to complete..."
echo "=============================================================================="

# Wait for all background jobs
wait $PID0
wait $PID1
wait $PID2
wait $PID3

echo ""
echo "=============================================================================="
echo "ALL DOWNSTREAM EVALUATIONS COMPLETE"
echo "=============================================================================="
echo "Check individual logs for results:"
echo "  logs/downstream_eval_gpu0.log"
echo "  logs/downstream_eval_gpu1.log"
echo "  logs/downstream_eval_gpu2.log"
echo "  logs/downstream_eval_gpu3.log"
echo "=============================================================================="

