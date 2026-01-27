#!/bin/bash

# Auto-generated parallel downstream evaluation script
# Source project: louis-van-langendonck-universitat-polit-cnica-de-catalunya/linkpred_pretraining
# Downstream project: linkpred_pretraining_downstream_eval_full
# Total tasks: 2784

# Create logs directory
mkdir -p logs

# Function to run a task
run_task() {
    local run_id=$1
    local checkpoint=$2
    local mode=$3
    local n_train=$4
    local device=$5
    local seed=$6
    local config_file=$7
    local p_num=$8
    local lr_override=$9

    # Build command
    local cmd="python3 tutorials/run_downstream_eval_single.py"
    cmd="$cmd --wandb_project louis-van-langendonck-universitat-polit-cnica-de-catalunya/linkpred_pretraining"
    cmd="$cmd --run_id $run_id"
    cmd="$cmd --checkpoint $checkpoint"
    cmd="$cmd --mode $mode"
    cmd="$cmd --n_train $n_train"
    cmd="$cmd --device $device"
    cmd="$cmd --seed $seed"
    cmd="$cmd --downstream_project linkpred_pretraining_downstream_eval_full"
    cmd="$cmd --config_json $config_file"
    cmd="$cmd --p_num $p_num"
    
    # Add lr_override if not 'none'
    if [ "$lr_override" != "none" ]; then
        cmd="$cmd --lr_override $lr_override"
    fi
    
    # Run command
    eval $cmd
}

# Export function for parallel
export -f run_task

# Task list (device assignment round-robin)
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 5 none &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 5 none &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 5 none &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 5 none &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 none &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 5 none &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 5 none &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 5 none &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 5 none &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 5 none &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 none &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 5 none &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 5 none &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 5 none &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 5 none &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 5 none &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 none &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 5 none &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 5 none &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 5 none &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 5 none &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 5 none &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 none &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 5 none &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 5 none &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 5 none &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 5 none &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 5 none &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 none &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 5 none &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 5 none &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 5 none &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 5 none &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 none &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 5 none &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 5 none &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 5 none &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 5 none &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 5 none &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 none &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 5 none &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 5 none &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 5 none &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 5 none &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 none &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 5 none &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 5 none &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 5 none &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 5 none &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 none &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 5 none &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 5 none &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 5 none &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 5 none &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 none &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 5 none &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 5 none &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 5 none &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 5 none &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 none &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 5 none &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 5 none &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 5 none &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 5 none &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 none &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 5 none &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 5 none &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 5 none &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 5 none &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 none &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 5 none &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 5 none &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 5 none &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 5 none &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 none &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 5 none &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 5 none &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 5 none &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 5 none &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 none &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 5 none &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 5 none &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 5 none &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 5 none &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 none &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 5 none &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 5 none &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 5 none &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 5 none &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 none &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 5 none &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 5 none &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 5 none &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 5 none &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 none &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 5 none &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 5 none &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 5 none &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 5 none &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 none &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 5 none &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 5 none &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 5 none &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 5 none &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 none &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 5 none &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 5 none &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 5 none &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 5 none &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 none &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 5 none &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 5 none &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 5 none &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 5 none &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 none &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 5 none &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 5 none &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 5 none &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 5 none &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 5 none &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 none &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 5 none &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 5 none &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 5 none &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 5 none &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 5 none &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 none &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 5 none &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 5 none &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 5 none &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 5 none &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 none &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 5 none &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 5 none &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 5 none &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 5 none &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 none &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 5 none &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 5 none &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 5 none &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 5 none &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 none &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 5 none &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 5 none &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 5 none &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 5 none &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 none &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 5 none &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 5 none &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 5 none &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 5 none &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 none &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 5 none &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 5 none &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 5 none &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 5 none &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 none &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 5 none &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 5 none &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 5 none &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 5 none &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 none &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 5 none &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 5 none &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 5 none &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 5 none &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 none &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 5 none &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 5 none &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 5 none &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 5 none &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 5 none &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 none &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 5 none &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 5 none &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 5 none &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 5 none &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 5 none &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 none &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 5 none &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 5 none &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 5 none &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 5 none &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 5 none &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 none &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 5 none &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 5 none &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 5 none &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 5 none &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 5 none &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 none &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 5 none &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 5 none &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 5 none &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 5 none &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 none &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 5 none &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 5 none &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 5 none &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 5 none &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 none &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 5 none &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 5 none &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 5 none &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 5 none &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 none &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 5 none &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 5 none &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 5 none &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 5 none &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 none &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 5 none &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 none &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 5 none &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 none &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 5 none &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 5 none &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 5 none &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 5 none &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 5 none &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 none &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 5 none &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 5 none &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 5 none &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 5 none &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 5 none &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 none &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 5 none &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 5 none &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 5 none &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 5 none &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 none &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 5 none &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 5 none &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 5 none &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 5 none &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 none &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 5 none &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 5 none &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 5 none &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 5 none &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 none &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 5 none &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 5 none &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 5 none &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 5 none &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 none &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 5 none &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 5 none &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 5 none &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 5 none &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 none &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 5 none &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 5 none &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 5 none &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 5 none &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 none &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 5 none &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 5 none &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 5 none &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 5 none &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 none &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 5 none &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 5 none &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 5 none &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 5 none &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 none &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 5 none &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 5 none &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 5 none &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 5 none &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 none &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 5 none &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 5 none &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 5 none &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 5 none &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 none &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 5 none &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 5 none &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 5 none &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 5 none &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 none &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 5 none &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 5 none &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 5 none &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 5 none &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 5 none &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 none &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 5 none &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 5 none &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 5 none &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 5 none &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 none &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 5 none &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 5 none &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 5 none &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 5 none &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 none &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 5 none &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 5 none &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 5 none &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 5 none &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 5 none &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 none &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 5 none &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 5 none &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 5 none &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 5 none &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 none &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 5 none &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 5 none &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 5 none &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 5 none &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 none &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 5 none &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 5 none &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 5 none &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 5 none &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 none &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 5 none &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 5 none &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 5 none &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 5 none &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 5 none &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 none &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 5 none &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 5 none &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 5 none &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 5 none &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 5 none &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 none &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 5 none &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 5 none &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 5 none &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 5 none &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 5 none &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 none &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 5 none &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 5 none &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 5 none &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 5 none &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 none &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 5 none &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 5 none &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 5 none &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 5 none &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 none &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 5 none &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 5 none &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 5 none &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 5 none &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 5 none &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 none &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 5 none &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 5 none &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 5 none &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 5 none &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 none &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 5 none &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 5 none &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 5 none &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 5 none &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 none &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 5 none &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 5 none &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 5 none &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 5 none &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 none &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 5 none &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 5 none &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 5 none &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 5 none &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 none &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 5 none &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 5 none &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 5 none &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 5 none &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 none &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 5 none &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 5 none &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 5 none &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 5 none &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 none &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 5 none &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 5 none &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 5 none &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 5 none &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 none &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 5 none &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 5 none &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 5 none &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 5 none &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 none &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 5 none &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 5 none &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 5 none &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 5 none &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 none &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 5 none &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 5 none &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 5 none &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 5 none &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 none &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 5 none &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 5 none &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 5 none &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 5 none &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 none &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 5 none &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 5 none &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 5 none &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 5 none &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 none &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 5 none &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 5 none &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 5 none &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 5 none &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 5 none &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 none &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 5 none &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 5 none &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 5 none &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 5 none &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 5 none &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 none &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 5 none &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 5 none &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 5 none &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 5 none &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 none &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 5 none &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 5 none &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 5 none &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 5 none &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 none &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 5 none &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 5 none &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 5 none &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 5 none &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 5 none &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 none &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 5 none &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 5 none &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 5 none &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 5 none &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 none &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 5 none &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 5 none &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 5 none &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 5 none &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "scratch" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "scratch" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "scratch" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "scratch" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 5 none &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 none &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 5 none &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 5 none &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 5 none &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 5 none &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 none &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 5 none &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 5 none &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 5 none &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 5 none &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 none &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 5 none &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 5 none &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 5 none &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 5 none &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 none &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 5 none &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 5 none &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 5 none &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 5 none &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 none &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 5 none &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 5 none &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 5 none &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 5 none &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 none &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 5 none &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 5 none &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 5 none &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 5 none &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 none &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 5 none &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 5 none &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 5 none &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 5 none &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 none &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 5 none &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 5 none &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 5 none &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 5 none &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 none &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 5 none &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 5 none &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 5 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 10 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 5 none &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "mlp" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "mlp" 5 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "mlp" 10 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "mlp" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 5 none &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "finetune" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "finetune" 5 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "finetune" 10 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 none &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "finetune" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 5 none &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 1 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 1 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 1 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 10 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_qxz8i55f.json" 5 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_qxz8i55f.json" 10 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_qxz8i55f.json" 5 0.001 &
run_task "qxz8i55f" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2449.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_qxz8i55f.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 1 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 1 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 1 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 10 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_m7zl30ee.json" 5 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_m7zl30ee.json" 10 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_m7zl30ee.json" 5 0.001 &
run_task "m7zl30ee" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2450.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_m7zl30ee.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 1 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 1 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 1 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 10 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_n5z253ke.json" 5 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_n5z253ke.json" 10 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_n5z253ke.json" 5 0.001 &
run_task "n5z253ke" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2451.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n5z253ke.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 1 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 1 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 1 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 10 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_2n4u49hi.json" 5 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_2n4u49hi.json" 10 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_2n4u49hi.json" 5 0.001 &
run_task "2n4u49hi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2452.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_2n4u49hi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 1 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 1 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 1 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 10 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_j9utbre9.json" 5 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_j9utbre9.json" 10 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_j9utbre9.json" 5 0.001 &
run_task "j9utbre9" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2453.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_j9utbre9.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 1 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 1 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 1 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 10 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_58bsrj1m.json" 5 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_58bsrj1m.json" 10 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_58bsrj1m.json" 5 0.001 &
run_task "58bsrj1m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2454.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_58bsrj1m.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 1 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 1 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 1 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 10 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_d8id2o60.json" 5 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_d8id2o60.json" 10 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_d8id2o60.json" 5 0.001 &
run_task "d8id2o60" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2455.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_d8id2o60.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 1 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 1 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 1 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 10 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_447gtp5h.json" 5 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_447gtp5h.json" 10 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_447gtp5h.json" 5 0.001 &
run_task "447gtp5h" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2456.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_447gtp5h.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 1 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 1 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 1 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 10 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_p9yytwaa.json" 5 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_p9yytwaa.json" 10 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_p9yytwaa.json" 5 0.001 &
run_task "p9yytwaa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2457.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_p9yytwaa.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 1 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_cex58n93.json" 1 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_cex58n93.json" 1 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_cex58n93.json" 10 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_cex58n93.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_cex58n93.json" 5 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_cex58n93.json" 10 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_cex58n93.json" 5 0.001 &
run_task "cex58n93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2459.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_cex58n93.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 1 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 1 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 1 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 10 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_jfeoyb2z.json" 5 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_jfeoyb2z.json" 10 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_jfeoyb2z.json" 5 0.001 &
run_task "jfeoyb2z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2458.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_jfeoyb2z.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 1 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 1 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 1 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 10 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_djzo3m6m.json" 5 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_djzo3m6m.json" 10 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_djzo3m6m.json" 5 0.001 &
run_task "djzo3m6m" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2460.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_djzo3m6m.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 1 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 1 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 1 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 10 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_vazmftj5.json" 5 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_vazmftj5.json" 10 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_vazmftj5.json" 5 0.001 &
run_task "vazmftj5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2461.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vazmftj5.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 1 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 1 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 1 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 10 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_b4kgc6ps.json" 5 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_b4kgc6ps.json" 10 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_b4kgc6ps.json" 5 0.001 &
run_task "b4kgc6ps" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2462.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_b4kgc6ps.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 1 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 1 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 1 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 10 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_lmle7zwu.json" 5 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_lmle7zwu.json" 10 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_lmle7zwu.json" 5 0.001 &
run_task "lmle7zwu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2463.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lmle7zwu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 1 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_94upewdy.json" 1 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_94upewdy.json" 1 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_94upewdy.json" 10 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_94upewdy.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_94upewdy.json" 5 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_94upewdy.json" 10 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_94upewdy.json" 5 0.001 &
run_task "94upewdy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2464.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_94upewdy.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 1 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 1 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 1 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 10 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_nteho7nl.json" 5 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_nteho7nl.json" 10 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_nteho7nl.json" 5 0.001 &
run_task "nteho7nl" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2465.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nteho7nl.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 1 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 1 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 1 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 10 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_8bu725t9.json" 5 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_8bu725t9.json" 10 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_8bu725t9.json" 5 0.001 &
run_task "8bu725t9" "/data/louisvl/TB/outputs/checkpoints/epoch_001-v313.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8bu725t9.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 1 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 1 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 1 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 10 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_z4y57mg2.json" 5 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_z4y57mg2.json" 10 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_z4y57mg2.json" 5 0.001 &
run_task "z4y57mg2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2467.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_z4y57mg2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 1 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 1 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 1 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 10 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ppxzmfkr.json" 5 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ppxzmfkr.json" 10 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ppxzmfkr.json" 5 0.001 &
run_task "ppxzmfkr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2468.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ppxzmfkr.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 1 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 1 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 1 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 10 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_5q32c2j2.json" 5 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_5q32c2j2.json" 10 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_5q32c2j2.json" 5 0.001 &
run_task "5q32c2j2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2466.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5q32c2j2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 1 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 1 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 1 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 10 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_7y8tx38g.json" 5 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_7y8tx38g.json" 10 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_7y8tx38g.json" 5 0.001 &
run_task "7y8tx38g" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2469.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7y8tx38g.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 1 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 1 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 1 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 10 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ervsw7lf.json" 5 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ervsw7lf.json" 10 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ervsw7lf.json" 5 0.001 &
run_task "ervsw7lf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2470.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ervsw7lf.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 1 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 1 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 1 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 10 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_5249ujlb.json" 5 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_5249ujlb.json" 10 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_5249ujlb.json" 5 0.001 &
run_task "5249ujlb" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2471.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5249ujlb.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 1 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 1 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 1 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 10 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_w5hgitvn.json" 5 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_w5hgitvn.json" 10 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_w5hgitvn.json" 5 0.001 &
run_task "w5hgitvn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2472.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_w5hgitvn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 1 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 1 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 1 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 10 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_1zpayfhs.json" 5 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_1zpayfhs.json" 10 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_1zpayfhs.json" 5 0.001 &
run_task "1zpayfhs" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2473.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_1zpayfhs.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 1 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 1 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 1 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 10 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_5xaeszku.json" 5 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_5xaeszku.json" 10 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_5xaeszku.json" 5 0.001 &
run_task "5xaeszku" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2474.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5xaeszku.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 1 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 1 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 1 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 10 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_hyofdsma.json" 5 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_hyofdsma.json" 10 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_hyofdsma.json" 5 0.001 &
run_task "hyofdsma" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2475.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_hyofdsma.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 1 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 1 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 1 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 10 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_7l4n65se.json" 5 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_7l4n65se.json" 10 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_7l4n65se.json" 5 0.001 &
run_task "7l4n65se" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2476.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7l4n65se.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 1 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 1 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 1 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 10 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_6lqsdq93.json" 5 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_6lqsdq93.json" 10 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_6lqsdq93.json" 5 0.001 &
run_task "6lqsdq93" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2477.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_6lqsdq93.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 1 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 1 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 1 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 10 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_0merrhqt.json" 5 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_0merrhqt.json" 10 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_0merrhqt.json" 5 0.001 &
run_task "0merrhqt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2478.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_0merrhqt.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 1 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 1 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 1 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 10 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_4umgyqo7.json" 5 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_4umgyqo7.json" 10 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_4umgyqo7.json" 5 0.001 &
run_task "4umgyqo7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2479.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4umgyqo7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 1 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 1 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 1 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 10 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_bsi3y7om.json" 5 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_bsi3y7om.json" 10 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_bsi3y7om.json" 5 0.001 &
run_task "bsi3y7om" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2480.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_bsi3y7om.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 1 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 1 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 1 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 10 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_qa3wbec0.json" 5 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_qa3wbec0.json" 10 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_qa3wbec0.json" 5 0.001 &
run_task "qa3wbec0" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2481.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_qa3wbec0.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 1 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 1 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 1 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 10 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_5aerf3de.json" 5 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_5aerf3de.json" 10 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_5aerf3de.json" 5 0.001 &
run_task "5aerf3de" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2482.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5aerf3de.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 1 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 1 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 1 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 10 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_8eenbv31.json" 5 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_8eenbv31.json" 10 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_8eenbv31.json" 5 0.001 &
run_task "8eenbv31" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2483.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8eenbv31.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 1 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 1 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 1 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 10 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_e0y7al95.json" 5 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_e0y7al95.json" 10 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_e0y7al95.json" 5 0.001 &
run_task "e0y7al95" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2484.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_e0y7al95.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 1 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 1 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 1 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 10 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_lzwxy8rr.json" 5 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_lzwxy8rr.json" 10 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_lzwxy8rr.json" 5 0.001 &
run_task "lzwxy8rr" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2485.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lzwxy8rr.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 1 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 1 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 1 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 10 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_91i5v8r2.json" 5 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_91i5v8r2.json" 10 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_91i5v8r2.json" 5 0.001 &
run_task "91i5v8r2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2486.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_91i5v8r2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 1 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 1 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 1 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 10 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_lchb4gky.json" 5 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_lchb4gky.json" 10 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_lchb4gky.json" 5 0.001 &
run_task "lchb4gky" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2487.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_lchb4gky.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 1 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 1 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 1 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 10 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_pmkuw2yi.json" 5 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_pmkuw2yi.json" 10 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_pmkuw2yi.json" 5 0.001 &
run_task "pmkuw2yi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2488.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pmkuw2yi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 1 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 1 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 1 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 10 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_pmmrgcbi.json" 5 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_pmmrgcbi.json" 10 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_pmmrgcbi.json" 5 0.001 &
run_task "pmmrgcbi" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2489.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pmmrgcbi.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 1 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 1 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 1 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 10 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_db2mefwq.json" 5 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_db2mefwq.json" 10 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_db2mefwq.json" 5 0.001 &
run_task "db2mefwq" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2490.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_db2mefwq.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 1 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 1 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 1 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 10 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_s1f0j20z.json" 5 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_s1f0j20z.json" 10 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_s1f0j20z.json" 5 0.001 &
run_task "s1f0j20z" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2491.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_s1f0j20z.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 1 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 1 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 1 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 10 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_pyfzlpi4.json" 5 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_pyfzlpi4.json" 10 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_pyfzlpi4.json" 5 0.001 &
run_task "pyfzlpi4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2492.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_pyfzlpi4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 1 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 1 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 1 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 10 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_wu2d1m64.json" 5 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_wu2d1m64.json" 10 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_wu2d1m64.json" 5 0.001 &
run_task "wu2d1m64" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2493.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_wu2d1m64.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 1 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 1 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 1 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 10 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_3ujj3q5u.json" 5 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_3ujj3q5u.json" 10 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_3ujj3q5u.json" 5 0.001 &
run_task "3ujj3q5u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2494.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_3ujj3q5u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 1 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 1 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 1 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 10 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_3kguaq4e.json" 5 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_3kguaq4e.json" 10 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_3kguaq4e.json" 5 0.001 &
run_task "3kguaq4e" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2495.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_3kguaq4e.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 1 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 1 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 1 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 10 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_vkaed2k3.json" 5 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_vkaed2k3.json" 10 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_vkaed2k3.json" 5 0.001 &
run_task "vkaed2k3" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2496.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vkaed2k3.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 1 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 1 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 1 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 10 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_7tjblzsw.json" 5 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_7tjblzsw.json" 10 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_7tjblzsw.json" 5 0.001 &
run_task "7tjblzsw" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2497.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7tjblzsw.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 1 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 1 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 1 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 10 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_mhyh80nn.json" 5 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_mhyh80nn.json" 10 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_mhyh80nn.json" 5 0.001 &
run_task "mhyh80nn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2498.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_mhyh80nn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 1 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 1 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 1 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 10 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_tr24lfkn.json" 5 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_tr24lfkn.json" 10 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_tr24lfkn.json" 5 0.001 &
run_task "tr24lfkn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2500.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tr24lfkn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 1 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 1 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 1 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 10 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_n0g267fp.json" 5 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_n0g267fp.json" 10 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_n0g267fp.json" 5 0.001 &
run_task "n0g267fp" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2499.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n0g267fp.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 1 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 1 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 1 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 10 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ndxbvbcd.json" 5 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ndxbvbcd.json" 10 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ndxbvbcd.json" 5 0.001 &
run_task "ndxbvbcd" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2501.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ndxbvbcd.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 1 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 1 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 1 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 10 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_4n5nv2mc.json" 5 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_4n5nv2mc.json" 10 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_4n5nv2mc.json" 5 0.001 &
run_task "4n5nv2mc" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2503.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4n5nv2mc.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 1 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8267410r.json" 1 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8267410r.json" 1 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_8267410r.json" 10 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_8267410r.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_8267410r.json" 5 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_8267410r.json" 10 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_8267410r.json" 5 0.001 &
run_task "8267410r" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2502.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8267410r.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 1 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 1 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 1 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 10 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_7i98qcga.json" 5 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_7i98qcga.json" 10 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_7i98qcga.json" 5 0.001 &
run_task "7i98qcga" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2504.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7i98qcga.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 1 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 1 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 1 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 10 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_tzlhp3oo.json" 5 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_tzlhp3oo.json" 10 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_tzlhp3oo.json" 5 0.001 &
run_task "tzlhp3oo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2506.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tzlhp3oo.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 1 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 1 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 1 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 10 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_jqw5122s.json" 5 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_jqw5122s.json" 10 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_jqw5122s.json" 5 0.001 &
run_task "jqw5122s" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2505.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_jqw5122s.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 1 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 1 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 1 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 10 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ivhca2s6.json" 5 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ivhca2s6.json" 10 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ivhca2s6.json" 5 0.001 &
run_task "ivhca2s6" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2507.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ivhca2s6.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 1 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 1 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 1 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 10 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_22e0biu2.json" 5 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_22e0biu2.json" 10 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_22e0biu2.json" 5 0.001 &
run_task "22e0biu2" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2508.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_22e0biu2.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 1 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 1 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 1 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 10 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_uif0tapy.json" 5 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_uif0tapy.json" 10 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_uif0tapy.json" 5 0.001 &
run_task "uif0tapy" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2509.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_uif0tapy.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 1 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 1 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 1 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 10 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_hx28jtd4.json" 5 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_hx28jtd4.json" 10 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_hx28jtd4.json" 5 0.001 &
run_task "hx28jtd4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2510.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_hx28jtd4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 1 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 1 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 1 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 10 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_nlom9ub1.json" 5 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_nlom9ub1.json" 10 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_nlom9ub1.json" 5 0.001 &
run_task "nlom9ub1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2511.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nlom9ub1.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 1 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 1 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 1 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 10 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ah4o1m97.json" 5 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ah4o1m97.json" 10 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ah4o1m97.json" 5 0.001 &
run_task "ah4o1m97" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2512.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ah4o1m97.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 1 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 1 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 1 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 10 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_fcffexvt.json" 5 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_fcffexvt.json" 10 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_fcffexvt.json" 5 0.001 &
run_task "fcffexvt" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2513.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_fcffexvt.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 1 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 1 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 1 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 10 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ly7hntbu.json" 5 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ly7hntbu.json" 10 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ly7hntbu.json" 5 0.001 &
run_task "ly7hntbu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2514.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ly7hntbu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 1 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 1 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 1 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 10 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_u2rrw26d.json" 5 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_u2rrw26d.json" 10 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_u2rrw26d.json" 5 0.001 &
run_task "u2rrw26d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2515.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_u2rrw26d.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 1 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 1 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 1 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 10 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_0ubgq0a1.json" 5 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_0ubgq0a1.json" 10 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_0ubgq0a1.json" 5 0.001 &
run_task "0ubgq0a1" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2516.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_0ubgq0a1.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 1 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 1 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 1 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 10 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_x1b4d8aa.json" 5 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_x1b4d8aa.json" 10 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_x1b4d8aa.json" 5 0.001 &
run_task "x1b4d8aa" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2518.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_x1b4d8aa.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 1 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 1 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 1 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 10 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_9egf3m13.json" 5 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_9egf3m13.json" 10 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_9egf3m13.json" 5 0.001 &
run_task "9egf3m13" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2517.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_9egf3m13.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 1 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 1 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 1 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 10 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_m8ocr8zn.json" 5 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_m8ocr8zn.json" 10 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_m8ocr8zn.json" 5 0.001 &
run_task "m8ocr8zn" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2519.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_m8ocr8zn.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 1 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 1 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 1 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 10 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_fitgb2m7.json" 5 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_fitgb2m7.json" 10 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_fitgb2m7.json" 5 0.001 &
run_task "fitgb2m7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2520.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_fitgb2m7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 1 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 1 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 1 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 10 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_tk5wo1ws.json" 5 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_tk5wo1ws.json" 10 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_tk5wo1ws.json" 5 0.001 &
run_task "tk5wo1ws" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2521.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_tk5wo1ws.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 1 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 1 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 1 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 10 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_xewvrg3u.json" 5 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_xewvrg3u.json" 10 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_xewvrg3u.json" 5 0.001 &
run_task "xewvrg3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2522.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_xewvrg3u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 1 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 1 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 1 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 10 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_eu8olsye.json" 5 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_eu8olsye.json" 10 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_eu8olsye.json" 5 0.001 &
run_task "eu8olsye" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2523.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_eu8olsye.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 1 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 1 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 1 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 10 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_nlkgjs21.json" 5 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_nlkgjs21.json" 10 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_nlkgjs21.json" 5 0.001 &
run_task "nlkgjs21" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2524.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_nlkgjs21.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 1 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 1 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 1 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 10 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_vmxwjsl4.json" 5 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_vmxwjsl4.json" 10 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_vmxwjsl4.json" 5 0.001 &
run_task "vmxwjsl4" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2525.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_vmxwjsl4.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 1 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_9eft616w.json" 1 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_9eft616w.json" 1 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_9eft616w.json" 10 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_9eft616w.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_9eft616w.json" 5 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_9eft616w.json" 10 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_9eft616w.json" 5 0.001 &
run_task "9eft616w" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2526.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_9eft616w.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 1 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 1 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 1 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 10 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_av14zv5i.json" 5 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_av14zv5i.json" 10 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_av14zv5i.json" 5 0.001 &
run_task "av14zv5i" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2527.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_av14zv5i.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 1 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 1 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 1 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 10 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_7rc4q62d.json" 5 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_7rc4q62d.json" 10 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_7rc4q62d.json" 5 0.001 &
run_task "7rc4q62d" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2528.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_7rc4q62d.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 1 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 1 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 1 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 10 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_4xy6r2iv.json" 5 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_4xy6r2iv.json" 10 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_4xy6r2iv.json" 5 0.001 &
run_task "4xy6r2iv" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2529.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_4xy6r2iv.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 1 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 1 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 1 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 10 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_2ct3gfg7.json" 5 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_2ct3gfg7.json" 10 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_2ct3gfg7.json" 5 0.001 &
run_task "2ct3gfg7" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2530.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_2ct3gfg7.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 1 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 1 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 1 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 10 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ny1ddcft.json" 5 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ny1ddcft.json" 10 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ny1ddcft.json" 5 0.001 &
run_task "ny1ddcft" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2531.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ny1ddcft.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 1 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 1 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 1 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 10 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_n3aqrt3u.json" 5 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_n3aqrt3u.json" 10 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_n3aqrt3u.json" 5 0.001 &
run_task "n3aqrt3u" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2532.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_n3aqrt3u.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 1 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 1 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 1 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 10 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_d0ya0liu.json" 5 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_d0ya0liu.json" 10 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_d0ya0liu.json" 5 0.001 &
run_task "d0ya0liu" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2533.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_d0ya0liu.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 1 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 1 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 1 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 10 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_6f0nhgq5.json" 5 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_6f0nhgq5.json" 10 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_6f0nhgq5.json" 5 0.001 &
run_task "6f0nhgq5" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2534.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_6f0nhgq5.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 1 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 1 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 1 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 10 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_5pca11wg.json" 5 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_5pca11wg.json" 10 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_5pca11wg.json" 5 0.001 &
run_task "5pca11wg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2535.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_5pca11wg.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 1 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 1 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 1 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 10 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_t62vikoo.json" 5 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_t62vikoo.json" 10 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_t62vikoo.json" 5 0.001 &
run_task "t62vikoo" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2536.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_t62vikoo.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 1 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 1 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 1 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 10 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_o5g0i994.json" 5 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_o5g0i994.json" 10 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_o5g0i994.json" 5 0.001 &
run_task "o5g0i994" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2537.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_o5g0i994.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 1 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 1 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 1 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 10 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_1235n2ya.json" 5 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_1235n2ya.json" 10 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_1235n2ya.json" 5 0.001 &
run_task "1235n2ya" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2538.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_1235n2ya.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 1 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 1 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 1 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 10 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ojfw9t3t.json" 5 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ojfw9t3t.json" 10 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ojfw9t3t.json" 5 0.001 &
run_task "ojfw9t3t" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2539.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ojfw9t3t.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 1 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 1 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 1 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 10 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_ddnwf4xf.json" 5 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_ddnwf4xf.json" 10 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_ddnwf4xf.json" 5 0.001 &
run_task "ddnwf4xf" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2540.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_ddnwf4xf.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 1 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 1 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 1 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 10 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_yxbqz147.json" 5 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_yxbqz147.json" 10 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_yxbqz147.json" 5 0.001 &
run_task "yxbqz147" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2541.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_yxbqz147.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 1 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 1 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 1 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 10 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_8ha8ha3x.json" 5 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_8ha8ha3x.json" 10 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_8ha8ha3x.json" 5 0.001 &
run_task "8ha8ha3x" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2542.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_8ha8ha3x.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 1 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 5 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 1 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 10 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 1 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 1 0.001 &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 1 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 1 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 10 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 5 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 5 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 10 0.001 &

# Wait for batch to complete
wait

run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 10 "cuda:0" 42 "configs_cache/config_yulza1jg.json" 5 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 10 "cuda:1" 43 "configs_cache/config_yulza1jg.json" 10 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 50 "cuda:2" 44 "configs_cache/config_yulza1jg.json" 5 0.001 &
run_task "yulza1jg" "/data/louisvl/TB/outputs/checkpoints/epoch_000-v2543.ckpt" "gpf-plus" 50 "cuda:3" 45 "configs_cache/config_yulza1jg.json" 10 0.001 &

# Wait for batch to complete
wait


# Wait for all remaining tasks
wait

echo "All 2784 downstream evaluations complete!"