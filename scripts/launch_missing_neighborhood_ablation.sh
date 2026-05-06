#!/bin/bash
# ==============================================================================
# SCRIPT: launch_missing_neighborhood_ablation.sh
# DESCRIPTION:
#   Launches the missing runs for the neighborhood ablation study.
#   - CONCURRENCY: Uses "Virtual Slots" to run N jobs per GPU.
#   - LOGGING: Uses run_and_log for robust execution and failure tracking.
#   - HYPERPARAMS: Explicitly loops over search space (no --multirun).
# ==============================================================================

export SELECTED_GPUS="0,1,2,3,4,5,6,7" 
wandb_entity="gbg141-hopse"
RESUME=true  # Set to true to skip already-completed runs

# ==============================================================================
# SECTION 1: LOGGING & ENVIRONMENT SETUP
# ==========================================================

# Kill all background child processes if this script is interrupted
trap 'echo -e "\n🛑 Interrupted! Cleaning up all background jobs..."; kill 0 2>/dev/null; exit 1' SIGINT SIGTERM

script_name="$(basename "${BASH_SOURCE[0]}" .sh)"
log_group="missing_neighborhood_ablation"
LOG_DIR="./logs/${log_group}"

mkdir -p "$LOG_DIR"

# Load logging utils
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
find_logging_script() {
    local dir="$1"
    while [[ "$dir" != "/" ]]; do
        if [[ -f "$dir/base/logging.sh" ]]; then echo "$dir/base/logging.sh"; return 0; fi
        if [[ -f "$dir/scripts/base/logging.sh" ]]; then echo "$dir/scripts/base/logging.sh"; return 0; fi
        dir="$(dirname "$dir")"
    done
    return 1
}
LOGGING_PATH=$(find_logging_script "$SCRIPT_DIR")
if [[ -n "$LOGGING_PATH" ]]; then
    source "$LOGGING_PATH"
else
    echo "❌ CRITICAL ERROR: Could not locate 'base/logging.sh'."
    exit 1
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ==============================================================================
# SECTION 2: HARDWARE & CONCURRENCY
# ==============================================================================

_gpu_info=$(python3 -c "
import subprocess, os
selected_env = os.environ.get('SELECTED_GPUS', '').strip()
allowed_gpus = [x.strip() for x in selected_env.split(',')] if selected_env else None
try:
    out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total', '--format=csv,noheader,nounits'], text=True)
    indices, mem_mb = [], []
    for line in out.strip().splitlines():
        idx, mem = line.split(',')
        idx = idx.strip()
        if allowed_gpus and idx not in allowed_gpus: continue
        indices.append(idx)
        mem_mb.append(int(mem.strip()))
    if not indices: print('0'); exit(0)
    min_mem_gb = min(mem_mb) / 1024
    jobs = 4 if min_mem_gb >= 80 else (2 if min_mem_gb <= 30 else 3)
    print(jobs, ' '.join(indices))
except Exception: print('2 0')
")
read -r JOBS_PER_GPU _gpu_ids <<< "$_gpu_info"
read -ra physical_gpus <<< "$_gpu_ids"
gpus=()
for gpu in "${physical_gpus[@]}"; do
    for ((i=1; i<=JOBS_PER_GPU; i++)); do gpus+=("$gpu"); done
done
declare -a slot_pids
for i in "${!gpus[@]}"; do slot_pids[$i]=0; done

# ==============================================================================
# SECTION 3: EXPERIMENT DEFINITIONS (The missing combinations)
# ==============================================================================

# Format: "Model | Dataset | Neighborhoods | Encodings (optional) | ExtraArgs"
# NOTE: Use ';' to separate multiple neighborhood/encoding options if needed.
MISSING_COMBOS=(
    # # --- HOPSE-M-F (cell) ---
    # "cell/hopse_m | graph/CYP3A4_Veith | [up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | [HKFE,KHopFE,PPRFE] | "
    
    # # --- HOPSE-M-PE (cell) ---
    # "cell/hopse_m | graph/BBB_Martins | [up_adjacency-0] | [LapPE,RWSE,ElectrostaticPE,HKdiagSE] | "
    # "cell/hopse_m | graph/CYP3A4_Veith | [up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | [LapPE,RWSE,ElectrostaticPE,HKdiagSE] | "
    
    # # --- HOPSE-G (cell) ---
    # "cell/hopse_g | graph/CYP3A4_Veith | [up_adjacency-0] | null | transforms.hopse_encoding.pretrain_model=molpcba,zinc"
    
    # --- HOPSE-G (simplicial) ---
    # "simplicial/hopse_g | simplicial/mantra_name | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,2-up_incidence-0];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | transforms.hopse_encoding.pretrain_model=molpcba,zinc"
    # "simplicial/hopse_g | simplicial/mantra_orientation | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,2-up_incidence-0];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | transforms.hopse_encoding.pretrain_model=molpcba,zinc"
    "simplicial/hopse_g | simplicial/mantra_betti_numbers | [up_adjacency-0];[up_adjacency-0,2-up_adjacency-0];[up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,2-up_incidence-0];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | transforms.hopse_encoding.pretrain_model=molpcba,zinc evaluator=betti_numbers"

    # --- TOPOTUNE (simplicial) ---
    # "simplicial/topotune | graph/MUTAG | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2] | null | "
    # "simplicial/topotune | graph/NCI1 | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2] | null | "
    # "simplicial/topotune | graph/NCI109 | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2] | null | "
    # "simplicial/topotune | graph/BBB_Martins | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | "
    # "simplicial/topotune | graph/CYP3A4_Veith | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2] | null | "
    # "simplicial/topotune | graph/Clearance_Hepatocyte_AZ | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2] | null | "
    # "simplicial/topotune | graph/Caco2_Wang | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,2-up_incidence-0];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | "
    # "simplicial/topotune | simplicial/mantra_betti_numbers | [up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | evaluator=betti_numbers"

    # # --- TOPOTUNE (cell) ---
    # "cell/topotune | graph/Caco2_Wang | [up_adjacency-0,up_adjacency-1,2-up_adjacency-0,down_adjacency-1,down_adjacency-2,2-down_adjacency-2];[up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | "
    # "cell/topotune | graph/BBB_Martins | [up_incidence-0,up_incidence-1,2-up_incidence-0,down_incidence-1,down_incidence-2,2-down_incidence-2] | null | "
)

# Shared Search Space (Hyperparameters)
L_vals=(2 4)
h_vals=(256)
pdro_vals=(0.5)
lr_vals=(0.01 0.001)
wd_vals=(0.0001)
bs_vals=(256)
SEEDS=(0 3 5 7 9)

FIXED_ARGS=(
    "trainer.max_epochs=500"
    "trainer.min_epochs=50"
    "trainer.check_val_every_n_epoch=5"
    "callbacks.early_stopping.patience=20"
    "delete_checkpoint_after_test=True"
)

# ==============================================================================
# SECTION 4: COMMAND GENERATOR
# ==============================================================================

generate_all_commands() {
    for combo in "${MISSING_COMBOS[@]}"; do
        IFS='|' read -r model_path dataset_path nbhd_list enc_list extra_args <<< "$combo"
        
        # Clean inputs
        model_path=$(echo "$model_path" | xargs)
        dataset_path=$(echo "$dataset_path" | xargs)
        nbhd_list=$(echo "$nbhd_list" | xargs)
        enc_list=$(echo "$enc_list" | xargs)
        extra_args=$(echo "$extra_args" | xargs)

        model_name=$(basename "$model_path")
        dataset_name=$(basename "$dataset_path")

        # Split multiple configurations if provided in the list (using ';' to avoid breaking lists with commas)
        IFS=';' read -ra NBHDS <<< "$nbhd_list"
        IFS=';' read -ra ENCS <<< "$enc_list"

        for nbhd in "${NBHDS[@]}"; do
            for enc in "${ENCS[@]}"; do
                # Determine project name
                proj="missing_ablation_${dataset_name}"
                
                # Nested loops for hyperparameters
                for L in "${L_vals[@]}"; do
                for h in "${h_vals[@]}"; do
                for pdro in "${pdro_vals[@]}"; do
                for lr in "${lr_vals[@]}"; do
                for wd in "${wd_vals[@]}"; do
                for bs in "${bs_vals[@]}"; do
                for seed in "${SEEDS[@]}"; do
                    
                    # 1. Build Run Name
                    # (Shorten nbhd for name)
                    nbhd_tag=$(echo "$nbhd" | grep -oE "adj[0-9]|inc[0-9]" | head -1)
                    if [ -z "$nbhd_tag" ]; then nbhd_tag="Ncustom"; fi
                    
                    run_name="${model_name}_${dataset_name}_${nbhd_tag}"
                    if [[ "$enc" != "null" ]]; then
                        enc_tag=$(echo "$enc" | grep -oE "pse|fe" | head -1)
                        if [ -z "$enc_tag" ]; then enc_tag="Ecustom"; fi
                        run_name="${run_name}_${enc_tag}"
                    fi
                    run_name="${run_name}_L${L}_h${h}_pdro${pdro}_lr${lr}_wd${wd}_bs${bs}_seed${seed}"

                    # 2. Build Hydra Arguments (Quote values for Hydra)
                    args=(
                        "model=$model_path"
                        "dataset=$dataset_path"
                        "dataset.split_params.data_seed=$seed"
                        "optimizer.parameters.lr=$lr"
                        "optimizer.parameters.weight_decay=$wd"
                        "dataset.dataloader_params.batch_size=$bs"
                        "model.feature_encoder.out_channels=$h"
                        "model.feature_encoder.proj_dropout=$pdro"
                    )
                    
                    # Backbone layer key varies
                    if [[ "$model_path" == *"topotune"* ]]; then
                        args+=("model.backbone.GNN.num_layers=$L")
                        args+=("model.backbone.neighborhoods='$nbhd'")
                    else
                        args+=("model.backbone.n_layers=$L")
                        args+=("model.preprocessing_params.neighborhoods='$nbhd'")
                    fi

                    if [[ "$enc" != "null" ]]; then
                        args+=("model.preprocessing_params.encodings='$enc'")
                    fi

                    # Cell model on graph lifting
                    if [[ "$model_path" == *"cell/"* && "$dataset_path" == *"graph/"* ]]; then
                        args+=("transforms.graph2cell_lifting.neighborhoods='$nbhd'")
                    fi
                    
                    # Append extra args
                    if [ -n "$extra_args" ]; then
                        read -ra EXTRA_ARRAY <<< "$extra_args"
                        for ex in "${EXTRA_ARRAY[@]}"; do
                            # If extra arg contains a comma (like pretrain_model=molpcba,zinc), loop it
                            if [[ "$ex" == *"="*","* ]]; then
                                key="${ex%%=*}"
                                vals_str="${ex#*=}"
                                IFS=',' read -ra VALS <<< "$vals_str"
                                for v in "${VALS[@]}"; do
                                    final_args=("${args[@]}" "$key=$v" "${FIXED_ARGS[@]}")
                                    final_name="${run_name}_${v}"
                                    echo "${final_name};python -m topobench ${final_args[*]} logger.wandb.project=$proj +logger.wandb.entity=$wandb_entity +logger.wandb.name=$final_name"
                                done
                                continue 2 
                            else
                                args+=("$ex")
                            fi
                        done
                    fi

                    final_args=("${args[@]}" "${FIXED_ARGS[@]}")
                    echo "${run_name};python -m topobench ${final_args[*]} logger.wandb.project=$proj +logger.wandb.entity=$wandb_entity +logger.wandb.name=$run_name"

                done; done; done; done; done; done; done
            done
        done
    done
}

# ==============================================================================
# SECTION 5: RESUME — LOAD COMPLETED RUNS
# ==============================================================================
declare -A _completed_runs
_success_log="$LOG_DIR/SUCCESSFUL_RUNS.log"
if [[ "$RESUME" == "true" && -f "$_success_log" ]]; then
    while IFS= read -r _line; do
        _rname="${_line##*\[SUCCESS\] }"
        _completed_runs["$_rname"]=1
    done < "$_success_log"
    echo "✔ Loaded ${#_completed_runs[@]} completed runs to skip."
fi

# ==============================================================================
# SECTION 6: MAIN EXECUTION LOOP
# ==============================================================================
echo "----------------------------------------------------------"
echo " Starting execution loop..."
echo "----------------------------------------------------------"

while IFS=";" read -r run_name cmd; do

    if [[ "$RESUME" == "true" && -n "${_completed_runs[$run_name]+x}" ]]; then
        continue
    fi

    # Find a Free GPU Slot
    assigned_slot=-1
    while [ "$assigned_slot" -eq -1 ]; do
        for i in "${!gpus[@]}"; do
            pid="${slot_pids[$i]}"
            if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
                assigned_slot=$i; break
            fi
        done
        if [ "$assigned_slot" -eq -1 ]; then wait -n; fi
    done

    # Prepare Command with assigned GPU
    current_gpu=${gpus[$assigned_slot]}
    full_cmd="$cmd trainer.devices=[$current_gpu]"

    # Execute via run_and_log
    run_and_log "$full_cmd" "$log_group" "$run_name" "$LOG_DIR" &
    slot_pids[$assigned_slot]=$!

done < <(generate_all_commands)

wait
echo "✔ All runs complete."
