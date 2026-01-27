"""
Generate a bash script for parallel downstream evaluation.

This script:
1. Fetches runs from wandb
2. Generates all evaluation tasks
3. Creates a bash script that runs them in parallel across 4 GPUs

Usage:
    python tutorials/generate_downstream_eval_script.py \
        --wandb_project "entity/project-name" \
        --output scripts/run_downstream_eval_tasks.sh
"""

import argparse
import sys
from pathlib import Path
import json

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(_THIS_DIR))

try:
    import wandb
except ImportError:
    print("ERROR: wandb not installed.")
    sys.exit(1)

# Import from the grid script
from run_downstream_eval_grid import (
    fetch_runs_from_wandb,
    extract_dataset_config,
    extract_base_model_config,
    config_to_hash,
    EVAL_MODES,
    PROMPT_MODES,
    N_TRAIN_VALUES,
    DOWNSTREAM_WANDB_PROJECT,
    LR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, default="louis-van-langendonck-universitat-polit-cnica-de-catalunya/linkpred_pretraining")
    parser.add_argument("--output", type=str, default="scripts/run_downstream_eval_tasks.sh")
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--downstream_project", type=str, default=DOWNSTREAM_WANDB_PROJECT)
    parser.add_argument("--enable_prompt_tuning", type=bool, default=True, help="Enable prompt tuning methods (GPF, GPF-Plus)")
    
    args = parser.parse_args()
    
    print("Fetching runs from wandb...")
    run_metadata_list = fetch_runs_from_wandb(args.wandb_project, max_runs=args.max_runs)
    
    if len(run_metadata_list) == 0:
        print("No valid runs found!")
        return
    
    # Build task list with deduplication
    print("\nBuilding task list...")
    tasks = []
    scratch_done = set()
    skipped = 0
    
    # Create configs directory
    configs_dir = Path("configs_cache")
    configs_dir.mkdir(exist_ok=True)
    
    # Standard modes (finetune, scratch)
    for run_metadata in run_metadata_list:
        for mode in EVAL_MODES:
            for n_train in N_TRAIN_VALUES:
                # Deduplicate scratch
                if mode == "scratch":
                    scratch_key = (
                        run_metadata["dataset_hash"],
                        run_metadata["base_model_hash"],
                        n_train
                    )
                    if scratch_key in scratch_done:
                        skipped += 1
                        continue
                    scratch_done.add(scratch_key)
                
                # Save config to JSON
                config_file = configs_dir / f"config_{run_metadata['run_id']}.json"
                with open(config_file, "w") as f:
                    json.dump(run_metadata["config"], f)
                
                tasks.append({
                    "run_id": run_metadata["run_id"],
                    "checkpoint": run_metadata["checkpoint_path"],
                    "mode": mode,
                    "n_train": n_train,
                    "config_file": str(config_file),
                    "pretraining_method": run_metadata["pretraining_method"],
                    "model_name": run_metadata["model_name"],
                    "p_num": 5,  # Default, not used
                    "lr_override": None,
                })
    
    # Prompt tuning modes (if enabled)
    if args.enable_prompt_tuning:
        for run_metadata in run_metadata_list:
            for prompt_mode, param_grid in PROMPT_MODES.items():
                for n_train in N_TRAIN_VALUES:
                    # Save config to JSON
                    config_file = configs_dir / f"config_{run_metadata['run_id']}.json"
                    if not config_file.exists():
                        with open(config_file, "w") as f:
                            json.dump(run_metadata["config"], f)
                    
                    # Grid search over hyperparameters
                    for p_num in param_grid.get("p_num", [5]):
                        for lr in param_grid.get("lr", [LR]):
                            tasks.append({
                                "run_id": run_metadata["run_id"],
                                "checkpoint": run_metadata["checkpoint_path"],
                                "mode": prompt_mode,
                                "n_train": n_train,
                                "config_file": str(config_file),
                                "pretraining_method": run_metadata["pretraining_method"],
                                "model_name": run_metadata["model_name"],
                                "p_num": p_num,
                                "lr_override": lr,
                            })
    
    print(f"Total tasks: {len(tasks)} (skipped {skipped} duplicate scratch)")
    
    # Generate bash script
    print(f"\nGenerating bash script: {args.output}")
    
    script_lines = [
        "#!/bin/bash",
        "",
        "# Auto-generated parallel downstream evaluation script",
        f"# Source project: {args.wandb_project}",
        f"# Downstream project: {args.downstream_project}",
        f"# Total tasks: {len(tasks)}",
        "",
        "# Create logs directory",
        "mkdir -p logs",
        "",
        "# Function to run a task",
        "run_task() {",
        "    local run_id=$1",
        "    local checkpoint=$2",
        "    local mode=$3",
        "    local n_train=$4",
        "    local device=$5",
        "    local seed=$6",
        "    local config_file=$7",
        "    local p_num=$8",
        "    local lr_override=$9",
        "",
        "    # Build command",
        "    local cmd=\"python3 tutorials/run_downstream_eval_single.py\"",
        f"    cmd=\"$cmd --wandb_project {args.wandb_project}\"",
        "    cmd=\"$cmd --run_id $run_id\"",
        "    cmd=\"$cmd --checkpoint $checkpoint\"",
        "    cmd=\"$cmd --mode $mode\"",
        "    cmd=\"$cmd --n_train $n_train\"",
        "    cmd=\"$cmd --device $device\"",
        "    cmd=\"$cmd --seed $seed\"",
        f"    cmd=\"$cmd --downstream_project {args.downstream_project}\"",
        "    cmd=\"$cmd --config_json $config_file\"",
        "    cmd=\"$cmd --p_num $p_num\"",
        "    ",
        "    # Add lr_override if not 'none'",
        "    if [ \"$lr_override\" != \"none\" ]; then",
        "        cmd=\"$cmd --lr_override $lr_override\"",
        "    fi",
        "    ",
        "    # Run command",
        "    eval $cmd",
        "}",
        "",
        "# Export function for parallel",
        "export -f run_task",
        "",
        "# Task list (device assignment round-robin)",
    ]
    
    # Add all tasks
    devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
    for idx, task in enumerate(tasks):
        device = devices[idx % len(devices)]
        seed = 42 + (idx % len(devices))
        
        # Handle lr_override (use "none" if None)
        lr_str = str(task['lr_override']) if task['lr_override'] is not None else "none"
        
        task_line = (
            f"run_task \"{task['run_id']}\" \"{task['checkpoint']}\" "
            f"\"{task['mode']}\" {task['n_train']} \"{device}\" {seed} "
            f"\"{task['config_file']}\" {task['p_num']} {lr_str} &"
        )
        script_lines.append(task_line)
        
        # Add wait every 4 tasks to limit parallelism
        if (idx + 1) % 4 == 0:
            script_lines.append("")
            script_lines.append("# Wait for batch to complete")
            script_lines.append("wait")
            script_lines.append("")
    
    script_lines.extend([
        "",
        "# Wait for all remaining tasks",
        "wait",
        "",
        f"echo \"All {len(tasks)} downstream evaluations complete!\"",
    ])
    
    # Write script
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("\n".join(script_lines))
    
    # Make executable
    import stat
    output_path.chmod(output_path.stat().st_mode | stat.S_IEXEC)
    
    print(f"\n✓ Script generated: {output_path}")
    print(f"\nTo run:")
    print(f"  bash {output_path}")


if __name__ == "__main__":
    main()

