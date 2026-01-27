"""
Run downstream evaluation for a SINGLE (run, mode, n_train) combination.

This script is designed to be called from a bash parallelization script.

Supports all adaptation methods:
- finetune, scratch, linear, mlp
- gpf, gpf-plus (prompt tuning)
- untrained_frozen (baseline)

Usage:
    # Standard methods
    python tutorials/run_downstream_eval_single.py \
        --wandb_project "entity/project" \
        --run_id <wandb_run_id> \
        --checkpoint <checkpoint_path> \
        --mode finetune \
        --n_train 50 \
        --device cuda:0 \
        --downstream_project "downstream_eval" \
        --config_json path/to/config.json
    
    # Prompt tuning methods
    python tutorials/run_downstream_eval_single.py \
        --mode gpf-plus \
        --p_num 5 \
        --n_train 50 \
        --device cuda:0 \
        ...
"""

import argparse
import sys
from pathlib import Path
import json
import yaml

# Allow running from local repo
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(_THIS_DIR))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("ERROR: wandb not installed.")
    sys.exit(1)

from downstream_eval import run_downstream_evaluation
from run_downstream_eval_grid_v2 import (
    detect_pretraining_method,
    config_to_hash,
    extract_dataset_config,
    extract_base_model_config,
    extract_pretraining_specific_config,
)

# Constants from the grid script
N_EVALUATION_VAL = 100
N_EVALUATION_TEST = 100
N_EVALUATION_TOTAL = N_EVALUATION_VAL + N_EVALUATION_TEST

EPOCHS = 300
LR = 0.001
BATCH_SIZE = 16
PATIENCE = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_project", type=str, required=True)
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True, 
                        choices=["finetune", "scratch", "linear", "mlp", "gpf", "gpf-plus", "untrained_frozen"],
                        help="Adaptation method to use")
    parser.add_argument("--n_train", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--downstream_project", type=str, required=True)
    parser.add_argument("--config_json", type=str, required=True, help="Path to run config JSON")
    parser.add_argument("--p_num", type=int, default=5, help="Number of basis vectors for GPF-Plus")
    parser.add_argument("--lr_override", type=float, default=None, help="Override learning rate")
    
    args = parser.parse_args()
    
    # Load config from JSON file
    with open(args.config_json, "r") as f:
        run_config = json.load(f)
    
    # Extract metadata from config
    pretraining_method = detect_pretraining_method(run_config)
    model_name = run_config.get("model", {}).get("model_name", "unknown")
    
    # Extract config hashes
    dataset_cfg = extract_dataset_config(run_config)
    base_model_cfg = extract_base_model_config(run_config)
    pretraining_cfg = extract_pretraining_specific_config(run_config)
    
    dataset_hash = config_to_hash(dataset_cfg)
    base_model_hash = config_to_hash(base_model_cfg)
    pretraining_hash = config_to_hash(pretraining_cfg)
    
    # Create descriptive wandb run name
    # Add p_num to name if using prompt methods
    if args.mode in ["gpf-plus"]:
        run_name = f"{pretraining_method}_{model_name}_{args.mode}_p{args.p_num}_n{args.n_train}"
    else:
        run_name = f"{pretraining_method}_{model_name}_{args.mode}_n{args.n_train}"
    
    print(f"\n[{args.device}] Starting downstream eval:")
    print(f"  Run: {run_name}")
    print(f"  Run ID: {args.run_id}")
    print(f"  Mode: {args.mode}, N_train: {args.n_train}")
    
    # Determine actual learning rate
    actual_lr = args.lr_override if args.lr_override is not None else LR
    
    # Initialize wandb with rich metadata BEFORE calling downstream_eval
    wandb_config = {
        # Downstream eval settings
        "downstream_mode": args.mode,
        "downstream_n_train": args.n_train,
        "downstream_n_val": N_EVALUATION_VAL,
        "downstream_n_test": N_EVALUATION_TEST,
        "downstream_n_evaluation_total": N_EVALUATION_TOTAL,
        "downstream_epochs": EPOCHS,
        "downstream_lr": actual_lr,
        "downstream_batch_size": BATCH_SIZE,
        "downstream_patience": PATIENCE,
        "downstream_seed": args.seed,
        
        # Prompt-specific settings (if applicable)
        "downstream_p_num": args.p_num if args.mode in ["gpf-plus"] else None,
        
        # Pre-training metadata
        "pretraining_run_id": args.run_id,
        "pretraining_method": pretraining_method,
        "pretraining_model_name": model_name,
        "pretraining_checkpoint": args.checkpoint,
        
        # Config hashes for grouping
        "dataset_hash": dataset_hash,
        "base_model_hash": base_model_hash,
        "pretraining_hash": pretraining_hash,
        
        # IMPORTANT: Include full original pre-training config
        "pretraining_config": run_config,
    }
    
    wandb.init(
        project=args.downstream_project,
        name=run_name,
        config=wandb_config,
        tags=[
            pretraining_method,
            model_name,
            args.mode,
            f"n_train_{args.n_train}",
            "graphuniverse",
        ],
        reinit=True,
    )
    
    try:
        # Create temporary directory for config
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files_dir = tmp_path / "files"
            files_dir.mkdir()
            
            # Save config
            config_path = files_dir / "config.yaml"
            with open(config_path, "w") as f:
                yaml.dump(run_config, f)
            
            # Save summary
            summary_path = files_dir / "wandb-summary.json"
            with open(summary_path, "w") as f:
                json.dump({"best_epoch/checkpoint": args.checkpoint}, f)
            
            # Run evaluation (with use_wandb=False since we already initialized it)
            results = run_downstream_evaluation(
                run_dir=str(tmp_path),
                checkpoint_path=args.checkpoint,
                n_evaluation_graphs=N_EVALUATION_TOTAL,
                n_train=args.n_train,
                mode=args.mode,
                epochs=EPOCHS,
                lr=actual_lr,
                batch_size=BATCH_SIZE,
                patience=PATIENCE,
                device=args.device,
                seed=args.seed,
                use_wandb=False,  # We handle wandb ourselves
                wandb_project=args.downstream_project,
                p_num=args.p_num,
            )
        
        # Log summary metrics to our wandb run
        if wandb.run is not None:
            wandb.log({
                "final/test_accuracy": results["test_accuracy"],
                "final/num_nodes": results.get("num_nodes", 0),
                "final/num_classes": results["num_classes"],
                "final/best_val_acc": max(results["history"]["val_acc"]),
                "final/best_train_acc": max(results["history"]["train_acc"]),
            })
    
    finally:
        # Finish wandb run
        if wandb.run is not None:
            wandb.finish()
    
    acc = results.get("test_accuracy", "N/A")
    print(f"[{args.device}] ✓ Completed - Acc: {acc}")
    
    return results


if __name__ == "__main__":
    main()

