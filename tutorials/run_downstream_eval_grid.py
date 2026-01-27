"""
Automated downstream evaluation for pre-trained models using wandb API.

This script:
1. Fetches runs from a wandb project via the API
2. Automatically detects unique dataset and base model configurations
3. Runs scratch baselines only once per unique (dataset_config, base_model_config, n_train)
4. Runs finetuning for all pretraining variations per base configuration
5. Logs all results to wandb with original pre-training config

Key Features:
- Works with wandb projects directly (no local file access needed)
- Automatically extracts unique dataset configs
- Automatically extracts base model configs (excluding pretraining-specific params)
- Smart deduplication for scratch models
- Supports any pretraining method (DGI, GraphCL, GraphMAE, etc.)
- **Parallel execution across multiple GPUs for faster evaluation**

Usage:
    # Sequential (single GPU)
    python tutorials/run_downstream_eval_grid_v2.py \
        --wandb_project "entity/project-name" \
        --device cuda
    
    # Parallel (4 GPUs)
    python tutorials/run_downstream_eval_grid_v2.py \
        --wandb_project "entity/project-name" \
        --device cuda \
        --num_workers 4 \
        --devices 0 1 2 3
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import hashlib
import yaml
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
import queue

# Allow running from local repo
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

# Add tutorials to path
_TUTORIALS_DIR = _REPO_ROOT / "tutorials"
sys.path.insert(0, str(_TUTORIALS_DIR))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("ERROR: wandb not installed. This script requires wandb.")
    sys.exit(1)

from downstream_eval import run_downstream_evaluation


# =============================================================================
# Configuration
# =============================================================================

# Modes to evaluate
EVAL_MODES = ['gpf', 'gpf-plus', 'mlp', 'scratch', 'finetune'] #["finetune", "scratch"]

# Prompt tuning modes (optional, with hyperparameter grids)
PROMPT_MODES = {
    "gpf": {
        "p_num": [1],  # Not used, but kept for consistency
        "lr": [0.001],
    },
    "gpf-plus": {
        "p_num": [5,10],  # Number of basis vectors to try
        "lr": [0.001],
    },
}

# Training set sizes to test
N_TRAIN_VALUES = [1,5,10,50]

# FIXED evaluation graphs (same for all N_train values for fair comparison)
N_EVALUATION_VAL = 100
N_EVALUATION_TEST = 100
N_EVALUATION_TOTAL = N_EVALUATION_VAL + N_EVALUATION_TEST  # 200

# Training hyperparameters
EPOCHS = 400
LR = 0.001
BATCH_SIZE = 16
PATIENCE = 30

# Wandb project name for logging downstream evaluations
DOWNSTREAM_WANDB_PROJECT = "linkpred_pretraining_downstream_eval_full"


# =============================================================================
# Configuration Extraction and Hashing
# =============================================================================

def extract_dataset_config(config: dict) -> dict:
    """
    Extract dataset configuration (everything that affects data generation).
    
    This includes:
    - GraphUniverse generation parameters (homophily, n_graphs, etc.)
    - TUDataset parameters
    - Any other dataset-specific settings
    
    Returns
    -------
    dict
        Normalized dataset config for comparison.
    """
    dataset_config = config.get("dataset", {})
    
    # Get loader parameters
    loader_params = dataset_config.get("loader", {}).get("parameters", {})
    
    # For GraphUniverse, extract generation parameters
    if "generation_parameters" in loader_params:
        gen_params = loader_params["generation_parameters"]
        return {
            "type": "GraphUniverse",
            "task": gen_params.get("task"),
            "universe_params": gen_params.get("universe_parameters", {}),
            "family_params": gen_params.get("family_parameters", {}),
        }
    
    # For other datasets, extract relevant params
    return {
        "type": dataset_config.get("loader", {}).get("_target_", "unknown"),
        "data_name": loader_params.get("data_name"),
        "parameters": dataset_config.get("parameters", {}),
    }


def extract_base_model_config(config: dict) -> dict:
    """
    Extract base model configuration (excluding pretraining-specific params).
    
    This includes:
    - Feature encoder settings
    - Backbone architecture (layers, hidden dims, dropout, etc.)
    - But EXCLUDES wrapper-specific params (augmentations, corruption, etc.)
    
    Returns
    -------
    dict
        Normalized base model config for comparison.
    """
    model_config = config.get("model", {})
    
    # Feature encoder (always included)
    feature_encoder = model_config.get("feature_encoder", {})
    
    # Backbone architecture (excluding wrapper)
    backbone = model_config.get("backbone", {})
    
    return {
        "model_name": model_config.get("model_name"),
        "feature_encoder": {
            "encoder_name": feature_encoder.get("encoder_name"),
            "out_channels": feature_encoder.get("out_channels"),
            "proj_dropout": feature_encoder.get("proj_dropout"),
        },
        "backbone": {
            "_target_": backbone.get("_target_"),
            "num_layers": backbone.get("num_layers"),
            "hidden_channels": backbone.get("hidden_channels"),
            "hidden_dim": backbone.get("hidden_dim"),
            "dropout": backbone.get("dropout"),
            "heads": backbone.get("heads"),
            "attn_type": backbone.get("attn_type"),
            # Add other architecture params but not wrapper-specific ones
        },
    }


def extract_pretraining_specific_config(config: dict) -> dict:
    """
    Extract pretraining-specific configuration.
    
    This includes:
    - Wrapper parameters (augmentations, corruption, masking, etc.)
    - Readout parameters specific to pretraining (discriminator, projection head, etc.)
    
    Returns
    -------
    dict
        Pretraining-specific config.
    """
    model_config = config.get("model", {})
    
    wrapper = model_config.get("backbone_wrapper", {})
    readout = model_config.get("readout", {})
    
    return {
        "wrapper": wrapper,
        "readout": readout,
    }


def config_to_hash(config_dict: dict) -> str:
    """
    Convert a config dict to a stable hash for comparison.
    
    Parameters
    ----------
    config_dict : dict
        Configuration dictionary.
    
    Returns
    -------
    str
        MD5 hash of the canonicalized JSON representation.
    """
    # Canonicalize: sort keys, convert to JSON
    canonical = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()


def detect_pretraining_method(config: dict) -> str:
    """
    Detect which pre-training method was used.
    
    Returns
    -------
    str
        One of: 'dgi', 'graphmae', 'graphmaev2', 'graphcl', 'linkpred', 'supervised', 'unknown'
    """
    model_config = config.get("model", {})
    wrapper_config = model_config.get("backbone_wrapper", {})
    
    if wrapper_config:
        wrapper_target = wrapper_config.get("_target_", "")
        if "DGI" in wrapper_target:
            return "dgi"
        elif "GraphMAEv2" in wrapper_target:
            return "graphmaev2"
        elif "GraphMAE" in wrapper_target:
            return "graphmae"
        elif "GraphCL" in wrapper_target:
            return "graphcl"
        elif "LinkPred" in wrapper_target:
            return "linkpred"
    
    return "supervised"


# =============================================================================
# Wandb API Integration
# =============================================================================

def fetch_runs_from_wandb(
    project_path: str,
    filters: dict = None,
    max_runs: int = None,
) -> list[dict]:
    """
    Fetch runs from wandb project via API.
    
    Parameters
    ----------
    project_path : str
        Wandb project path in format "entity/project-name".
    filters : dict, optional
        Additional filters for wandb API query.
    max_runs : int, optional
        Maximum number of runs to fetch.
    
    Returns
    -------
    list[dict]
        List of run metadata dicts.
    """
    print(f"\nFetching runs from wandb project: {project_path}")
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Fetch runs
    runs = api.runs(project_path, filters=filters)
    
    run_metadata_list = []
    
    for i, run in enumerate(runs):
        if max_runs and i >= max_runs:
            break
        
        print(f"\n[{i+1}] {run.name} ({run.id})")
        print(f"  State: {run.state}")
        print(f"  Created: {run.created_at}")
        
        # Skip failed/crashed runs
        if run.state not in ["finished", "running"]:
            print(f"  ⚠️  Skipping (state: {run.state})")
            continue
        
        # Get config
        config = run.config
        
        # Get checkpoint path from summary
        checkpoint_path = run.summary.get("best_epoch/checkpoint")
        
        if not checkpoint_path:
            print(f"  ⚠️  Skipping (no checkpoint found)")
            continue
        
        # Check if GraphUniverse
        dataset_config = config.get("dataset", {})
        loader_target = dataset_config.get("loader", {}).get("_target_", "")
        is_graph_universe = "GraphUniverse" in loader_target
        
        if not is_graph_universe:
            print(f"  ⚠️  Skipping (not GraphUniverse)")
            continue
        
        # Check if pretraining run
        wrapper_config = config.get("model", {}).get("backbone_wrapper", {})
        is_pretraining = wrapper_config is not None and len(wrapper_config) > 0
        
        if not is_pretraining:
            print(f"  ⚠️  Skipping (not a pretraining run)")
            continue
        
        # Detect pretraining method
        pretraining_method = detect_pretraining_method(config)
        model_name = config.get("model", {}).get("model_name", "unknown")
        
        print(f"  ✓ {pretraining_method} - {model_name}")
        
        # Extract configs
        dataset_cfg = extract_dataset_config(config)
        base_model_cfg = extract_base_model_config(config)
        pretraining_cfg = extract_pretraining_specific_config(config)
        
        metadata = {
            "run_id": run.id,
            "run_name": run.name,
            "run_path": run.path,
            "checkpoint_path": checkpoint_path,
            "pretraining_method": pretraining_method,
            "model_name": model_name,
            "config": config,
            "dataset_config": dataset_cfg,
            "base_model_config": base_model_cfg,
            "pretraining_config": pretraining_cfg,
            "dataset_hash": config_to_hash(dataset_cfg),
            "base_model_hash": config_to_hash(base_model_cfg),
            "pretraining_hash": config_to_hash(pretraining_cfg),
        }
        
        run_metadata_list.append(metadata)
    
    print(f"\nFetched {len(run_metadata_list)} valid pretraining runs")
    
    return run_metadata_list


# =============================================================================
# Downstream Evaluation
# =============================================================================

def run_single_downstream_eval(
    run_metadata: dict,
    mode: str,
    n_train: int,
    device: str,
    seed: int = 42,
    wandb_project: str = "downstream_eval_unified",
    p_num: int = 5,
    lr_override: float = None,
) -> dict:
    """
    Run a single downstream evaluation experiment.
    
    Parameters
    ----------
    run_metadata : dict
        Metadata from pre-training run.
    mode : str
        Evaluation mode: "finetune" or "scratch".
    n_train : int
        Number of training graphs.
    device : str
        Device to use.
    seed : int
        Random seed.
    wandb_project : str
        Wandb project for logging.
    p_num : int
        Number of basis vectors for GPF-Plus.
    lr_override : float, optional
        Override learning rate for this run (useful for hyperparameter search).
    
    Returns
    -------
    dict
        Results from downstream evaluation.
    """
    # For wandb API runs, we need to download the checkpoint
    checkpoint_path = run_metadata["checkpoint_path"]
    
    # Create descriptive wandb run name
    pretraining_method = run_metadata["pretraining_method"]
    model_name = run_metadata["model_name"]
    
    # Add p_num to name if using prompt methods
    if mode in ["gpf-plus"]:
        run_name = f"{pretraining_method}_{model_name}_{mode}_p{p_num}_n{n_train}"
    else:
        run_name = f"{pretraining_method}_{model_name}_{mode}_n{n_train}"
    
    print("\n" + "=" * 80)
    print(f"DOWNSTREAM EVALUATION: {run_name}")
    print("=" * 80)
    print(f"  Pre-training run: {run_metadata['run_id']}")
    print(f"  Mode: {mode}")
    print(f"  N_train: {n_train}")
    print(f"  Device: {device}")
    
    # Initialize wandb with original config embedded
    actual_lr = lr_override if lr_override is not None else LR
    
    wandb_config = {
        # Downstream eval settings
        "downstream_mode": mode,
        "downstream_n_train": n_train,
        "downstream_n_val": N_EVALUATION_VAL,
        "downstream_n_test": N_EVALUATION_TEST,
        "downstream_n_evaluation_total": N_EVALUATION_TOTAL,
        "downstream_epochs": EPOCHS,
        "downstream_lr": actual_lr,
        "downstream_batch_size": BATCH_SIZE,
        "downstream_patience": PATIENCE,
        "downstream_seed": seed,
        
        # Pre-training metadata
        "pretraining_run_id": run_metadata["run_id"],
        "pretraining_run_name": run_metadata["run_name"],
        "pretraining_method": pretraining_method,
        "pretraining_model_name": model_name,
        "pretraining_checkpoint": checkpoint_path,
        
        # Config hashes for grouping
        "dataset_hash": run_metadata["dataset_hash"],
        "base_model_hash": run_metadata["base_model_hash"],
        "pretraining_hash": run_metadata["pretraining_hash"],
        
        # Prompt-specific settings (if applicable)
        "downstream_p_num": p_num if mode in ["gpf-plus"] else None,
        
        # IMPORTANT: Include full original pre-training config
        "pretraining_config": run_metadata["config"],
    }
    
    wandb.init(
        project=wandb_project,
        name=run_name,
        config=wandb_config,
        tags=[
            pretraining_method,
            model_name,
            mode,
            f"n_train_{n_train}",
            "graphuniverse",
        ],
        reinit=True,
    )
    
    try:
        # Create a temporary directory structure to mimic local run_dir
        # This allows downstream_eval to load the config
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files_dir = tmp_path / "files"
            files_dir.mkdir()
            
            # Save config to temporary location
            config_path = files_dir / "config.yaml"
            import yaml
            with open(config_path, "w") as f:
                yaml.dump(run_metadata["config"], f)
            
            # Save summary with checkpoint path
            summary_path = files_dir / "wandb-summary.json"
            with open(summary_path, "w") as f:
                json.dump({"best_epoch/checkpoint": checkpoint_path}, f)
            
            # Run downstream evaluation with FIXED eval set and ADDITIONAL train graphs
            # Note: downstream_eval will handle its own wandb session
            results = run_downstream_evaluation(
                run_dir=str(tmp_path),
                checkpoint_path=checkpoint_path,
                n_evaluation_graphs=N_EVALUATION_TOTAL,
                n_train=n_train,
                mode=mode,
                epochs=EPOCHS,
                lr=actual_lr,
                batch_size=BATCH_SIZE,
                patience=PATIENCE,
                device=device,
                seed=seed,
                use_wandb=False,  # Don't use wandb inside downstream_eval, we'll handle it here
                wandb_project=wandb_project,
                p_num=p_num,
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
        
        success = True
        
    except Exception as e:
        print(f"\n❌ ERROR in downstream evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        # Only log error if wandb run is active
        if wandb.run is not None:
            try:
                wandb.log({"error": str(e)})
            except:
                pass  # Ignore if logging fails
        
        results = {"error": str(e), "test_accuracy": None}
        success = False
    
    finally:
        # Finish wandb run if it's still active
        if wandb.run is not None:
            wandb.finish()
    
    # Add metadata to results
    results["run_metadata"] = run_metadata
    results["downstream_config"] = {
        "mode": mode,
        "n_train": n_train,
        "n_evaluation_graphs": N_EVALUATION_TOTAL,
        "n_val": N_EVALUATION_VAL,
        "n_test": N_EVALUATION_TEST,
        "success": success,
    }
    
    return results


def run_downstream_eval_grid(
    wandb_project_path: str,
    device: str = "cuda",
    seed: int = 42,
    max_runs: int = None,
    downstream_wandb_project: str = "downstream_eval_unified",
    num_workers: int = 1,
    devices: list[int] = None,
    enable_prompt_tuning: bool = False,
) -> list[dict]:
    """
    Run downstream evaluation grid for runs from wandb project.
    
    Smart deduplication:
    - Scratch: One per unique (dataset_config, base_model_config, n_train)
    - Finetune: All pretraining variations per base configuration
    
    Parallelization:
    - Uses ProcessPoolExecutor to run evaluations on multiple GPUs simultaneously
    - Each worker gets assigned to a specific GPU device
    
    Parameters
    ----------
    wandb_project_path : str
        Wandb project path "entity/project-name".
    device : str
        Device type to use (default: "cuda").
    seed : int
        Random seed.
    max_runs : int, optional
        Maximum number of runs to fetch from wandb.
    downstream_wandb_project : str
        Wandb project for logging downstream results.
    num_workers : int
        Number of parallel workers (default: 1 for sequential).
    devices : list[int], optional
        List of GPU device IDs to use (e.g., [0, 1, 2, 3]).
        If None and num_workers > 1, will use devices [0, 1, ..., num_workers-1].
    enable_prompt_tuning : bool
        If True, also run prompt tuning methods (GPF, GPF-Plus) with hyperparameter grids.
    
    Returns
    -------
    list[dict]
        List of results from all downstream evaluations.
    """
    # Setup devices for parallel execution
    if devices is None and num_workers > 1:
        devices = list(range(num_workers))
    elif devices is None:
        devices = [0]  # Default single device
    
    print("\n" + "=" * 80)
    print("DOWNSTREAM EVALUATION GRID (Wandb API)")
    print("=" * 80)
    print(f"Source wandb project: {wandb_project_path}")
    print(f"Downstream wandb project: {downstream_wandb_project}")
    print(f"Device type: {device}")
    print(f"Parallel workers: {num_workers}")
    print(f"GPU devices: {devices}")
    print(f"Seed: {seed}")
    print(f"Modes: {EVAL_MODES}")
    if enable_prompt_tuning:
        print(f"Prompt modes enabled: {list(PROMPT_MODES.keys())}")
        for prompt_mode, params in PROMPT_MODES.items():
            print(f"  {prompt_mode}: {params}")
    print(f"N_train values: {N_TRAIN_VALUES}")
    print(f"N_evaluation_total: {N_EVALUATION_TOTAL} (val: {N_EVALUATION_VAL}, test: {N_EVALUATION_TEST})")
    print(f"Note: Training graphs generated ADDITIONALLY (not taken from evaluation set)")
    print("=" * 80)
    
    # Fetch runs from wandb
    run_metadata_list = fetch_runs_from_wandb(
        wandb_project_path,
        max_runs=max_runs,
    )
    
    if len(run_metadata_list) == 0:
        print("\nNo valid runs found!")
        return []
    
    # Group runs by (dataset_hash, base_model_hash)
    base_configs = {}
    for metadata in run_metadata_list:
        key = (metadata["dataset_hash"], metadata["base_model_hash"])
        if key not in base_configs:
            base_configs[key] = []
        base_configs[key].append(metadata)
    
    print(f"\n{'=' * 80}")
    print("CONFIGURATION ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Total runs: {len(run_metadata_list)}")
    print(f"Unique (dataset, base_model) combinations: {len(base_configs)}")
    
    for i, (key, runs) in enumerate(base_configs.items(), 1):
        print(f"\n[{i}] Base config with {len(runs)} pretraining variations:")
        example = runs[0]
        print(f"  Dataset: {example['dataset_config'].get('family_params', {}).get('homophily_range', 'N/A')}")
        print(f"  Model: {example['model_name']}")
        print(f"  Pretraining methods: {set(r['pretraining_method'] for r in runs)}")
    
    # Calculate experiments
    scratch_experiments = len(base_configs) * len(N_TRAIN_VALUES)
    finetune_experiments = len(run_metadata_list) * len(N_TRAIN_VALUES)
    
    # Calculate prompt tuning experiments if enabled
    prompt_experiments = 0
    if enable_prompt_tuning:
        for prompt_mode, param_grid in PROMPT_MODES.items():
            # Number of hyperparameter combinations
            num_combinations = len(param_grid.get("p_num", [1])) * len(param_grid.get("lr", [LR]))
            prompt_experiments += len(run_metadata_list) * len(N_TRAIN_VALUES) * num_combinations
    
    total_experiments = scratch_experiments + finetune_experiments + prompt_experiments
    
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT COUNT")
    print(f"{'=' * 80}")
    print(f"Scratch: {len(base_configs)} base configs × {len(N_TRAIN_VALUES)} n_train = {scratch_experiments}")
    print(f"Finetune: {len(run_metadata_list)} runs × {len(N_TRAIN_VALUES)} n_train = {finetune_experiments}")
    if enable_prompt_tuning:
        print(f"Prompt tuning: {prompt_experiments} (with hyperparameter search)")
    print(f"Total: {total_experiments}")
    
    # Confirm
    response = input("\nProceed with downstream evaluation grid? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return []
    
    # Build list of all tasks
    print("\n" + "=" * 80)
    print("BUILDING TASK LIST")
    print("=" * 80)
    
    tasks = []
    scratch_done = set()
    skipped_scratch = 0
    
    # Standard modes (finetune, scratch)
    for run_idx, run_metadata in enumerate(run_metadata_list, 1):
        for mode in EVAL_MODES:
            for n_train in N_TRAIN_VALUES:
                # For scratch: skip if already done for this base config
                if mode == "scratch":
                    scratch_key = (
                        run_metadata["dataset_hash"],
                        run_metadata["base_model_hash"],
                        n_train
                    )
                    
                    if scratch_key in scratch_done:
                        skipped_scratch += 1
                        continue
                    else:
                        scratch_done.add(scratch_key)
                
                # Add task
                tasks.append({
                    "run_metadata": run_metadata,
                    "mode": mode,
                    "n_train": n_train,
                    "seed": seed,
                    "wandb_project": downstream_wandb_project,
                    "p_num": 5,  # Default, not used for finetune/scratch
                    "lr_override": None,
                })
    
    # Prompt tuning modes (if enabled)
    if enable_prompt_tuning:
        for run_metadata in run_metadata_list:
            for prompt_mode, param_grid in PROMPT_MODES.items():
                for n_train in N_TRAIN_VALUES:
                    # Grid search over hyperparameters
                    for p_num in param_grid.get("p_num", [5]):
                        for lr in param_grid.get("lr", [LR]):
                            tasks.append({
                                "run_metadata": run_metadata,
                                "mode": prompt_mode,
                                "n_train": n_train,
                                "seed": seed,
                                "wandb_project": downstream_wandb_project,
                                "p_num": p_num,
                                "lr_override": lr,
                            })
    
    print(f"Total tasks to run: {len(tasks)}")
    if skipped_scratch > 0:
        print(f"Skipped scratch experiments (deduplicated): {skipped_scratch}")
    
    # Run grid (parallel or sequential)
    print("\n" + "=" * 80)
    print(f"STARTING DOWNSTREAM EVALUATION GRID ({'PARALLEL' if num_workers > 1 else 'SEQUENTIAL'})")
    print("=" * 80)
    
    start_time = datetime.now()
    all_results = []
    
    if num_workers > 1:
        # Parallel execution
        print(f"Running {len(tasks)} tasks on {num_workers} workers across GPUs {devices}")
        
        # Use ProcessPoolExecutor for true parallelism
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            for task_idx, task in enumerate(tasks):
                # Assign device in round-robin fashion
                device_id = devices[task_idx % len(devices)]
                device_str = f"cuda:{device_id}"
                
                future = executor.submit(
                    run_single_downstream_eval,
                    run_metadata=task["run_metadata"],
                    mode=task["mode"],
                    n_train=task["n_train"],
                    device=device_str,
                    seed=task["seed"],
                    wandb_project=task["wandb_project"],
                    p_num=task["p_num"],
                    lr_override=task["lr_override"],
                )
                future_to_task[future] = {
                    "task": task,
                    "task_idx": task_idx + 1,
                    "device": device_str,
                }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                completed += 1
                task_info = future_to_task[future]
                task = task_info["task"]
                
                try:
                    results = future.result()
                    all_results.append(results)
                    
                    acc = results.get("test_accuracy")
                    status = f"✓ Acc: {acc:.4f}" if acc is not None else "✗ Failed"
                    
                    print(f"[{completed}/{len(tasks)}] {status} | "
                          f"{task['run_metadata']['pretraining_method']} | "
                          f"{task['mode']} | n_train={task['n_train']} | "
                          f"device={task_info['device']}")
                
                except Exception as e:
                    print(f"[{completed}/{len(tasks)}] ❌ ERROR: {e}")
                    all_results.append({
                        "error": str(e),
                        "run_metadata": task["run_metadata"],
                        "downstream_config": {
                            "mode": task["mode"],
                            "n_train": task["n_train"],
                            "success": False,
                        }
                    })
    else:
        # Sequential execution (original behavior)
        for task_idx, task in enumerate(tasks, 1):
            run_metadata = task["run_metadata"]
            
            print(f"\n{'=' * 80}")
            print(f"TASK {task_idx}/{len(tasks)}")
            print(f"{'=' * 80}")
            print(f"Run ID: {run_metadata['run_id']}")
            print(f"Method: {run_metadata['pretraining_method']}")
            print(f"Model: {run_metadata['model_name']}")
            print(f"Mode: {task['mode']}, N_train: {task['n_train']}")
            
            try:
                results = run_single_downstream_eval(
                    run_metadata=run_metadata,
                    mode=task["mode"],
                    n_train=task["n_train"],
                    device=device if device.startswith("cuda:") else f"{device}:0",
                    seed=task["seed"],
                    wandb_project=task["wandb_project"],
                    p_num=task["p_num"],
                    lr_override=task["lr_override"],
                )
                
                all_results.append(results)
                
                if results.get("test_accuracy") is not None:
                    print(f"\n✓ Test Accuracy: {results['test_accuracy']:.4f}")
                else:
                    print(f"\n✗ Experiment failed")
            
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                raise
            
            except Exception as e:
                print(f"\n❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                
                all_results.append({
                    "error": str(e),
                    "run_metadata": run_metadata,
                    "downstream_config": {
                        "mode": task["mode"],
                        "n_train": task["n_train"],
                        "success": False,
                    }
                })
    
    # Final summary
    total_duration = datetime.now() - start_time
    
    print("\n" + "=" * 80)
    print("DOWNSTREAM EVALUATION GRID COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration}")
    print(f"Experiments run: {len(tasks)}")
    
    if skipped_scratch > 0:
        print(f"Skipped scratch experiments (deduplicated): {skipped_scratch}")
    
    successful = sum(1 for r in all_results if r.get("downstream_config", {}).get("success", False))
    failed = len(all_results) - successful
    
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    # Save results summary
    summary_path = Path("downstream_eval_grid_summary.json")
    summary = {
        "total_experiments": len(tasks),
        "successful": successful,
        "failed": failed,
        "duration": str(total_duration),
        "timestamp": datetime.now().isoformat(),
        "source_project": wandb_project_path,
        "parallelization": {
            "num_workers": num_workers,
            "devices": devices,
        },
        "config": {
            "modes": EVAL_MODES,
            "n_train_values": N_TRAIN_VALUES,
            "n_evaluation_total": N_EVALUATION_TOTAL,
            "n_evaluation_val": N_EVALUATION_VAL,
            "n_evaluation_test": N_EVALUATION_TEST,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
            "patience": PATIENCE,
        },
    }
    
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {summary_path}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Automated downstream evaluation using wandb API."
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        required=True,
        help="Wandb project path (entity/project-name)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--max_runs",
        type=int,
        default=None,
        help="Maximum number of runs to fetch (for testing)"
    )
    parser.add_argument(
        "--downstream_project",
        type=str,
        default=DOWNSTREAM_WANDB_PROJECT,
        help=f"Wandb project for downstream results (default: {DOWNSTREAM_WANDB_PROJECT})"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 1 for sequential)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="GPU device IDs to use (e.g., --devices 0 1 2 3). If not specified, uses [0, 1, ..., num_workers-1]"
    )
    parser.add_argument(
        "--enable_prompt_tuning",
        action="store_true",
        help="Enable prompt tuning methods (GPF, GPF-Plus) with hyperparameter grids"
    )
    
    args = parser.parse_args()
    
    results = run_downstream_eval_grid(
        wandb_project_path=args.wandb_project,
        device=args.device,
        seed=args.seed,
        max_runs=args.max_runs,
        downstream_wandb_project=args.downstream_project,
        num_workers=args.num_workers,
        devices=args.devices,
        enable_prompt_tuning=args.enable_prompt_tuning,
    )
    
    print("\n" + "=" * 80)
    print("ALL DOWNSTREAM EVALUATIONS COMPLETE")
    print("=" * 80)
    print(f"Total results: {len(results)}")
    
    return results


if __name__ == "__main__":
    main()

