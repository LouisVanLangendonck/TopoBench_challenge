import argparse
import sys
import json
import itertools
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(_THIS_DIR))

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("ERROR: wandb not installed. This script requires wandb.")
    sys.exit(1)

from downstream_eval_clean import run_downstream_evaluation, load_wandb_config, get_checkpoint_path_from_summary


# =============================================================================
# JSON Serialization Helper
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# =============================================================================
# Wandb Project Fetching
# =============================================================================

def fetch_runs_from_wandb_project(
    project_path: str,
    filters: dict | None = None,
    min_runs: int = 1,
) -> List[Dict[str, Any]]:
    """
    Fetch all runs from a wandb project and extract their local paths and configs.
    """
    api = wandb.Api()
    
    if "/" in project_path:
        entity, project = project_path.split("/", 1)
    else:
        entity = None
        project = project_path
    
    print(f"\n{'=' * 80}")
    print(f"FETCHING RUNS FROM WANDB PROJECT: {project_path}")
    print(f"{'=' * 80}")
    
    if filters is None:
        filters = {"state": "finished"}
    
    if entity:
        runs = api.runs(f"{entity}/{project}", filters=filters)
    else:
        runs = api.runs(project, filters=filters)
    
    run_infos = []
    
    for run in runs:
        run_id = run.id
        run_name = run.name
        
        checkpoint_path = run.summary.get("best_epoch/checkpoint")
        
        if checkpoint_path is None:
            print(f"  ⚠️  Skipping {run_id} ({run_name}): no checkpoint in summary")
            continue
        
        run_dir = None
        
        potential_wandb_dirs = [
            Path("data/outputs/wandb"),
            Path("wandb"),
        ]
        
        for wandb_dir in potential_wandb_dirs:
            if wandb_dir.exists():
                for run_path in wandb_dir.iterdir():
                    if run_path.is_dir() and run_id in run_path.name:
                        run_dir = str(run_path)
                        break
            if run_dir:
                break
        
        if run_dir is None:
            print(f"  ⚠️  Skipping {run_id} ({run_name}): local run directory not found")
            continue
        
        config = dict(run.config)
        pretrain_config = load_wandb_config(run_dir)
        
        run_infos.append({
            "run_dir": run_dir,
            "run_id": run_id,
            "run_name": run_name,
            "config": config,
            "checkpoint_path": checkpoint_path,
            "pretrain_config": pretrain_config,
        })
        
        print(f"  ✓ {run_id} ({run_name})")
        print(f"    Dir: {run_dir}")
        print(f"    Checkpoint: {checkpoint_path}")
    
    print(f"\n{'=' * 80}")
    print(f"FOUND {len(run_infos)} RUNS WITH VALID CHECKPOINTS")
    print(f"{'=' * 80}\n")
    
    if len(run_infos) < min_runs:
        raise ValueError(
            f"Expected at least {min_runs} runs, but only found {len(run_infos)} "
            f"with valid checkpoints in project {project_path}"
        )
    
    return run_infos


# =============================================================================
# Default GraphUniverse Override Configurations
# =============================================================================

DEFAULT_GRAPHUNIVERSE_OVERRIDES = [
    None
]


# =============================================================================
# Grid Configuration
# =============================================================================

def generate_grid_configs(
    run_dirs: List[str],
    n_train_values: List[int],
    tasks: List[str],
    modes: List[str],
    graphuniverse_overrides: List[Dict | None],
    n_evaluation_graphs: int = 200,
    readout_types: List[str] = None,
    run_infos: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Generate all combinations of grid parameters."""
    
    if readout_types is None:
        readout_types = ["mean"]
    
    run_dir_to_config = {}
    if run_infos is not None:
        for info in run_infos:
            run_dir_to_config[info["run_dir"]] = info.get("pretrain_config")
    
    configs = []
    
    for run_dir in run_dirs:
        pretrain_config = run_dir_to_config.get(run_dir)
        
        for n_train in n_train_values:
            for task in tasks:
                for mode in modes:
                    for override in graphuniverse_overrides:
                        # For graph-level tasks, iterate over readout types
                        if task in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
                            for readout_type in readout_types:
                                configs.append({
                                    "run_dir": run_dir,
                                    "n_train": n_train,
                                    "task": task,
                                    "mode": mode,
                                    "graphuniverse_override": override,
                                    "n_evaluation_graphs": n_evaluation_graphs,
                                    "readout_type": readout_type,
                                    "pretrain_config": pretrain_config,
                                })
                        else:
                            configs.append({
                                "run_dir": run_dir,
                                "n_train": n_train,
                                "task": task,
                                "mode": mode,
                                "graphuniverse_override": override,
                                "n_evaluation_graphs": n_evaluation_graphs,
                                "readout_type": readout_types[0],
                                "pretrain_config": pretrain_config,
                            })
    
    return configs


def get_experiment_name(config: Dict[str, Any], run_dir: str) -> str:
    """Generate a descriptive name for the experiment."""
    run_id = Path(run_dir).name
    
    task_abbrev = {
        "basic_property_reconstruction": "BPR",
        "community_related_property_reconstruction": "CPR",
    }
    
    components = [
        run_id,
        task_abbrev.get(config["task"], config["task"][:3].upper()),
        config["mode"],
        f"n{config['n_train']}",
    ]
    
    if config["task"] in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
        components.append(f"ro_{config['readout_type']}")
    
    if config["graphuniverse_override"] is not None:
        import hashlib
        override_hash = hashlib.md5(
            json.dumps(config["graphuniverse_override"], sort_keys=True).encode()
        ).hexdigest()[:6]
        components.append(f"ov_{override_hash}")
    
    return "_".join(components)


def print_grid_summary(configs: List[Dict[str, Any]]):
    """Print a summary of the grid configuration."""
    print("\n" + "=" * 80)
    print("DOWNSTREAM EVALUATION GRID SUMMARY")
    print("=" * 80)
    
    run_dirs = sorted(set(c["run_dir"] for c in configs))
    n_trains = sorted(set(c["n_train"] for c in configs))
    tasks = sorted(set(c["task"] for c in configs))
    modes = sorted(set(c["mode"] for c in configs))
    overrides = list(set(json.dumps(c["graphuniverse_override"], sort_keys=True) for c in configs))
    readout_types = sorted(set(c["readout_type"] for c in configs))
    
    print(f"\nRun Directories ({len(run_dirs)}):")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  [{i}] {run_dir}")
    
    print(f"\nN_train values: {n_trains}")
    print(f"Tasks: {tasks}")
    print(f"Modes: {modes}")
    print(f"Readout types: {readout_types}")
    
    print(f"\nGraphUniverse Overrides ({len(overrides)}):")
    for i, override_str in enumerate(overrides, 1):
        override = json.loads(override_str)
        if override is None:
            print(f"  [{i}] None (use pretraining config)")
        else:
            print(f"  [{i}] {json.dumps(override, indent=6)}")
    
    print(f"\nTotal experiments: {len(configs)}")
    print("=" * 80)


# =============================================================================
# Execution
# =============================================================================

def run_single_experiment(
    config: Dict[str, Any],
    device: str,
    seed: int,
    wandb_project: str,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    classifier_dropout: float,
    input_dropout: float | None,
) -> Dict[str, Any]:
    """Run a single downstream evaluation experiment."""
    
    run_dir = config["run_dir"]
    
    checkpoint_path = get_checkpoint_path_from_summary(run_dir)
    if checkpoint_path is None:
        raise ValueError(f"No checkpoint found for {run_dir}")
    
    exp_name = get_experiment_name(config, run_dir)
    
    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"  Run dir: {run_dir}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Task: {config['task']}")
    print(f"  Mode: {config['mode']}")
    print(f"  N_train: {config['n_train']}")
    print(f"  N_evaluation: {config['n_evaluation_graphs']}")
    if config["task"] in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
        print(f"  Readout type: {config['readout_type']}")
    if config["graphuniverse_override"] is not None:
        print(f"  GraphUniverse override: {json.dumps(config['graphuniverse_override'], indent=4)}")
    print("=" * 80)
    
    try:
        results = run_downstream_evaluation(
            run_dir=run_dir,
            n_evaluation_graphs=config["n_evaluation_graphs"],
            n_train=config["n_train"],
            mode=config["mode"],
            downstream_task=config["task"],
            readout_type=config["readout_type"],
            graphuniverse_override=config["graphuniverse_override"],
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            patience=patience,
            device=device,
            seed=seed,
            use_wandb=True,
            wandb_project=wandb_project,
            classifier_dropout=classifier_dropout,
            input_dropout=input_dropout,
            pretraining_config=config.get("pretrain_config"),
        )
        
        results["success"] = True
        results["experiment_name"] = exp_name
        
        # Print result
        if config["task"] in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
            mae = results.get('test_mae_weighted', 'N/A')
            if mae != 'N/A':
                print(f"\n✓ Test Weighted MAE: {mae:.4f}")
            else:
                print(f"\n✓ Test Weighted MAE: {mae}")
        else:
            acc = results.get('test_accuracy', 'N/A')
            if acc != 'N/A':
                print(f"\n✓ Test Accuracy: {acc:.4f}")
            else:
                print(f"\n✓ Test Accuracy: {acc}")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        results = {
            "success": False,
            "error": str(e),
            "experiment_name": exp_name,
        }
    
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
                print("\n✓ Wandb run properly closed")
        except:
            pass
    
    results["config"] = config
    
    return results


def run_grid(
    configs: List[Dict[str, Any]],
    device: str,
    seed: int,
    wandb_project: str,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    classifier_dropout: float,
    input_dropout: float | None,
    save_results: bool = True,
) -> List[Dict[str, Any]]:
    """Run the full grid of downstream evaluations."""
    
    start_time = datetime.now()
    all_results = []
    
    print("\n" + "=" * 80)
    print("STARTING GRID EXECUTION")
    print("=" * 80)
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT {i}/{len(configs)}")
        print(f"{'=' * 80}")
        
        try:
            results = run_single_experiment(
                config=config,
                device=device,
                seed=seed,
                wandb_project=wandb_project,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                patience=patience,
                classifier_dropout=classifier_dropout,
                input_dropout=input_dropout,
            )
            all_results.append(results)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        
        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                "success": False,
                "error": str(e),
                "config": config,
            })
    
    total_duration = datetime.now() - start_time
    successful = sum(1 for r in all_results if r.get("success", False))
    failed = len(all_results) - successful
    
    print("\n" + "=" * 80)
    print("GRID EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration}")
    print(f"Experiments completed: {len(all_results)}/{len(configs)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 80)
    
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = Path(f"downstream_eval_grid_results_{timestamp}.json")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_experiments": len(configs),
            "completed": len(all_results),
            "successful": successful,
            "failed": failed,
            "duration": str(total_duration),
            "results": all_results,
        }
        
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nResults saved to {results_path}")
    
    return all_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Grid search for downstream evaluation (clean version).",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    run_selection = parser.add_mutually_exclusive_group(required=True)
    run_selection.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        help="List of pretrained model run directories"
    )
    run_selection.add_argument(
        "--wandb_pretrain_project",
        type=str,
        help="Wandb project to fetch all runs from (format: 'entity/project' or 'project')"
    )
    
    parser.add_argument(
        "--n_train",
        type=int,
        nargs="+",
        default=[5, 15, 25, 50, 100, 200],
        help="List of training set sizes to test (default: 5 15 25 50 100 200)"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["basic_property_reconstruction", "community_related_property_reconstruction"],
        default=["basic_property_reconstruction", "community_related_property_reconstruction"],
        help="List of tasks to evaluate (default: basic_property_reconstruction community_related_property_reconstruction)"
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=["linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "scratch_frozen"],
        default=["linear", "finetune-linear", "scratch", "scratch_frozen"],
        help="List of evaluation modes (default: linear finetune-linear scratch scratch_frozen)"
    )
    parser.add_argument(
        "--graphuniverse_overrides",
        type=str,
        nargs="+",
        default=None,
        help="List of GraphUniverse override JSON strings. Use 'null' for no override."
    )
    parser.add_argument(
        "--readout_types",
        type=str,
        nargs="+",
        choices=["mean", "max", "sum"],
        default=["mean"],
        help="List of readout types for graph-level tasks (default: mean)"
    )
    
    parser.add_argument(
        "--n_evaluation_graphs",
        type=int,
        default=400,
        help="Number of fixed evaluation graphs (default: 400)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs (default: 300)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (default: 30)"
    )
    parser.add_argument(
        "--classifier_dropout",
        type=float,
        default=0.3,
        help="Dropout rate for classifier (default: 0.3)"
    )
    parser.add_argument(
        "--input_dropout",
        type=float,
        default=None,
        help="Dropout rate for encoder output (default: None)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="test_eval_grid",
        help="Wandb project for logging (default: downstream_eval_grid)"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Determine run directories
    if args.wandb_pretrain_project:
        print(f"\n{'=' * 80}")
        print("FETCHING RUNS FROM WANDB PROJECT")
        print(f"{'=' * 80}")
        print(f"Project: {args.wandb_pretrain_project}")
        
        run_infos = fetch_runs_from_wandb_project(
            project_path=args.wandb_pretrain_project,
            filters={"state": "finished"},
            min_runs=1,
        )
        
        run_dirs = [info["run_dir"] for info in run_infos]
        
        print(f"\n✓ Found {len(run_dirs)} runs with valid checkpoints")
        print(f"  Run IDs: {[info['run_id'] for info in run_infos]}")
    else:
        run_dirs = args.run_dirs
        print(f"\n✓ Using manually specified run directories ({len(run_dirs)} runs)")
        
        run_infos = []
        for run_dir in run_dirs:
            pretrain_config = load_wandb_config(run_dir)
            run_infos.append({
                "run_dir": run_dir,
                "pretrain_config": pretrain_config,
            })
        print(f"  Loaded pretraining configs for {len(run_infos)} runs")
    
    # Parse GraphUniverse overrides
    if args.graphuniverse_overrides is None:
        parsed_overrides = DEFAULT_GRAPHUNIVERSE_OVERRIDES
        print(f"\n✓ Using DEFAULT_GRAPHUNIVERSE_OVERRIDES from script ({len(parsed_overrides)} configurations)")
    else:
        parsed_overrides = []
        for override_str in args.graphuniverse_overrides:
            if override_str is None or override_str.lower() == "null" or override_str.lower() == "none":
                parsed_overrides.append(None)
            else:
                try:
                    parsed_overrides.append(json.loads(override_str))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON in GraphUniverse override: {override_str}")
                    print(f"  {e}")
                    sys.exit(1)
        print(f"\n✓ Using GraphUniverse overrides from command line ({len(parsed_overrides)} configurations)")
    
    # Generate grid configurations
    configs = generate_grid_configs(
        run_dirs=run_dirs,
        n_train_values=args.n_train,
        tasks=args.tasks,
        modes=args.modes,
        graphuniverse_overrides=parsed_overrides,
        n_evaluation_graphs=args.n_evaluation_graphs,
        readout_types=args.readout_types,
        run_infos=run_infos,
    )
    
    print_grid_summary(configs)
    
    response = input("\nProceed with grid execution? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    results = run_grid(
        configs=configs,
        device=args.device,
        seed=args.seed,
        wandb_project=args.wandb_project,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        classifier_dropout=args.classifier_dropout,
        input_dropout=args.input_dropout,
        save_results=not args.no_save,
    )
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()

