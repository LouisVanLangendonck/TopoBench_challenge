import argparse
import json
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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

from downstream_eval import run_downstream_evaluation
from downstream_eval_utils import (
    DOWNSTREAM_MODES,
    downstream_mode_requires_checkpoint,
    fetch_runs_from_wandb_project,
    get_checkpoint_path_from_summary,
    load_wandb_config,
)
from grid_config_loader import (
    build_worker_devices,
    coalesce,
    coerce_optional_int_list,
    coerce_optional_str_list,
    coerce_str_list,
    load_grid_yaml,
    normalize_graphuniverse_overrides,
)


def coerce_optional_float_list(field: str, val: Any) -> list[float] | None:
    """Accept YAML null, a single float/int, or a list of floats/ints."""
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError(f"{field}: bool is not allowed; use floats or null")
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return [float(val)]
    if isinstance(val, list):
        out: list[float] = []
        for i, x in enumerate(val):
            if isinstance(x, bool):
                raise TypeError(f"{field}[{i}] must be float/int, got bool")
            if not isinstance(x, (int, float)):
                raise TypeError(f"{field}[{i}] must be float/int, got {x!r}")
            out.append(float(x))
        return out
    raise TypeError(
        f"{field} must be null, float/int, or list[float/int]; got {type(val).__name__}"
    )


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
# Default GraphUniverse Override Configurations
# =============================================================================

DEFAULT_GRAPHUNIVERSE_OVERRIDES = [
    {"family_parameters": {"homophily_range": [0.0, 0.1]}},
    {"family_parameters": {"homophily_range": [0.4, 0.6]}},
    {"family_parameters": {"homophily_range": [0.9, 1.0]}},
    {"family_parameters": {"avg_degree_range": [1.0, 2.0]}},
    {"family_parameters": {"avg_degree_range": [2.0, 3.0]}},
    {"family_parameters": {"avg_degree_range": [4.0, 5.0]}},
    {"family_parameters": {"power_law_exponent_range": [1.0, 1.0]}},
    {"family_parameters": {"power_law_exponent_range": [2.5, 2.5]}},
    {"family_parameters": {"power_law_exponent_range": [5.0, 5.0]}},
    {"universe_parameters": {"cluster_variance": 0.2}},
    {"universe_parameters": {"cluster_variance": 0.4}},
    {"universe_parameters": {"edge_propensity_variance": 0.1}},
    {"universe_parameters": {"edge_propensity_variance": 0.5}},
    {"universe_parameters": {"edge_propensity_variance": 0.9}},
    {"family_parameters": {"degree_separation_range": [0.1, 0.1]}},
    {"family_parameters": {"degree_separation_range": [0.5, 0.5]}},
    {"family_parameters": {"degree_separation_range": [0.9, 0.9]}},
]

_INDUCTIVE_SCRIPT_DEFAULTS = {
    "n_train": [5, 15],
    "tasks": ["community_detection", "community_presence"],
    "modes": [
        "linear-probe",
        "full-finetune",
        "random-init-full-finetune",
        "random-init-linear-probe",
    ],
    "readout_types": ["mean"],
    "n_evaluation_graphs": 400,
    "epochs": 300,
    "lr": 0.001,
    "batch_size": 32,
    "patience": 30,
    "classifier_dropout": 0.3,
    "input_dropout": None,
    "device": "cuda:0",
    "seed": 42,
    "wandb_project": "test_eval_grid",
    "fetch_filters": {"state": "finished"},
    "min_runs": 1,
    "save_results": True,
    "confirm_before_run": True,
    "parallel_workers": 1,
    "eval_devices": None,
    "repeat_on_different_family_seed": 1,
}


# =============================================================================
# Grid Configuration
# =============================================================================


def generate_grid_configs(
    run_dirs: list[str],
    n_train_values: list[int],
    tasks: list[str],
    modes: list[str],
    graphuniverse_overrides: list[dict | None],
    n_evaluation_graphs: int = 200,
    readout_types: List[str] = None,
    run_infos: List[Dict[str, Any]] = None,
    repeat_on_different_family_seed: int = 1,
    lr_values: List[float] = None,
    classifier_dropout_values: List[float] = None,
) -> List[Dict[str, Any]]:
    """Generate all combinations of grid parameters."""

    if readout_types is None:
        readout_types = ["mean"]
    if lr_values is None:
        lr_values = [0.001]
    if classifier_dropout_values is None:
        classifier_dropout_values = [0.0]

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
                    for readout_type in readout_types:
                        for override in graphuniverse_overrides:
                            for lr in lr_values:
                                for (
                                    classifier_dropout
                                ) in classifier_dropout_values:
                                    for repeat_idx in range(
                                        repeat_on_different_family_seed
                                    ):
                                        configs.append(
                                            {
                                                "run_dir": run_dir,
                                                "n_train": n_train,
                                                "task": task,
                                                "mode": mode,
                                                "graphuniverse_override": override,
                                                "n_evaluation_graphs": n_evaluation_graphs,
                                                "readout_type": readout_type,
                                                "pretrain_config": pretrain_config,
                                                "repeat_idx": repeat_idx,
                                                "repeat_on_different_family_seed": repeat_on_different_family_seed,
                                                "lr": lr,
                                                "classifier_dropout": classifier_dropout,
                                            }
                                        )

    return configs


def get_experiment_name(config: dict[str, Any], run_dir: str) -> str:
    """Generate a descriptive name for the experiment."""
    run_id = Path(run_dir).name

    task_abbrev = {
        "community_detection": "CD",
        "community_presence": "CP",
    }

    components = [
        run_id,
        task_abbrev.get(config["task"], config["task"][:3].upper()),
        config["mode"],
        f"n{config['n_train']}",
    ]

    components.append(f"ro_{config['readout_type']}")

    # Add lr and classifier_dropout to experiment name
    components.append(f"lr{config['lr']}")
    components.append(f"drop{config['classifier_dropout']}")

    if config["graphuniverse_override"] is not None:
        import hashlib

        override_hash = hashlib.md5(
            json.dumps(
                config["graphuniverse_override"], sort_keys=True
            ).encode()
        ).hexdigest()[:6]
        components.append(f"ov_{override_hash}")

    if config.get("repeat_on_different_family_seed", 1) > 1:
        components.append(f"rep{config['repeat_idx']}")

    return "_".join(components)


def print_grid_summary(
    configs: list[dict[str, Any]],
    *,
    parallel_workers: int = 1,
    device: str = "cuda:0",
    eval_devices: list[str] | None = None,
):
    """Print a summary of the grid configuration."""
    print("\n" + "=" * 80)
    print("DOWNSTREAM EVALUATION GRID SUMMARY")
    print("=" * 80)

    run_dirs = sorted(set(c["run_dir"] for c in configs))
    n_trains = sorted(set(c["n_train"] for c in configs))
    tasks = sorted(set(c["task"] for c in configs))
    modes = sorted(set(c["mode"] for c in configs))
    overrides = list(
        set(
            json.dumps(c["graphuniverse_override"], sort_keys=True)
            for c in configs
        )
    )
    readout_types = sorted(set(c["readout_type"] for c in configs))
    lr_values = sorted(set(c["lr"] for c in configs))
    classifier_dropout_values = sorted(
        set(c["classifier_dropout"] for c in configs)
    )

    print(f"\nRun Directories ({len(run_dirs)}):")
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"  [{i}] {run_dir}")

    print(f"\nN_train values: {n_trains}")
    print(f"Tasks: {tasks}")
    print(f"Modes: {modes}")
    print(f"Readout types: {readout_types}")
    print(f"Learning rates: {lr_values}")
    print(f"Classifier dropout values: {classifier_dropout_values}")

    # Check if we have repeat info
    repeat_count = (
        configs[0].get("repeat_on_different_family_seed", 1) if configs else 1
    )
    if repeat_count > 1:
        print(f"Repeats per setting (different family seeds): {repeat_count}")

    print(f"\nGraphUniverse Overrides ({len(overrides)}):")
    for i, override_str in enumerate(overrides, 1):
        override = json.loads(override_str)
        if override is None:
            print(f"  [{i}] None (use pretraining config)")
        else:
            print(f"  [{i}] {json.dumps(override, indent=6)}")

    print(f"\nTotal experiments: {len(configs)}")
    try:
        wdev = build_worker_devices(parallel_workers, device, eval_devices)
    except ValueError as e:
        print(f"\nDevice layout: (invalid — {e})")
    else:
        if parallel_workers > 1:
            print(
                f"\nParallel workers: {parallel_workers} (max concurrent jobs)"
            )
            print(f"  GPU map (round-robin): {wdev}")
        else:
            print(f"\nExecution: sequential on {wdev[0]}")
    print("=" * 80)


# =============================================================================
# Execution
# =============================================================================


def run_single_experiment(
    config: dict[str, Any],
    device: str,
    seed: int,
    wandb_project: str,
    epochs: int,
    batch_size: int,
    patience: int,
    input_dropout: float | None,
) -> dict[str, Any]:
    """Run a single downstream evaluation experiment."""

    run_dir = config["run_dir"]

    # Get lr and classifier_dropout from config (grid parameters)
    lr = config.get("lr", 0.001)
    classifier_dropout = config.get("classifier_dropout", 0.0)

    if downstream_mode_requires_checkpoint(config["mode"]):
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError(f"No checkpoint found for {run_dir}")
    else:
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)

    exp_name = get_experiment_name(config, run_dir)

    print("\n" + "=" * 80)
    print(f"EXPERIMENT: {exp_name}")
    print("=" * 80)
    print(f"  Run dir: {run_dir}")
    print(f"  Checkpoint: {checkpoint_path or '(none — random-init mode)'}")
    print(f"  Task: {config['task']}")
    print(f"  Mode: {config['mode']}")
    print(f"  N_train: {config['n_train']}")
    print(f"  N_evaluation: {config['n_evaluation_graphs']}")
    print(f"  Readout type: {config['readout_type']}")
    print(f"  Learning rate: {lr}")
    print(f"  Classifier dropout: {classifier_dropout}")
    if config.get("repeat_on_different_family_seed", 1) > 1:
        print(
            f"  Repeat idx: {config['repeat_idx']} / {config['repeat_on_different_family_seed']}"
        )
    if config["graphuniverse_override"] is not None:
        print(
            f"  GraphUniverse override: {json.dumps(config['graphuniverse_override'], indent=4)}"
        )
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
            repeat_idx=config.get("repeat_idx", 0),
            repeat_on_different_family_seed=config.get(
                "repeat_on_different_family_seed", 1
            ),
        )

        results["success"] = True
        results["experiment_name"] = exp_name

        if results.get("task_type") == "multilabel_bce":
            mae = results.get("test_mae", "N/A")
            if mae != "N/A":
                print(f"\n✓ Test MAE (community presence): {mae:.4f}")
            else:
                print(f"\n✓ Test MAE (community presence): {mae}")
        else:
            acc = results.get("test_accuracy", "N/A")
            if acc != "N/A":
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


def _grid_worker(payload: dict[str, Any]) -> dict[str, Any]:
    """Picklable entry point for ``ProcessPoolExecutor`` (must stay at module level)."""
    return run_single_experiment(
        config=payload["config"],
        device=payload["device"],
        seed=payload["seed"],
        wandb_project=payload["wandb_project"],
        epochs=payload["epochs"],
        batch_size=payload["batch_size"],
        patience=payload["patience"],
        input_dropout=payload["input_dropout"],
    )


def _grid_worker_indexed(
    item: tuple[int, dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    """Return (grid index, result) so the parent can preserve experiment order."""
    idx, payload = item
    return idx, _grid_worker(payload)


def _save_grid_results_json(
    configs: list[dict[str, Any]],
    all_results: list[dict[str, Any]],
    total_duration,
) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = Path(f"downstream_eval_grid_results_{timestamp}.json")
    successful = sum(1 for r in all_results if r.get("success", False))
    failed = len(all_results) - successful
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


def _run_grid_parallel(
    configs: list[dict[str, Any]],
    worker_devices: list[str],
    seed: int,
    wandb_project: str,
    epochs: int,
    batch_size: int,
    patience: int,
    input_dropout: float | None,
    save_results: bool = True,
) -> list[dict[str, Any]]:
    """Run grid with up to ``len(worker_devices)`` experiments at a time (spawn + CUDA-safe)."""
    n_slots = len(worker_devices)
    start_time = datetime.now()
    all_results: list[dict[str, Any] | None] = [None] * len(configs)

    print("\n" + "=" * 80)
    print(f"STARTING GRID EXECUTION (parallel, {n_slots} workers)")
    print(f"  Devices: {worker_devices}")
    print("=" * 80)

    tasks: list[tuple[int, dict[str, Any]]] = []
    for i, config in enumerate(configs):
        dev = worker_devices[i % n_slots]
        payload = {
            "config": config,
            "device": dev,
            "seed": seed,
            "wandb_project": wandb_project,
            "epochs": epochs,
            "batch_size": batch_size,
            "patience": patience,
            "input_dropout": input_dropout,
        }
        tasks.append((i, payload))

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_slots, mp_context=ctx) as executor:
        future_map = {
            executor.submit(_grid_worker_indexed, t): t[0] for t in tasks
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                _, result = fut.result()
                all_results[idx] = result
            except Exception as e:
                print(
                    f"\n❌ CRITICAL ERROR (job {idx + 1}/{len(configs)}): {e}"
                )
                import traceback

                traceback.print_exc()
                all_results[idx] = {
                    "success": False,
                    "error": str(e),
                    "config": configs[idx],
                }

    resolved: list[dict[str, Any]] = [
        r
        if r is not None
        else {
            "success": False,
            "error": "missing result",
            "config": configs[i],
        }
        for i, r in enumerate(all_results)
    ]

    total_duration = datetime.now() - start_time
    successful = sum(1 for r in resolved if r.get("success", False))
    failed = len(resolved) - successful

    print("\n" + "=" * 80)
    print("GRID EXECUTION COMPLETE")
    print("=" * 80)
    print(f"Total time: {total_duration}")
    print(f"Experiments completed: {len(resolved)}/{len(configs)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print("=" * 80)

    if save_results:
        _save_grid_results_json(configs, resolved, total_duration)

    return resolved


def run_grid(
    configs: list[dict[str, Any]],
    device: str,
    seed: int,
    wandb_project: str,
    epochs: int,
    batch_size: int,
    patience: int,
    input_dropout: float | None,
    save_results: bool = True,
    *,
    parallel_workers: int = 1,
    eval_devices: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run the full grid of downstream evaluations (sequential or multi-GPU parallel)."""
    worker_devices = build_worker_devices(
        parallel_workers, device, eval_devices
    )
    if parallel_workers > 1:
        return _run_grid_parallel(
            configs=configs,
            worker_devices=worker_devices,
            seed=seed,
            wandb_project=wandb_project,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            input_dropout=input_dropout,
            save_results=save_results,
        )

    start_time = datetime.now()
    all_results: list[dict[str, Any]] = []

    print("\n" + "=" * 80)
    print("STARTING GRID EXECUTION (sequential)")
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
                batch_size=batch_size,
                patience=patience,
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

            all_results.append(
                {
                    "success": False,
                    "error": str(e),
                    "config": config,
                }
            )

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
        _save_grid_results_json(configs, all_results, total_duration)

    return all_results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Grid search for downstream evaluation (YAML + CLI).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            f"  python {Path(__file__).name} --config grid_configs/grid_inductive.yaml\n"
            "  (run from downstream_adaptation/ or pass a full path)\n"
        ),
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="downstream_adaptation/grid_configs/grid_inductive.yaml",
        help="YAML file with grid settings (see grid_configs/grid_inductive.yaml).",
    )
    run_selection = parser.add_mutually_exclusive_group(required=False)
    run_selection.add_argument(
        "--run_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Pretrained wandb run directories (overrides config).",
    )
    run_selection.add_argument(
        "--wandb_pretrain_project",
        type=str,
        default=None,
        help="Wandb project to fetch runs from (overrides config).",
    )
    parser.add_argument(
        "--wandb-local-root",
        type=str,
        nargs="+",
        default=None,
        help=(
            "If set: scan ONLY these wandb roots (run-* live under <paths.output_dir>/wandb). "
            "If omitted: use TOPOBENCH_OUTPUT_DIR/wandb or legacy repo-relative paths."
        ),
    )
    parser.add_argument(
        "--min_runs",
        type=int,
        default=None,
        help="Minimum runs when fetching from wandb (default: from YAML or 1).",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        nargs="+",
        default=None,
        help="Training set sizes grid (default: from YAML or script defaults).",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["community_detection", "community_presence"],
        default=None,
        help="Downstream tasks (default: from YAML or script defaults).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        choices=list(DOWNSTREAM_MODES),
        default=None,
        help="Evaluation modes (default: from YAML or script defaults).",
    )
    parser.add_argument(
        "--graphuniverse_overrides",
        type=str,
        nargs="+",
        default=None,
        help="JSON override strings; use null for baseline. Overrides YAML list if set.",
    )
    parser.add_argument(
        "--readout_types",
        type=str,
        nargs="+",
        choices=["mean", "max", "sum"],
        default=None,
        help="Graph-level readout pooling (default: from YAML or mean).",
    )
    parser.add_argument("--n_evaluation_graphs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=None)
    parser.add_argument("--input_dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Max concurrent experiment processes (default: YAML or 1). Uses distinct GPUs.",
    )
    parser.add_argument(
        "--eval-devices",
        type=str,
        nargs="+",
        default=None,
        help="e.g. cuda:0 cuda:1 ...; length must equal --parallel-workers when parallel_workers>1.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument(
        "--repeat-on-different-family-seed",
        type=int,
        default=None,
        help="Number of times to repeat each experiment with different family seeds (default: 1).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt.",
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Do not write results JSON.",
    )

    args = parser.parse_args()
    file_cfg = load_grid_yaml(args.config) if args.config else {}

    def eff(key: str, arg_val):
        return coalesce(
            arg_val, file_cfg.get(key), _INDUCTIVE_SCRIPT_DEFAULTS.get(key)
        )

    run_dirs = coalesce(args.run_dirs, file_cfg.get("run_dirs"))
    if isinstance(run_dirs, str):
        run_dirs = [run_dirs]
    wandb_pretrain = coalesce(
        args.wandb_pretrain_project, file_cfg.get("wandb_pretrain_project")
    )
    wandb_local_roots = coalesce(
        args.wandb_local_root, file_cfg.get("wandb_local_roots")
    )
    if isinstance(wandb_local_roots, str):
        wandb_local_roots = [wandb_local_roots]

    if not run_dirs and not wandb_pretrain:
        parser.error(
            "Provide run_dirs or wandb_pretrain_project via --config YAML or CLI "
            "(see grid_configs/grid_inductive.yaml)."
        )

    n_train = coerce_optional_int_list("n_train", eff("n_train", args.n_train))
    tasks = coerce_str_list("tasks", eff("tasks", args.tasks))
    modes = coerce_str_list("modes", eff("modes", args.modes))
    readout_types = coerce_str_list(
        "readout_types", eff("readout_types", args.readout_types)
    )
    n_evaluation_graphs = eff("n_evaluation_graphs", args.n_evaluation_graphs)
    epochs = eff("epochs", args.epochs)
    lr_values = coerce_optional_float_list("lr", eff("lr", args.lr))
    batch_size = eff("batch_size", args.batch_size)
    patience = eff("patience", args.patience)
    classifier_dropout_values = coerce_optional_float_list(
        "classifier_dropout",
        eff("classifier_dropout", args.classifier_dropout),
    )
    input_dropout = eff("input_dropout", args.input_dropout)
    device = eff("device", args.device)
    pw_arg = eff("parallel_workers", args.parallel_workers)
    if pw_arg is None:
        parallel_workers = 1
    else:
        parallel_workers = int(pw_arg)
        if parallel_workers < 1:
            parser.error("parallel_workers must be >= 1")
    if args.eval_devices is not None:
        eval_devices: list[str] | None = list(args.eval_devices)
    elif file_cfg.get("eval_devices") is not None:
        eval_devices = coerce_optional_str_list(
            "eval_devices", file_cfg["eval_devices"]
        )
    else:
        eval_devices = None

    try:
        build_worker_devices(parallel_workers, device, eval_devices)
    except ValueError as e:
        parser.error(str(e))

    seed = eff("seed", args.seed)
    wandb_project = eff("wandb_project", args.wandb_project)
    fetch_filters = coalesce(
        file_cfg.get("fetch_filters"), {"state": "finished"}
    )
    min_runs = coalesce(args.min_runs, file_cfg.get("min_runs"), 1)
    repeat_on_different_family_seed = eff(
        "repeat_on_different_family_seed", args.repeat_on_different_family_seed
    )
    if repeat_on_different_family_seed is None:
        repeat_on_different_family_seed = 1
    else:
        repeat_on_different_family_seed = int(repeat_on_different_family_seed)
        if repeat_on_different_family_seed < 1:
            parser.error("repeat_on_different_family_seed must be >= 1")

    if args.graphuniverse_overrides is not None:
        parsed_overrides = []
        for override_str in args.graphuniverse_overrides:
            if override_str is None or str(override_str).lower() in (
                "null",
                "none",
            ):
                parsed_overrides.append(None)
            else:
                try:
                    parsed_overrides.append(json.loads(override_str))
                except json.JSONDecodeError as e:
                    print(
                        f"ERROR: Invalid JSON in GraphUniverse override: {override_str}\n  {e}"
                    )
                    sys.exit(1)
        print(
            f"\n✓ GraphUniverse overrides from CLI ({len(parsed_overrides)} entries)"
        )
    elif file_cfg.get("graphuniverse_overrides") is not None:
        parsed_overrides = normalize_graphuniverse_overrides(
            file_cfg["graphuniverse_overrides"]
        )
        print(
            f"\n✓ GraphUniverse overrides from YAML ({len(parsed_overrides)} entries)"
        )
    else:
        parsed_overrides = list(DEFAULT_GRAPHUNIVERSE_OVERRIDES)
        print(
            f"\n✓ Using built-in DEFAULT_GRAPHUNIVERSE_OVERRIDES ({len(parsed_overrides)} entries)"
        )

    save_results = coalesce(file_cfg.get("save_results"), True)
    if args.no_save:
        save_results = False
    confirm_before_run = (
        coalesce(file_cfg.get("confirm_before_run"), True) and not args.yes
    )

    if wandb_pretrain:
        print(f"\n{'=' * 80}")
        print("FETCHING RUNS FROM WANDB PROJECT")
        print(f"{'=' * 80}")
        print(f"Project: {wandb_pretrain}")
        run_infos = fetch_runs_from_wandb_project(
            project_path=wandb_pretrain,
            filters=fetch_filters,
            min_runs=min_runs,
            wandb_local_roots=wandb_local_roots,
        )
        run_dirs = [info["run_dir"] for info in run_infos]
        print(f"\n✓ Found {len(run_dirs)} runs")
        print(f"  Run IDs: {[info['run_id'] for info in run_infos]}")
    else:
        print(f"\n✓ Using {len(run_dirs)} run directories from config/CLI")
        run_infos = []
        for run_dir in run_dirs:
            pretrain_config = load_wandb_config(run_dir)
            run_infos.append(
                {"run_dir": run_dir, "pretrain_config": pretrain_config}
            )
        print(f"  Loaded pretraining configs for {len(run_infos)} runs")

    configs = generate_grid_configs(
        run_dirs=run_dirs,
        n_train_values=n_train,
        tasks=tasks,
        modes=modes,
        graphuniverse_overrides=parsed_overrides,
        n_evaluation_graphs=n_evaluation_graphs,
        readout_types=readout_types,
        run_infos=run_infos,
        repeat_on_different_family_seed=repeat_on_different_family_seed,
        lr_values=lr_values,
        classifier_dropout_values=classifier_dropout_values,
    )

    print_grid_summary(
        configs,
        parallel_workers=parallel_workers,
        device=device,
        eval_devices=eval_devices,
    )

    if confirm_before_run:
        response = input("\nProceed with grid execution? [y/N]: ")
        if response.lower() != "y":
            print("Aborted.")
            return None

    results = run_grid(
        configs=configs,
        device=device,
        seed=seed,
        wandb_project=wandb_project,
        epochs=epochs,
        batch_size=batch_size,
        patience=patience,
        input_dropout=input_dropout,
        save_results=save_results,
        parallel_workers=parallel_workers,
        eval_devices=eval_devices,
    )

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 80)

    return results


if __name__ == "__main__":
    main()
