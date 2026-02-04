"""
Downstream evaluation of pre-trained TopoBench models for NODE-LEVEL and GRAPH-LEVEL tasks.

This script loads a pre-trained model (e.g., from DGI, GraphMAE, GraphCL),
extracts the encoder components, generates a new dataset with the same
configuration, and runs downstream evaluation with multiple modes:
1. Linear probing (frozen encoder + linear classifier)
2. MLP probing (frozen encoder + MLP classifier)
3. Fine-tuning (unfrozen encoder + MLP classifier)
4. Scratch (randomly initialized encoder + MLP classifier)

Supported tasks:
- NODE-LEVEL: GraphUniverse community detection (predicts community label for each node)
  Number of classes = K (from universe_parameters)
- GRAPH-LEVEL: GraphUniverse triangle counting (predicts number of triangles per graph)
  Regression task (loss_type: mae)

FAIR EVALUATION METHODOLOGY (for GraphUniverse):
- Generates (n_evaluation + n_train) graphs TOGETHER with the same seed
- First n_evaluation graphs: FIXED evaluation set (val + test), same for all n_train values
- Next n_train graphs: training set, additional to evaluation set
- This ensures all models are evaluated on the SAME test set regardless of n_train
- All graphs from same generation = same distribution, fair comparison

GRAPHUNIVERSE OVERRIDE:
- You can override specific GraphUniverse generation parameters for downstream evaluation
- Edit GRAPHUNIVERSE_OVERRIDE_DEFAULT below to set a default override
- Or use --graphuniverse_override from command line with JSON string

WANDB TRACKING FOR OVERRIDES:
When using wandb, the following config fields are logged for easy plotting/filtering:
- graphuniverse_override: Full nested dict of override parameters
- has_override: Boolean flag (True/False)
- override_hash: Short hash for identifying unique override configs
- override/*: Flattened override parameters (e.g., override/family_parameters/homophily_range)
- override_label/homophily: Human-readable label ("high", "low", or range)
- override_label/graph_size: Graph size range if overridden
- override_label/max_communities: Max communities if overridden

Example wandb queries for plotting:
- Group by homophily: Use "override_label/homophily" as X-axis or color
- Filter by override: has_override == True
- Compare specific params: override/family_parameters/homophily_range
- Plot by n_train with different overrides: X=n_train, Color=override_label/homophily

Designed to work both locally in the topobench repo and in separate
repos where topobench is pip-installed.

Usage:
    # Node-level classification (community detection):
    python downstream_eval.py --run_dir /path/to/wandb/run --n_evaluation_graphs 200 --n_train 20 --mode finetune
    
    # Graph-level Classification (triangle counting):
    python downstream_eval.py --run_dir /path/to/wandb/run/community_detection \
        --n_train 50 --mode finetune --downstream_task triangle_counting --readout_type sum
    
    # With GraphUniverse override (high homophily):
    python downstream_eval.py --run_dir /path/to/wandb/run --n_train 50 --mode finetune \
        --graphuniverse_override '{"family_parameters": {"homophily_range": [0.9, 1.0]}}'
"""

# =============================================================================
# GraphUniverse Override Configuration
# =============================================================================
# Edit this dict to override GraphUniverse generation parameters for downstream evaluation.
# This is used as the default when running this script directly (if --graphuniverse_override not specified).
# Set to None or {} to use the original pretraining config without modifications.
#
# Examples:
#
# 1. Test on high homophily graphs:
# GRAPHUNIVERSE_OVERRIDE_DEFAULT = {
#     "family_parameters": {
#         "homophily_range": [0.9, 1.0],
#     }
# }
#
# 2. Test on larger graphs with more communities:
# GRAPHUNIVERSE_OVERRIDE_DEFAULT = {
#     "family_parameters": {
#         "min_n_nodes": 100,
#         "max_n_nodes": 300,
#         "max_communities": 10,
#     }
# }
#
# 3. Test on low homophily graphs:
# GRAPHUNIVERSE_OVERRIDE_DEFAULT = {
#     "family_parameters": {
#         "homophily_range": [0.0,0.05],
#     }
# }

GRAPHUNIVERSE_OVERRIDE_DEFAULT = None #{
#     "family_parameters": {
#         "homophily_range": [0.0,1.0],
#     }
# }  # Set to None to use pretraining config as-is

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

# Allow running from local repo
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if (_REPO_ROOT / "topobench").exists():
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import yaml
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# TopoBench imports
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import DataloadDataset


# =============================================================================
# Configuration Loading
# =============================================================================

def load_wandb_config(run_dir: str | Path) -> dict:
    """Load and unwrap config from wandb run directory."""
    run_dir = Path(run_dir)
    config_path = run_dir / "files" / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    
    config = {}
    for key, val in raw_config.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "value" in val:
            config[key] = val["value"]
        else:
            config[key] = val
    
    return config


def get_checkpoint_path_from_summary(run_dir: str | Path) -> str | None:
    """Extract checkpoint path from wandb-summary.json."""
    run_dir = Path(run_dir)
    summary_path = run_dir / "files" / "wandb-summary.json"
    
    if not summary_path.exists():
        return None
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    return summary.get("best_epoch/checkpoint")


# =============================================================================
# Dataset Generation
# =============================================================================

def create_dataset_from_config(
    config: dict,
    n_graphs: int | None = None,
    n_train: int | None = None,  # Add this parameter
    subsample_train: bool = False,  # Add this parameter
    data_dir: str | None = None,
    seed: int = 42,
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,  # Override task for downstream eval
) -> tuple:
    """
    Create/load a dataset using the same configuration as a pre-trained run.
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    n_graphs : int, optional
        Override number of graphs. Only applies to GraphUniverse datasets.
    n_train : int, optional
        For TUDataset: number of training samples to subsample if subsample_train=True
    subsample_train : bool
        For TUDataset: whether to subsample from training set
    data_dir : str, optional
        Override data directory. If None, uses config value.
    seed : int
        Random seed for dataset generation.
    graphuniverse_override : dict, optional
        Override specific GraphUniverse generation parameters. 
        Example: {'family_parameters': {'homophily_range': [0.9, 1.0], 'max_communities': 10}}
        Will recursively update the generation_parameters from the original config.
    downstream_task : str, optional
        Override the task for downstream evaluation (e.g., "triangle_counting").
        This changes the task in the generation_parameters.
    
    Returns
    -------
    tuple
        (dataset, data_dir, dataset_info) - the loaded dataset, its directory, and metadata
    """
    dataset_config = deepcopy(config["dataset"])
    
    # Determine loader type
    loader_target = dataset_config["loader"]["_target_"]
    
    if "GraphUniverse" in loader_target:
        dataset, data_dir = _create_graph_universe_dataset(
            dataset_config, n_graphs, data_dir, seed, graphuniverse_override, downstream_task
        )
        return dataset, data_dir, {"type": "GraphUniverse", "subsample_info": None}
    else:
        dataset, data_dir = _create_standard_dataset(dataset_config, data_dir)
        dataset_info = {
            "type": "TUDataset", 
            "total_size": len(dataset),
            "subsample_train": subsample_train,
            "n_train_requested": n_train if subsample_train else None
        }
        return dataset, data_dir, dataset_info


def create_dataset_from_config_v2(
    config: dict,
    n_graphs: int | None = None,
    seed: int = 42,
    dataset_purpose: str = "eval",  # "eval" or "train"
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
    data_dir: str | None = None,
) -> tuple:
    """
    Create/load a dataset using v2 approach with independent seed handling.
    
    This version ensures that eval and train datasets are generated independently
    with different family seeds but the same universe seed.
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    n_graphs : int, optional
        Number of graphs to generate
    seed : int
        Base seed (from pretraining config)
    dataset_purpose : str
        Either "eval" or "train" - determines which family seed to use
    graphuniverse_override : dict, optional
        Override specific GraphUniverse generation parameters
    downstream_task : str, optional
        Override the task for downstream evaluation
    data_dir : str, optional
        Override data directory. If None, uses config value.
    
    Returns
    -------
    tuple
        (dataset, data_dir, dataset_info) - the loaded dataset, its directory, and metadata
    """
    dataset_config = deepcopy(config["dataset"])
    
    # Determine loader type
    loader_target = dataset_config["loader"]["_target_"]
    
    if "GraphUniverse" in loader_target:
        dataset, data_dir = _create_graph_universe_dataset_v2(
            dataset_config, n_graphs, data_dir, seed, dataset_purpose, 
            graphuniverse_override, downstream_task
        )
        return dataset, data_dir, {"type": "GraphUniverse", "subsample_info": None}
    else:
        # For non-GraphUniverse, fall back to standard approach
        dataset, data_dir = _create_standard_dataset(dataset_config, data_dir)
        dataset_info = {
            "type": "TUDataset", 
            "total_size": len(dataset),
            "subsample_train": False,
            "n_train_requested": None
        }
        return dataset, data_dir, dataset_info

def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """
    Recursively update base_dict with values from update_dict.
    
    Parameters
    ----------
    base_dict : dict
        Base dictionary to update.
    update_dict : dict
        Dictionary with values to update.
    
    Returns
    -------
    dict
        Updated dictionary (modifies base_dict in place and returns it).
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _print_config_diff(original: dict, updated: dict, prefix: str = ""):
    """
    Print differences between original and updated configurations.
    
    Parameters
    ----------
    original : dict
        Original configuration.
    updated : dict
        Updated configuration.
    prefix : str
        Prefix for nested keys (used in recursion).
    """
    all_keys = set(original.keys()) | set(updated.keys())
    
    for key in sorted(all_keys):
        full_key = f"{prefix}.{key}" if prefix else key
        
        if key not in original:
            print(f"  + {full_key}: {updated[key]} (NEW)")
        elif key not in updated:
            print(f"  - {full_key}: {original[key]} (REMOVED)")
        elif isinstance(original[key], dict) and isinstance(updated[key], dict):
            # Recursively check nested dicts
            _print_config_diff(original[key], updated[key], full_key)
        elif original[key] != updated[key]:
            print(f"  ✎ {full_key}: {original[key]} → {updated[key]}")


def _create_graph_universe_dataset(
    dataset_config: dict,
    n_graphs: int | None,
    data_dir: str | None,
    seed: int,
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
) -> tuple:
    """Create GraphUniverse dataset with optional overrides."""
    from graph_universe import GraphUniverseDataset
    
    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    # Store original for diff printing
    original_gen_params = deepcopy(gen_params)
    
    # Override task if specified (for transfer learning)
    # IMPORTANT: For property reconstruction, we still need to generate with a valid task
    # (e.g., community_detection) to get the node labels needed for property computation
    if downstream_task is not None:
        if downstream_task in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
            # Use community_detection for graph generation (we need labels for property/homophily computation)
            gen_params["task"] = "community_detection"
            print(f"  ✓ Downstream task: {downstream_task}")
            print(f"  ✓ Generating graphs with: community_detection (needed for labels/properties)")
        else:
            gen_params["task"] = downstream_task
            print(f"  ✓ Overriding task: {original_gen_params.get('task')} → {downstream_task}")
    
    # Override number of graphs if specified
    if n_graphs is not None:
        gen_params["family_parameters"]["n_graphs"] = n_graphs
    
    # Override seed
    # Increases the seed by 1 to have completely new graphs for downstream evaluation
    gen_params["family_parameters"]["seed"] = seed + 1 
    # Keeps the same seed for the universe parameters to have the same distribution of graphs
    gen_params["universe_parameters"]["seed"] = seed
    
    # Apply GraphUniverse overrides if specified
    has_overrides = graphuniverse_override is not None and len(graphuniverse_override) > 0
    if has_overrides:
        print("\n" + "=" * 80)
        print("APPLYING GRAPHUNIVERSE OVERRIDES")
        print("=" * 80)
        _deep_update(gen_params, graphuniverse_override)
    
    # Override data directory if specified
    root_dir = data_dir if data_dir else params["data_dir"]
    
    # Add suffix to distinguish from original dataset
    if n_graphs is not None:
        root_dir = f"{root_dir}_downstream_{n_graphs}graphs_seed{seed}"
    
    # Add task suffix if task was overridden
    if downstream_task is not None:
        root_dir = f"{root_dir}_task_{downstream_task}"
    
    # Add override suffix to data dir if overrides were applied
    if has_overrides:
        import hashlib
        import json
        override_hash = hashlib.md5(json.dumps(graphuniverse_override, sort_keys=True).encode()).hexdigest()[:8]
        root_dir = f"{root_dir}_override_{override_hash}"
    
    print(f"\nCreating GraphUniverse dataset with {gen_params['family_parameters']['n_graphs']} graphs...")
    print(f"Data directory: {root_dir}")
    
    # Print configuration differences
    print("\n" + "-" * 80)
    print("GRAPHUNIVERSE CONFIG COMPARISON (Pretraining → Downstream)")
    print("-" * 80)
    
    if has_overrides:
        print("Changes from pretraining config:")
        _print_config_diff(original_gen_params, gen_params)
    else:
        print("No overrides specified - using pretraining config (except seeds)")
    
    print("\n" + "-" * 80)
    print("FINAL DOWNSTREAM CONFIG:")
    print("-" * 80)
    print(f"Family parameters:")
    for k, v in gen_params.get("family_parameters", {}).items():
        print(f"  {k}: {v}")
    print(f"Universe parameters:")
    for k, v in gen_params.get("universe_parameters", {}).items():
        print(f"  {k}: {v}")
    print("-" * 80 + "\n")
    
    dataset = GraphUniverseDataset(
        root=root_dir,
        parameters=gen_params,
    )
    
    return dataset, dataset.raw_dir


def _create_graph_universe_dataset_v2(
    dataset_config: dict,
    n_graphs: int | None,
    data_dir: str | None,
    base_seed: int,
    dataset_purpose: str = "eval",  # "eval" or "train"
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
) -> tuple:
    """
    Create GraphUniverse dataset with proper seed separation.
    
    CRITICAL: Uses different family seeds for eval vs train to ensure independence.
    - eval:  family_seed = base_seed + 1
    - train: family_seed = base_seed + 2
    
    This guarantees that eval set is ALWAYS the same regardless of train set size.
    
    Parameters
    ----------
    dataset_config : dict
        Dataset configuration from wandb run
    n_graphs : int, optional
        Number of graphs to generate
    data_dir : str, optional
        Override data directory
    base_seed : int
        Base seed (from pretraining config)
    dataset_purpose : str
        Either "eval" or "train" - determines which family seed to use
    graphuniverse_override : dict, optional
        Override specific GraphUniverse generation parameters
    downstream_task : str, optional
        Override the task for downstream evaluation
        
    Returns
    -------
    tuple
        (dataset, data_dir) - the loaded dataset and its directory
    """
    from graph_universe import GraphUniverseDataset
    
    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    # Store original for diff printing
    original_gen_params = deepcopy(gen_params)
    
    # Override task if specified
    if downstream_task is not None:
        if downstream_task in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
            gen_params["task"] = "community_detection"
            print(f"  ✓ Downstream task: {downstream_task}")
            print(f"  ✓ Generating graphs with: community_detection (needed for labels/properties)")
        else:
            gen_params["task"] = downstream_task
            print(f"  ✓ Overriding task: {original_gen_params.get('task')} → {downstream_task}")
    
    # Override number of graphs if specified
    if n_graphs is not None:
        gen_params["family_parameters"]["n_graphs"] = n_graphs
    
    # CRITICAL: Set seeds based on dataset purpose
    # - Universe seed: ALWAYS the same (same distribution)
    # - Family seed: DIFFERENT for eval vs train (independence)
    gen_params["universe_parameters"]["seed"] = base_seed
    
    if dataset_purpose == "eval":
        gen_params["family_parameters"]["seed"] = base_seed + 1
        print(f"  ✓ EVAL set: universe_seed={base_seed}, family_seed={base_seed + 1}")
    elif dataset_purpose == "train":
        gen_params["family_parameters"]["seed"] = base_seed + 2
        print(f"  ✓ TRAIN set: universe_seed={base_seed}, family_seed={base_seed + 2}")
    else:
        raise ValueError(f"Unknown dataset_purpose: {dataset_purpose}. Must be 'eval' or 'train'.")
    
    # Apply GraphUniverse overrides if specified
    has_overrides = graphuniverse_override is not None and len(graphuniverse_override) > 0
    if has_overrides:
        print("\n" + "=" * 80)
        print("APPLYING GRAPHUNIVERSE OVERRIDES")
        print("=" * 80)
        _deep_update(gen_params, graphuniverse_override)
    
    # Override data directory
    root_dir = data_dir if data_dir else params["data_dir"]
    
    # Add suffix to distinguish datasets
    root_dir = f"{root_dir}_{dataset_purpose}_{n_graphs}graphs_seed{base_seed}"
    
    if downstream_task is not None:
        root_dir = f"{root_dir}_task_{downstream_task}"
    
    if has_overrides:
        import hashlib
        import json
        override_hash = hashlib.md5(json.dumps(graphuniverse_override, sort_keys=True).encode()).hexdigest()[:8]
        root_dir = f"{root_dir}_override_{override_hash}"
    
    print(f"\nCreating GraphUniverse dataset ({dataset_purpose})...")
    print(f"  Number of graphs: {gen_params['family_parameters']['n_graphs']}")
    print(f"  Data directory: {root_dir}")
    
    # Print configuration
    print("\n" + "-" * 80)
    print("GRAPHUNIVERSE CONFIG:")
    print("-" * 80)
    if has_overrides:
        print("Changes from pretraining config:")
        _print_config_diff(original_gen_params, gen_params)
    else:
        print("No overrides - using pretraining config (except seeds)")
    print("-" * 80 + "\n")
    
    dataset = GraphUniverseDataset(
        root=root_dir,
        parameters=gen_params,
    )
    
    return dataset, dataset.raw_dir


def _create_standard_dataset(dataset_config: dict, data_dir: str | None) -> tuple:
    """Create standard dataset (TUDataset, etc.) from config."""
    import hydra
    
    loader_config = dataset_config["loader"]
    if data_dir:
        loader_config["parameters"]["data_dir"] = data_dir
    
    loader = hydra.utils.instantiate(loader_config)
    dataset, dataset_dir = loader.load()
    
    return dataset, dataset_dir


def apply_transforms(
    dataset,
    data_dir: str,
    transforms_config: dict | None,
) -> PreProcessor:
    """
    Apply the same transforms that were used during pre-training.
    
    Parameters
    ----------
    dataset : Dataset
        The loaded dataset.
    data_dir : str
        Directory containing the data.
    transforms_config : dict, optional
        Transform configuration from the pre-trained run.
    
    Returns
    -------
    PreProcessor
        Preprocessed dataset with transforms applied.
    """
    if transforms_config:
        print(f"Applying transforms: {list(transforms_config.keys())}")
        transforms_omega = OmegaConf.create(transforms_config)
    else:
        print("No transforms to apply.")
        transforms_omega = None
    
    preprocessor = PreProcessor(dataset, data_dir, transforms_omega)
    return preprocessor


# =============================================================================
# Model Loading & Feature Extraction
# =============================================================================

def get_class_from_target(target: str):
    """Get class from Hydra _target_ string."""
    from topobench.nn.backbones import MODEL_CLASSES as BACKBONE_CLASSES
    from topobench.nn.encoders import FEATURE_ENCODERS
    from topobench.nn.readouts import READOUT_CLASSES
    from topobench.nn.wrappers import WRAPPER_CLASSES
    
    parts = target.split(".")
    class_name = parts[-1]
    
    if class_name in BACKBONE_CLASSES:
        return BACKBONE_CLASSES[class_name]
    if class_name in FEATURE_ENCODERS:
        return FEATURE_ENCODERS[class_name]
    if class_name in READOUT_CLASSES:
        return READOUT_CLASSES[class_name]
    if class_name in WRAPPER_CLASSES:
        return WRAPPER_CLASSES[class_name]
    
    import importlib
    module_path = ".".join(parts[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def instantiate_from_config(config: dict, **override_kwargs):
    """Instantiate a class from Hydra-style config."""
    from functools import partial
    
    if config is None:
        return None
    
    config = dict(config)
    target = config.pop("_target_")
    is_partial = config.pop("_partial_", False)
    config.update(override_kwargs)
    
    cls = get_class_from_target(target)
    
    if is_partial:
        return partial(cls, **config)
    return cls(**config)


def detect_pretraining_method(config: dict) -> str:
    """
    Detect which pre-training method was used based on config.
    
    Returns
    -------
    str
        One of: 'dgi', 'graphmae', 'graphmaev2', 'dgmae', 'graphcl', 'linkpred', 's2gae', 'higmae', 'supervised', 'unknown'
    """
    model_config = config.get("model", {})
    wrapper_config = model_config.get("backbone_wrapper", {})
    
    if wrapper_config:
        wrapper_target = wrapper_config.get("_target_", "")
        if "DGI" in wrapper_target:
            return "dgi"
        elif "GraphMAEv2" in wrapper_target:
            return "graphmaev2"
        elif "DGMAE" in wrapper_target or "dgmae" in wrapper_target.lower():
            return "dgmae"
        elif "GraphMAE" in wrapper_target:
            return "graphmae"
        elif "GraphCL" in wrapper_target:
            return "graphcl"
        elif "LinkPred" in wrapper_target:
            return "linkpred"
        elif "S2GAE" in wrapper_target or "s2gae" in wrapper_target.lower():
            return "s2gae"
        elif "HiGMAE" in wrapper_target or "higmae" in wrapper_target.lower():
            return "higmae"
    
    # Check loss
    loss_config = config.get("loss", {})
    dataset_loss = loss_config.get("dataset_loss", {})
    loss_target = dataset_loss.get("_target_", "")
    
    if "DGI" in loss_target:
        return "dgi"
    elif "GraphMAEv2" in loss_target:
        return "graphmaev2"
    elif "DGMAE" in loss_target or "dgmae" in loss_target.lower():
        return "dgmae"
    elif "GraphMAE" in loss_target:
        return "graphmae"
    elif "GraphCL" in loss_target:
        return "graphcl"
    elif "LinkPred" in loss_target:
        return "linkpred"
    elif "S2GAE" in loss_target or "s2gae" in loss_target.lower():
        return "s2gae"
    elif "HiGMAE" in loss_target or "higmae" in loss_target.lower():
        return "higmae"
    
    return "supervised"


def create_random_encoder(
    config: dict,
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module, int]:
    """
    Create encoder with random initialization (for scratch baseline).
    
    Same architecture as pre-trained model but with random weights.
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    device : str
        Device to load on.
    
    Returns
    -------
    tuple
        (feature_encoder, backbone, hidden_dim) with random weights.
    """
    model_config = config["model"]
    
    # Build feature encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
    else:
        feature_encoder = nn.Identity()
    
    # Build backbone (clean, without wrapper)
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    
    # Get hidden dimension from config - different backbones use different keys
    # GPS uses hidden_dim, GCN uses hidden_channels, others might use out_channels
    hidden_dim = (
        backbone_config.get("hidden_dim") or 
        backbone_config.get("hidden_channels") or 
        backbone_config.get("out_channels") or
        model_config.get("feature_encoder", {}).get("out_channels") or
        64
    )
    
    # Move to device (weights are already randomly initialized)
    feature_encoder = feature_encoder.to(device)
    backbone = backbone.to(device)
    
    print(f"  Created random encoder: {type(backbone).__name__}")
    print(f"  Hidden dim: {hidden_dim}")
    
    return feature_encoder, backbone, hidden_dim

def detect_task_level(config: dict) -> str:
    """
    Detect whether this is a node-level or graph-level task.
    
    Returns
    -------
    str
        "node" or "graph"
    """
    dataset_params = config.get("dataset", {}).get("parameters", {})
    task_level = dataset_params.get("task_level", "node")
    
    print(f"Detected task level: {task_level}")
    return task_level

def load_pretrained_encoder(
    config: dict,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module, int]:
    """
    Load the pre-trained feature encoder and backbone.
    
    Handles different pre-training methods (DGI, GraphMAE, GraphCL) by
    extracting the clean backbone from their wrappers.
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    checkpoint_path : str or Path
        Path to the checkpoint file.
    device : str
        Device to load on.
    
    Returns
    -------
    tuple
        (feature_encoder, backbone, hidden_dim)
        - feature_encoder: The feature encoding layer
        - backbone: The clean GNN backbone (unwrapped from DGI/GraphMAE/etc.)
        - hidden_dim: Output dimension of the backbone
    """
    model_config = config["model"]
    pretraining_method = detect_pretraining_method(config)
    print(f"Detected pre-training method: {pretraining_method}")
    
    # Build feature encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
    else:
        print("No feature encoder found, using identity!")
        feature_encoder = nn.Identity()
    
    # Build backbone (clean, without wrapper)
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    
    # Get hidden dimension from config - different backbones use different keys
    # GPS uses hidden_dim, GCN uses hidden_channels, others might use out_channels
    hidden_dim = (
        backbone_config.get("hidden_dim") or 
        backbone_config.get("hidden_channels") or 
        backbone_config.get("out_channels") or
        model_config.get("feature_encoder", {}).get("out_channels") or
        64
    )
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Debug: show all top-level key prefixes in checkpoint
    prefixes = set(k.split('.')[0] for k in state_dict.keys())
    print(f"  Checkpoint modules: {prefixes}")
    
    # Filter state dict for encoder components
    encoder_state = {}
    backbone_state = {}
    wrapper_state = {}  # Architectural components (ln_i) to keep
    wrapper_only_state = {}  # Pre-training specific params to discard
    
    # Check if this is a wrapped backbone (DGI, GraphMAE, GraphMAEv2, DGMAE, GraphCL, LinkPred, S2GAE, HiGMAE, etc.)
    has_wrapper = pretraining_method in ["dgi", "graphmae", "graphmaev2", "dgmae", "graphcl", "linkpred", "s2gae", "higmae"]
    
    # Get expected backbone parameter names for filtering
    expected_backbone_keys = set(backbone.state_dict().keys())
    
    # DEBUG: Show a few expected keys
    print(f"  Expected backbone keys (first 5): {list(expected_backbone_keys)[:5]}")
    print(f"  Total expected backbone keys: {len(expected_backbone_keys)}")
    
    # Pre-training specific params to skip (not architectural)
    pretraining_only_params = [
        "enc_mask_token", "dec_mask_token",  # Masking tokens
        "projector", "predictor",  # Projection heads
        "encoder_ema", "projector_ema",  # EMA models
        "discriminator",  # DGI discriminator
    ]
    
    for key, value in state_dict.items():
        if key.startswith("feature_encoder."):
            new_key = key.replace("feature_encoder.", "", 1)
            encoder_state[new_key] = value
        elif key.startswith("backbone."):
            # Strip first "backbone." prefix
            stripped_key = key.replace("backbone.", "", 1)
            
            if has_wrapper:
                # For wrapped backbones, structure is:
                # TBModel.backbone (wrapper) -> wrapper.backbone (actual backbone)
                # Keys in checkpoint: "backbone.backbone.convs.X" (double nested)
                # or "backbone.ln_0.weight" (wrapper architectural)
                # or "backbone.enc_mask_token" (wrapper pre-training only)
                
                # If still starts with "backbone.", strip it again (double nesting)
                if stripped_key.startswith("backbone."):
                    final_key = stripped_key.replace("backbone.", "", 1)
                    # This is definitely a backbone param
                    backbone_state[final_key] = value
                else:
                    # Single "backbone." prefix - could be:
                    # 1. Direct backbone param (shouldn't happen with proper wrapping)
                    # 2. Wrapper architectural param (ln_i) - KEEP
                    # 3. Wrapper pre-training param (enc_mask_token, projector) - DISCARD
                    
                    if stripped_key in expected_backbone_keys:
                        # Direct backbone param
                        backbone_state[stripped_key] = value
                    elif stripped_key.startswith("ln_"):
                        # Wrapper architectural: LayerNorm - KEEP
                        wrapper_state[stripped_key] = value
                    elif any(stripped_key.startswith(p) for p in pretraining_only_params):
                        # Wrapper pre-training specific - DISCARD
                        wrapper_only_state[stripped_key] = value
                    else:
                        # Unknown wrapper param - be conservative and keep
                        print(f"  WARNING: Unknown wrapper param '{stripped_key}', keeping it")
                        wrapper_state[stripped_key] = value
            else:
                # No wrapper, direct backbone
                backbone_state[stripped_key] = value
    
    # Load weights into feature encoder
    if encoder_state:
        feature_encoder.load_state_dict(encoder_state, strict=True)
        print(f"  Loaded feature encoder weights ({len(encoder_state)} keys)")
    
    # Report wrapper architectural params being kept
    if wrapper_state:
        print(f"  Keeping {len(wrapper_state)} wrapper architectural params: {list(wrapper_state.keys())}")
        print(f"  (These are part of the encoding architecture: LayerNorms, etc.)")
    
    # Report wrapper-only params that are being skipped
    if wrapper_only_state:
        print(f"  Skipping {len(wrapper_only_state)} pre-training-only params: {list(wrapper_only_state.keys())[:5]}...")
        print(f"  (These are pre-training specific: mask tokens, projectors, EMA, etc.)")
    
    # Create clean wrapper if we have wrapper architectural components
    if wrapper_state and has_wrapper:
        from topobench.nn.wrappers.graph.clean_inference_wrapper import CleanInferenceWrapper
        
        # Get num_cell_dimensions from model config
        wrapper_config = model_config.get("backbone_wrapper", {})
        num_cell_dimensions = wrapper_config.get("num_cell_dimensions", 1)
        out_channels = wrapper_config.get("out_channels", hidden_dim)
        
        # Determine if manual layer iteration was used during pre-training
        # Only S2GAE uses manual iteration through convs
        manual_iteration = (pretraining_method == "s2gae")
        
        print(f"  Creating CleanInferenceWrapper (preserves architecture)")
        clean_wrapper = CleanInferenceWrapper(
            backbone=backbone,
            out_channels=out_channels,
            num_cell_dimensions=num_cell_dimensions,
            residual_connections=True,
            manual_iteration=manual_iteration,
        )
        if manual_iteration:
            print(f"  Using manual layer iteration (S2GAE-style)")
        else:
            print(f"  Using standard backbone forward pass")
        
        # Load backbone weights into the wrapper's backbone
        clean_wrapper.backbone.load_state_dict(backbone_state, strict=True)
        print(f"  Loaded backbone weights into wrapper ({len(backbone_state)} keys) [STRICT]")
        
        # Load wrapper architectural weights (ln_i)
        wrapper_load_result = clean_wrapper.load_state_dict(wrapper_state, strict=False)
        if wrapper_load_result.unexpected_keys:
            print(f"  WARNING: Unexpected wrapper keys: {wrapper_load_result.unexpected_keys}")
        print(f"  Loaded wrapper architectural weights ({len(wrapper_state)} keys)")
        
        # Use the wrapped backbone for downstream
        backbone = clean_wrapper
    else:
        # No wrapper or no wrapper state, just load backbone directly
        if backbone_state:
            backbone.load_state_dict(backbone_state, strict=True)
            print(f"  Loaded backbone weights ({len(backbone_state)} keys) [STRICT]")
    
    # Move to device
    feature_encoder = feature_encoder.to(device)
    backbone = backbone.to(device)
    
    print(f"  Clean backbone type: {type(backbone).__name__}")
    
    return feature_encoder, backbone, hidden_dim


def prepare_batch_for_topobench(batch):
    """
    Prepare a batch for TopoBench models by ensuring required attributes exist.
    
    TopoBench expects x_0, batch_0 etc. but standard PyG DataLoader creates x, batch.
    This function adds the required aliases.
    """
    # Ensure x_0 exists (TopoBench convention for rank-0 cell features)
    if not hasattr(batch, 'x_0') and hasattr(batch, 'x'):
        batch.x_0 = batch.x
    
    # Ensure batch_0 exists (TopoBench convention for rank-0 batch indices)
    if not hasattr(batch, 'batch_0') and hasattr(batch, 'batch'):
        batch.batch_0 = batch.batch
    
    return batch


def verify_encoder_outputs(encoder: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    """
    Verify that the encoder produces meaningful outputs.
    
    Checks:
    1. Outputs are not all zeros
    2. Outputs have variance (not constant)
    3. Outputs have reasonable magnitude
    
    Returns a dict with diagnostic info.
    """
    encoder.eval()
    encoder = encoder.to(device)
    
    # Get one batch
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = prepare_batch_for_topobench(batch)
    
    with torch.no_grad():
        try:
            features = encoder(batch)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    # Compute statistics
    mean = features.mean().item()
    std = features.std().item()
    min_val = features.min().item()
    max_val = features.max().item()
    num_zeros = (features == 0).sum().item()
    total_elements = features.numel()
    
    # Check for issues
    issues = []
    if std < 1e-6:
        issues.append("Very low variance (outputs nearly constant)")
    if num_zeros == total_elements:
        issues.append("All outputs are zero")
    if abs(mean) > 1000:
        issues.append(f"Unusually large mean: {mean}")
    
    status = "OK" if not issues else "WARNING"
    
    return {
        "status": status,
        "mean": mean,
        "std": std,
        "min": min_val,
        "max": max_val,
        "zero_fraction": num_zeros / total_elements,
        "shape": list(features.shape),
        "issues": issues,
    }


class PretrainedEncoder(nn.Module):
    """
    Wrapper combining feature encoder + backbone for clean inference.
    
    This provides a simple interface to get encoded NODE-LEVEL features
    from a pre-trained model, without any pre-training specific logic
    (no corruption, no masking, etc.).
    
    For node-level tasks (like community detection), returns per-node embeddings.
    """
    
    def __init__(
        self,
        feature_encoder: nn.Module,
        backbone: nn.Module,
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
    
    def forward(self, batch):
        """
        Get encoded NODE-LEVEL features from the pre-trained model.
        
        Parameters
        ----------
        batch : torch_geometric.data.Data
            Input batch with x/x_0, edge_index, batch/batch_0 attributes.
        
        Returns
        -------
        torch.Tensor
            Node-level encoded features. Shape: (num_nodes, hidden_dim)
        """
        # Prepare batch with TopoBench-expected attributes
        batch = prepare_batch_for_topobench(batch)
        
        # DEBUG: Check input features
        input_x = batch.x_0 if hasattr(batch, 'x_0') else batch.x
        
        # Feature encoding
        batch_encoded = self.feature_encoder(batch)
        
        # Extract the encoded features
        if hasattr(batch_encoded, 'x_0'):
            x = batch_encoded.x_0
        elif isinstance(batch_encoded, dict):
            x = batch_encoded.get('x_0', batch_encoded.get('x'))
        else:
            x = batch_encoded
        
        # DEBUG: Check if feature encoder actually changed the features
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG PretrainedEncoder]")
            print(f"  Input shape: {input_x.shape}")
            print(f"  After feature_encoder: {x.shape}")
            print(f"  Feature encoder type: {type(self.feature_encoder).__name__}")
            print(f"  Features changed: {not torch.allclose(input_x, x) if input_x.shape == x.shape else 'shape changed'}")
            self._debug_printed = True
        
        # Get other attributes from original batch
        edge_index = batch.edge_index
        batch_indices = batch.batch_0 if hasattr(batch, 'batch_0') else batch.batch
        edge_weight = getattr(batch, 'edge_weight', None)
        
        # Backbone encoding - returns node-level features
        node_features = self.backbone(
            x,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        return node_features


# =============================================================================
# Downstream Classifiers
# =============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier for probing."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Initialize with small weights for better finetuning
        # This prevents the random head from overwhelming pretrained features
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.dropout(x)  # Dropout on encoder output
        return self.linear(x)


class MLPClassifier(nn.Module):
    """MLP classifier with configurable layers."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.5,
        input_dropout: float = None,  # Separate dropout for encoder output
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        # Use input_dropout if specified, otherwise use same dropout rate
        if input_dropout is None:
            input_dropout = dropout
        
        layers = []
        
        # Add dropout on encoder output FIRST
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights for better finetuning
        # This prevents random gradients from destroying pretrained features
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for better finetuning."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Xavier with small gain for hidden layers
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)


class LinearRegressor(nn.Module):
    """Simple linear regressor for probing."""
    
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Initialize with small weights for better finetuning
        # For regression, even smaller init to avoid large initial predictions
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.dropout(x)  # Dropout on encoder output
        return self.linear(x)


class MLPRegressor(nn.Module):
    """MLP regressor with configurable layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        dropout: float = 0.5,
        input_dropout: float = None,  # Separate dropout for encoder output
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        # Use input_dropout if specified, otherwise use same dropout rate
        if input_dropout is None:
            input_dropout = dropout
        
        layers = []
        
        # Add dropout on encoder output FIRST
        if input_dropout > 0:
            layers.append(nn.Dropout(input_dropout))
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
        
        # Initialize with small weights for better finetuning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for better finetuning."""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                # Xavier with small gain for hidden layers
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# =============================================================================
# Property Reconstruction Components
# =============================================================================

class MultiPropertyRegressor(nn.Module):
    """
    Multi-head regressor for basic graph property reconstruction.
    
    Predicts 4 basic properties from graph-level embeddings:
    - avg_degree: average degree of nodes [0, inf)
    - size: number of nodes [0, inf)
    - gini: GINI coefficient of degree distribution [0, 1]
    - diameter: graph diameter [0, inf)
    
    NOTE: homophily, community_presence, and edge_prob_matrix are handled by CommunityPropertyRegressor
    
    Each property has its own prediction head (can be linear or MLP).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        input_dropout: float = None,
        use_mlp_heads: bool = True,
        K: int = 10,  # Number of communities for community_presence and edge_prob_matrix
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of graph-level embeddings from encoder.
        hidden_dim : int, optional
            Hidden dimension for MLP heads. If None, uses input_dim // 2.
        dropout : float
            Dropout rate for hidden layers in MLP heads.
        input_dropout : float, optional
            Dropout on encoder output. If None, uses dropout value.
        use_mlp_heads : bool
            Whether to use MLP heads (True) or linear heads (False).
        K : int
            Number of communities (for community_presence and edge_prob_matrix).
        """
        super().__init__()
        
        self.K = K
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)
        
        if input_dropout is None:
            input_dropout = dropout
        
        # Shared input dropout (applied to encoder output)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        # Property-specific prediction heads
        # IMPORTANT: We predict on ORIGINAL scale (not normalized)
        # Use appropriate activations for each property's natural range
        if use_mlp_heads:
            # MLP heads: input -> hidden -> output
            # For unbounded properties, use ELU+1 which gives smooth [1, inf)
            # This avoids dead neurons at 0 and gives better gradients
            self.head_avg_degree = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            # No activation - let it learn the right scale!
            
            self.head_size = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            # No activation - let it learn the right scale!
            
            self.head_gini = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # [0, 1] - naturally bounded
            )
            
            self.head_diameter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            # No activation - let it learn the right scale!
        else:
            # Linear heads with appropriate activations
            self.head_avg_degree = nn.Linear(input_dim, 1)
            # No activation
            
            self.head_size = nn.Linear(input_dim, 1)
            # No activation
            
            self.head_gini = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()  # [0, 1]
            )
            
            self.head_diameter = nn.Linear(input_dim, 1)
            # No activation
        
        # Initialize all heads with small weights for better finetuning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for better finetuning."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small initialization to avoid overwhelming pretrained features
                # and to start with predictions close to 0 before activation
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Predict all properties from graph-level embeddings.
        
        Parameters
        ----------
        x : torch.Tensor
            Graph-level embeddings. Shape: (batch_size, input_dim)
        
        Returns
        -------
        dict
            Dictionary with predicted values for each property:
            {
                'avg_degree': Tensor of shape (batch_size, 1),
                'size': Tensor of shape (batch_size, 1),
                'gini': Tensor of shape (batch_size, 1),
                'diameter': Tensor of shape (batch_size, 1),
            }
        """
        # Apply shared input dropout
        x = self.input_dropout(x)
        
        # Predict each property independently
        return {
            'avg_degree': self.head_avg_degree(x),
            'size': self.head_size(x),
            'gini': self.head_gini(x),
            'diameter': self.head_diameter(x),
        }


class CommunityPropertyRegressor(nn.Module):
    """
    Multi-head regressor for community-related graph property reconstruction.
    
    Predicts 4 community-related properties:
    - homophily: fraction of edges connecting same-label nodes [0, 1] (GRAPH-LEVEL)
    - community_presence: binary vector [K] indicating which communities are present (GRAPH-LEVEL)
    - edge_prob_matrix: symmetric matrix [K, K] of inter-community edge probabilities (GRAPH-LEVEL)
    - community_detection: node-level community labels (K-class classification) (NODE-LEVEL)
    
    NOTE: This is a MIXED task-level model:
    - homophily, community_presence, edge_prob_matrix: use graph-level embeddings
    - community_detection: uses node-level embeddings
    
    Each property has its own prediction head (can be linear or MLP).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        input_dropout: float = None,
        use_mlp_heads: bool = True,
        K: int = 10,  # Number of communities
    ):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of node/graph embeddings from encoder.
        hidden_dim : int, optional
            Hidden dimension for MLP heads. If None, uses input_dim // 2.
        dropout : float
            Dropout rate for hidden layers in MLP heads.
        input_dropout : float, optional
            Dropout on encoder output. If None, uses dropout value.
        use_mlp_heads : bool
            Whether to use MLP heads (True) or linear heads (False).
        K : int
            Number of communities (for community_presence, edge_prob_matrix, and community_detection).
        """
        super().__init__()
        
        self.K = K
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)
        
        if input_dropout is None:
            input_dropout = dropout
        
        # Shared input dropout (applied to encoder output)
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        # Property-specific prediction heads
        if use_mlp_heads:
            # Homophily head: [0, 1] bounded (GRAPH-LEVEL)
            self.head_homophily = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()  # [0, 1] - naturally bounded
            )
            
            # Community presence head: predicts K-dimensional binary vector (GRAPH-LEVEL)
            self.head_community_presence = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, K),
                nn.Sigmoid()  # [0, 1] for each community
            )
            
            # Edge prob matrix head: predicts K*K values, reshaped to [K, K] (GRAPH-LEVEL)
            self.head_edge_prob_matrix = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, K * K),
                nn.Sigmoid()  # [0, 1] for each edge probability
            )
            
            # Community detection head: K-class classification (NODE-LEVEL)
            # Takes node embeddings, outputs K logits per node
            self.head_community_detection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, K),
                # No activation - logits for CrossEntropyLoss
            )
        else:
            # Linear heads with appropriate activations
            self.head_homophily = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()  # [0, 1]
            )
            
            # Community presence head (linear)
            self.head_community_presence = nn.Sequential(
                nn.Linear(input_dim, K),
                nn.Sigmoid()  # [0, 1] for each community
            )
            
            # Edge prob matrix head (linear)
            self.head_edge_prob_matrix = nn.Sequential(
                nn.Linear(input_dim, K * K),
                nn.Sigmoid()  # [0, 1] for each edge probability
            )
            
            # Community detection head (linear, no activation) (NODE-LEVEL)
            self.head_community_detection = nn.Linear(input_dim, K)
        
        # Initialize all heads with small weights for better finetuning
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for better finetuning."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_graph=None, x_node=None):
        """
        Predict all community-related properties.
        
        IMPORTANT: This model handles mixed task levels:
        - Graph-level properties (homophily, community_presence, edge_prob_matrix): use x_graph
        - Node-level property (community_detection): uses x_node
        
        Parameters
        ----------
        x_graph : torch.Tensor, optional
            Graph-level embeddings. Shape: (batch_size, input_dim)
            Required for: homophily, community_presence, edge_prob_matrix
        x_node : torch.Tensor, optional
            Node-level embeddings. Shape: (num_nodes, input_dim)
            Required for: community_detection
        
        Returns
        -------
        dict
            Dictionary with predicted values for each property:
            {
                'homophily': Tensor of shape (batch_size, 1),
                'community_presence': Tensor of shape (batch_size, K),
                'edge_prob_matrix': Tensor of shape (batch_size, K, K),
                'community_detection': Tensor of shape (num_nodes, K) - logits for classification
            }
        """
        result = {}
        
        # Graph-level properties
        if x_graph is not None:
            x_graph_dropout = self.input_dropout(x_graph)
            
            # Homophily
            result['homophily'] = self.head_homophily(x_graph_dropout)
            
            # Community presence
            result['community_presence'] = self.head_community_presence(x_graph_dropout)
            
            # Edge prob matrix: reshape from (batch_size, K*K) to (batch_size, K, K)
            edge_prob_flat = self.head_edge_prob_matrix(x_graph_dropout)
            batch_size = x_graph.size(0)
            edge_prob_matrix = edge_prob_flat.view(batch_size, self.K, self.K)
            
            # Make edge_prob_matrix symmetric by averaging with transpose
            edge_prob_matrix = (edge_prob_matrix + edge_prob_matrix.transpose(1, 2)) / 2
            result['edge_prob_matrix'] = edge_prob_matrix
        
        # Node-level property
        if x_node is not None:
            x_node_dropout = self.input_dropout(x_node)
            
            # Community detection: predict K logits for each node
            result['community_detection'] = self.head_community_detection(x_node_dropout)
        
        return result


def extract_normalization_scales_from_config(config: dict) -> dict:
    """
    Extract normalization scales from GraphUniverse config.
    
    Uses the max values from generation parameters:
    - homophily: 1.0 (already normalized)
    - avg_degree: max from avg_degree_range
    - size: max_n_nodes
    - gini: 1.0 (already normalized)
    - diameter: max_n_nodes (upper bound, though actual is usually much smaller)
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    
    Returns
    -------
    dict
        Normalization scales for each property.
    """
    dataset_config = config.get("dataset", {})
    loader_target = dataset_config.get("loader", {}).get("_target_", "")
    
    if "GraphUniverse" in loader_target:
        gen_params = dataset_config.get("loader", {}).get("parameters", {}).get("generation_parameters", {})
        family_params = gen_params.get("family_parameters", {})
        
        # Extract max values from ranges
        max_n_nodes = family_params.get("max_n_nodes", 200)
        avg_degree_range = family_params.get("avg_degree_range", [2.0, 10.0])
        max_avg_degree = max(avg_degree_range)
        
        scales = {
            'homophily': 1.0,  # Already [0, 1]
            'avg_degree': max_avg_degree,
            'size': float(max_n_nodes),
            'gini': 1.0,  # Already [0, 1]
            'diameter': float(max_n_nodes),  # Upper bound (actual diameter usually much smaller)
        }
        
        print(f"\n  Extracted normalization scales from GraphUniverse config:")
        print(f"    max_n_nodes: {max_n_nodes}")
        print(f"    max_avg_degree: {max_avg_degree}")
        
        return scales
    else:
        # Default scales for non-GraphUniverse datasets
        return {
            'homophily': 1.0,
            'avg_degree': 20.0,
            'size': 200.0,
            'gini': 1.0,
            'diameter': 20.0,
        }


def extract_property_targets_from_batch(batch):
    """
    Extract pre-computed property targets from a batch of graphs.
    
    Properties should have been pre-computed and stored as attributes:
    - property_homophily
    - property_avg_degree
    - property_size
    - property_gini
    - property_diameter
    - property_community_presence
    - property_edge_prob_matrix
    
    Parameters
    ----------
    batch : torch_geometric.data.Batch
        Batched graphs with pre-computed property attributes.
    
    Returns
    -------
    dict
        Dictionary with target values for each property:
        {
            'homophily': Tensor of shape (batch_size, 1),
            'avg_degree': Tensor of shape (batch_size, 1),
            'size': Tensor of shape (batch_size, 1),
            'gini': Tensor of shape (batch_size, 1),
            'diameter': Tensor of shape (batch_size, 1),
            'community_presence': Tensor of shape (batch_size, K),
            'edge_prob_matrix': Tensor of shape (batch_size, K, K),
        }
    """
    # Check if properties are pre-computed
    if not hasattr(batch, 'property_homophily'):
        raise ValueError(
            "Graph properties not found! Please pre-compute properties using "
            "add_properties_to_dataset() from graph_properties.py before training."
        )
    
    # Extract simple properties and ensure shape [batch_size, 1]
    # PyG batching may result in shapes [batch_size] or [batch_size, 1]
    properties = {}
    for prop_name in ['avg_degree', 'size', 'gini', 'diameter']:
        prop_tensor = getattr(batch, f'property_{prop_name}')
        
        # Ensure shape is [batch_size, 1]
        if prop_tensor.dim() == 1:
            prop_tensor = prop_tensor.unsqueeze(1)
        
        properties[prop_name] = prop_tensor
    
    # Extract complex properties (already have correct shapes from batching)
    # community_presence: [batch_size, K]
    # edge_prob_matrix: [batch_size, K, K]
    if hasattr(batch, 'property_community_presence'):
        properties['community_presence'] = batch.property_community_presence
    
    if hasattr(batch, 'property_edge_prob_matrix'):
        properties['edge_prob_matrix'] = batch.property_edge_prob_matrix
    
    return properties


def extract_community_property_targets_from_batch(batch):
    """
    Extract pre-computed community-related property targets from a batch of graphs.
    
    NOTE: This handles MIXED task levels:
    - homophily, community_presence, edge_prob_matrix: GRAPH-LEVEL targets
    - community_detection: NODE-LEVEL targets (each node has a community label)
    
    Properties should have been pre-computed and stored as attributes:
    - property_homophily (graph-level)
    - property_community_presence (graph-level)
    - property_edge_prob_matrix (graph-level)
    - y (node-level community labels for community_detection)
    
    Parameters
    ----------
    batch : torch_geometric.data.Batch
        Batched graphs with pre-computed property attributes.
    
    Returns
    -------
    dict
        Dictionary with target values for each community property:
        {
            'homophily': Tensor of shape (batch_size, 1),
            'community_presence': Tensor of shape (batch_size, K),
            'edge_prob_matrix': Tensor of shape (batch_size, K, K),
            'community_detection': Tensor of shape (num_nodes,) - NODE-LEVEL labels
        }
    """
    # Check if properties are pre-computed
    if not hasattr(batch, 'property_homophily'):
        raise ValueError(
            "Graph properties not found! Please pre-compute properties using "
            "add_properties_to_dataset() from graph_properties.py before training."
        )
    
    properties = {}
    
    # Extract homophily
    homophily = batch.property_homophily
    if homophily.dim() == 1:
        homophily = homophily.unsqueeze(1)
    properties['homophily'] = homophily
    
    # Extract complex community properties
    if hasattr(batch, 'property_community_presence'):
        properties['community_presence'] = batch.property_community_presence
    
    if hasattr(batch, 'property_edge_prob_matrix'):
        properties['edge_prob_matrix'] = batch.property_edge_prob_matrix
    
    # Extract community detection labels (NODE-LEVEL)
    # For community_detection: we use the actual node labels (batch.y)
    # This is a NODE-LEVEL classification task (inductive)
    # Each node has a community label (0 to K-1)
    if hasattr(batch, 'y'):
        # Node-level labels: shape (num_nodes,)
        properties['community_detection'] = batch.y
    
    return properties


class GraphLevelEncoder(nn.Module):
    """
    Wrapper for graph-level tasks: encoder + readout/pooling.
    
    Takes node embeddings and pools them to graph-level representations.
    """
    
    def __init__(
        self,
        feature_encoder: nn.Module,
        backbone: nn.Module,
        readout_type: str = "sum",  # "mean", "max", "sum"
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
        self.readout_type = readout_type
        
        # Select pooling function
        if readout_type == "mean":
            self.pool = global_mean_pool
        elif readout_type == "max":
            self.pool = global_max_pool
        elif readout_type == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown readout type: {readout_type}")
    
    def forward(self, batch, return_node_features=False):
        """
        Get encoded features (graph-level by default, optionally with node-level).
        
        Parameters
        ----------
        batch : torch_geometric.data.Batch
            Batched graphs.
        return_node_features : bool
            If True, returns both node and graph features as a dict.
            If False, returns only graph features (backward compatible).
        
        Returns
        -------
        torch.Tensor or dict
            If return_node_features=False: Graph-level features (batch_size, hidden_dim)
            If return_node_features=True: Dict with 'node' and 'graph' features
        """
        # Prepare batch with TopoBench-expected attributes
        batch = prepare_batch_for_topobench(batch)
        
        # Feature encoding
        batch_encoded = self.feature_encoder(batch)
        
        # Extract the encoded features
        if hasattr(batch_encoded, 'x_0'):
            x = batch_encoded.x_0
        elif isinstance(batch_encoded, dict):
            x = batch_encoded.get('x_0', batch_encoded.get('x'))
        else:
            x = batch_encoded
        
        # Get other attributes from original batch
        edge_index = batch.edge_index
        batch_indices = batch.batch_0 if hasattr(batch, 'batch_0') else batch.batch
        edge_weight = getattr(batch, 'edge_weight', None)
        
        # Backbone encoding - returns node-level features
        node_features = self.backbone(
            x,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Pool to graph-level
        graph_features = self.pool(node_features, batch_indices)
        
        if return_node_features:
            return {
                'node': node_features,  # (num_nodes, hidden_dim)
                'graph': graph_features,  # (batch_size, hidden_dim)
            }
        else:
            return graph_features  # Backward compatible

def save_downstream_model(
    model: nn.Module,
    path: str | Path,
    config: dict = None,
    optimizer_state: dict = None,
    epoch: int = None,
    metrics: dict = None,
):
    """
    Properly save a downstream model with all necessary components.
    
    Saves:
    - Model state_dict (all weights including frozen ones)
    - Model config for reconstruction
    - Optimizer state (optional, for resuming training)
    - Training metadata
    
    Parameters
    ----------
    model : nn.Module
        The model to save.
    path : str or Path
        Path to save the checkpoint.
    config : dict, optional
        Configuration dict for model reconstruction.
    optimizer_state : dict, optional
        Optimizer state dict for resuming training.
    epoch : int, optional
        Current epoch number.
    metrics : dict, optional
        Training/validation metrics.
    """
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_class": model.__class__.__name__,
    }
    
    if config is not None:
        checkpoint["config"] = config
    if optimizer_state is not None:
        checkpoint["optimizer_state"] = optimizer_state
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Add model state info
    if hasattr(model, 'get_model_state_info'):
        checkpoint["model_state_info"] = model.get_model_state_info()
    
    torch.save(checkpoint, path)
    print(f"Saved model checkpoint to {path}")


def load_downstream_model_state(
    model: nn.Module,
    path: str | Path,
    strict: bool = True,
) -> dict:
    """
    Load state dict into a downstream model.
    
    Parameters
    ----------
    model : nn.Module
        The model to load weights into.
    path : str or Path
        Path to the checkpoint.
    strict : bool
        Whether to strictly enforce state_dict key matching.
    
    Returns
    -------
    dict
        The full checkpoint dict (contains metadata, optimizer state, etc.)
    """
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=strict)
    
    # Re-apply frozen state if applicable
    if hasattr(model, 'freeze_encoder') and model.freeze_encoder:
        model._freeze_encoder()
    
    print(f"Loaded model checkpoint from {path}")
    
    return checkpoint


class CommunityPropertyReconstructionModel(nn.Module):
    """
    Model for community-related property reconstruction task.
    
    Encoder (frozen/unfrozen) + Community property regressor.
    Predicts: homophily, community_presence, edge_prob_matrix, community_detection.
    
    NOTE: This handles MIXED task levels:
    - homophily, community_presence, edge_prob_matrix: use graph-level features
    - community_detection: uses node-level features
    """
    
    def __init__(
        self,
        encoder: nn.Module,  # GraphLevelEncoder (with pooling)
        property_regressor: CommunityPropertyRegressor,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.property_regressor = property_regressor
        self.freeze_encoder = freeze_encoder
        self.task_level = "mixed"  # Mixed: some graph-level, some node-level
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder: disable gradients and set to eval mode."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Freeze BatchNorm running stats
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
    def forward(self, batch):
        """
        Forward pass: encode graphs and predict community-related properties.
        
        This returns both graph-level and node-level predictions:
        - Graph-level: homophily, community_presence, edge_prob_matrix
        - Node-level: community_detection
        
        Returns
        -------
        dict
            Predicted property values (same format as CommunityPropertyRegressor output).
        """
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(batch, return_node_features=True)
        else:
            features = self.encoder(batch, return_node_features=True)
        
        # features is a dict with 'node' and 'graph' keys
        # Predict properties using both node and graph features
        return self.property_regressor(
            x_graph=features['graph'],  # For graph-level properties
            x_node=features['node'],    # For node-level community_detection
        )
    
    def train(self, mode=True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self
    
    def get_model_state_info(self) -> dict:
        """Get diagnostic info about model state."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        regressor_params = sum(p.numel() for p in self.property_regressor.parameters())
        regressor_trainable = sum(p.numel() for p in self.property_regressor.parameters() if p.requires_grad)
        
        return {
            "encoder_params": encoder_params,
            "encoder_trainable": encoder_trainable,
            "regressor_params": regressor_params,
            "regressor_trainable": regressor_trainable,
            "freeze_encoder": self.freeze_encoder,
            "encoder_mode": "eval" if not self.encoder.training else "train",
            "encoder_in_train_mode": self.encoder.training,
        }


class PropertyReconstructionModel(nn.Module):
    """
    Model for multi-property reconstruction task.
    
    Encoder (frozen/unfrozen) + Multi-head property regressor.
    """
    
    def __init__(
        self,
        encoder: nn.Module,  # GraphLevelEncoder (with pooling)
        property_regressor: MultiPropertyRegressor,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.property_regressor = property_regressor
        self.freeze_encoder = freeze_encoder
        self.task_level = "graph"  # Always graph-level
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder: disable gradients and set to eval mode."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        # Freeze BatchNorm running stats
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
    def forward(self, batch):
        """
        Forward pass: encode graph and predict properties.
        
        Returns
        -------
        dict
            Predicted property values (same format as MultiPropertyRegressor output).
        """
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                graph_features = self.encoder(batch)
        else:
            graph_features = self.encoder(batch)
        
        # Predict properties
        return self.property_regressor(graph_features)
    
    def train(self, mode=True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self
    
    def get_model_state_info(self) -> dict:
        """Get diagnostic info about model state."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        regressor_params = sum(p.numel() for p in self.property_regressor.parameters())
        regressor_trainable = sum(p.numel() for p in self.property_regressor.parameters() if p.requires_grad)
        
        encoder_training = any(m.training for m in self.encoder.modules())
        regressor_training = any(m.training for m in self.property_regressor.modules())
        
        return {
            "encoder_params": encoder_params,
            "encoder_trainable": encoder_trainable,
            "encoder_frozen": self.freeze_encoder,
            "encoder_in_train_mode": encoder_training,
            "regressor_params": regressor_params,
            "regressor_trainable": regressor_trainable,
            "regressor_in_train_mode": regressor_training,
        }


class DownstreamModel(nn.Module):
    """
    Complete downstream model: encoder + classifier.
    
    Supports both node-level and graph-level tasks.
    """
    
    def __init__(
        self,
        encoder: nn.Module,  # PretrainedEncoder or GraphLevelEncoder
        classifier: nn.Module,
        freeze_encoder: bool = True,
        task_level: str = "node",  # "node" or "graph"
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder
        self.task_level = task_level
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder: disable gradients and set to eval mode."""
        # Disable gradients for all encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Set to eval mode (disables dropout, freezes BatchNorm running stats)
        self.encoder.eval()
        
        # Additionally, explicitly freeze BatchNorm running stats
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False  # Extra safety
    
    def forward(self, batch):
        """
        Forward pass.
        
        Returns
        -------
        torch.Tensor
            - For node-level: (num_nodes, num_classes)
            - For graph-level: (batch_size, num_classes)
        """
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(batch)
        else:
            features = self.encoder(batch)
        
        return self.classifier(features)
    
    def train(self, mode=True):
        """Override train to keep frozen encoder in eval mode."""
        super().train(mode)
        if self.freeze_encoder:
            # Always keep encoder in eval mode when frozen
            self.encoder.eval()
        return self
    
    def get_model_state_info(self) -> dict:
        """Get diagnostic info about model state."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        # Check module modes
        encoder_training = any(m.training for m in self.encoder.modules())
        classifier_training = any(m.training for m in self.classifier.modules())
        
        return {
            "encoder_params": encoder_params,
            "encoder_trainable": encoder_trainable,
            "encoder_frozen": self.freeze_encoder,
            "encoder_in_train_mode": encoder_training,
            "classifier_params": classifier_params,
            "classifier_trainable": classifier_trainable,
            "classifier_in_train_mode": classifier_training,
        }
# =============================================================================
# Training & Evaluation
# =============================================================================

def create_data_splits(
    data_list: list,
    n_train: int,
    dataset_info: dict = None,
    val_ratio: float = 0.5,
    seed: int = 42,
):
    """
    Split data into train/val/test sets.
    
    For GraphUniverse: uses n_train as absolute number of training graphs
    For TUDataset: 
        - If subsample_train=False: uses original train_prop split, ignores n_train
        - If subsample_train=True: uses original train_prop split, then subsamples n_train from training set
    
    Parameters
    ----------
    data_list : list
        List of Data objects.
    n_train : int
        Number of training samples (interpretation depends on dataset type).
    dataset_info : dict
        Metadata about dataset type and subsampling preferences
    val_ratio : float
        Ratio of remaining data for validation (rest goes to test).
    seed : int
        Random seed.
    
    Returns
    -------
    tuple
        (train_data, val_data, test_data)
    """
    import random
    random.seed(seed)
    
    # For GraphUniverse: use the original logic (n_train absolute graphs)
    if dataset_info is None or dataset_info.get("type") == "GraphUniverse":
        indices = list(range(len(data_list)))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        remaining = indices[n_train:]
        
        n_val = int(len(remaining) * val_ratio)
        val_indices = remaining[:n_val]
        test_indices = remaining[n_val:]
        
    else:  # TUDataset
        # Use the original train_prop from config to create base splits
        # Then optionally subsample from training set
        
        total_size = len(data_list)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        # Assume train_prop = 0.5 (could extract from config if needed)
        train_prop = 0.5
        base_train_size = int(total_size * train_prop)
        
        base_train_indices = indices[:base_train_size]
        remaining = indices[base_train_size:]
        
        # If subsampling requested and n_train specified
        if dataset_info.get("subsample_train") and dataset_info.get("n_train_requested"):
            n_train_actual = min(dataset_info["n_train_requested"], len(base_train_indices))
            train_indices = base_train_indices[:n_train_actual]
            print(f"Subsampled training set: {len(base_train_indices)} -> {n_train_actual}")
        else:
            train_indices = base_train_indices
            print(f"Using full training set: {len(train_indices)} samples")
        
        # Split remaining into val/test
        n_val = int(len(remaining) * val_ratio)
        val_indices = remaining[:n_val]
        test_indices = remaining[n_val:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    print(f"Data split: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    return train_data, val_data, test_data

def _train_single_property(
    encoder: nn.Module,
    property_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,  # Add test loader
    hidden_dim: int,
    freeze_encoder: bool,
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_wandb: bool,
    K: int = 10,  # Number of communities (for complex properties)
    baseline_val_mae: float | None = None,  # For comparison
    baseline_test_mae: float | None = None,  # For comparison
    train_mean: float | None = None,  # For baseline reporting
) -> tuple:
    """
    Train a single property prediction head and evaluate on test set.
    
    Returns
    -------
    tuple
        (best_val_loss, test_mae, history_dict, model)
    """
    # Property-specific configurations
    # Key insight: Use appropriate loss for each property type
    # - Simple properties (scalars): Use MSE loss AFTER activation
    # - Complex properties (vectors/matrices): Use appropriate loss for their structure
    # - Classification (community_detection): Use CrossEntropyLoss on logits, measure accuracy
    property_configs = {
        'homophily': {
            'loss_fn': nn.MSELoss(),  # MSE on [0,1] scale
            'activation': 'sigmoid',  # Maps to [0,1]
            'description': 'Bounded [0,1] regression with sigmoid',
            'output_dim': 1,
            'metric': 'mae',  # Use MAE as metric
        },
        'avg_degree': {
            'loss_fn': nn.MSELoss(),  # MSE on positive scale
            'activation': 'softplus',  # Maps to [0,inf), smooth at 0
            'description': 'Continuous positive [2-10], softplus ensures >0',
            'output_dim': 1,
            'metric': 'mae',  # Use MAE as metric
        },
        'size': {
            'loss_fn': nn.MSELoss(),  # MSE on positive scale
            'activation': 'softplus',  # Maps to [0,inf), naturally handles large values
            'description': 'Integer large range [50-1000+], softplus ensures >0',
            'output_dim': 1,
            'metric': 'mae',  # Use MAE as metric
        },
        'gini': {
            'loss_fn': nn.MSELoss(),  # MSE on [0,1] scale
            'activation': 'sigmoid',  # Maps to [0,1]
            'description': 'Bounded [0,1], often < 0.5, sigmoid mapping',
            'output_dim': 1,
            'metric': 'mae',  # Use MAE as metric
        },
        'diameter': {
            'loss_fn': nn.MSELoss(),  # MSE on positive scale
            'activation': 'softplus',  # Maps to [0,inf), smooth growth
            'description': 'Integer moderate [1-100+], softplus ensures >0',
            'output_dim': 1,
            'metric': 'mae',  # Use MAE as metric
        },
        'community_presence': {
            'loss_fn': nn.BCEWithLogitsLoss(),  # More stable: combines sigmoid + BCE
            'activation': 'sigmoid_for_pred_only',  # Logits for loss, sigmoid for metrics
            'description': 'K-dimensional binary vector, BCEWithLogitsLoss (more stable)',
            'output_dim': 'K',  # Will be replaced with actual K value
            'metric': 'mae',  # Use MAE as metric
        },
        'edge_prob_matrix': {
            'loss_fn': 'custom_matrix_loss',  # Custom loss: MSE + Frobenius norm penalty
            'activation': 'sigmoid',  # Maps to [0,1] for each edge probability
            'description': 'K×K symmetric matrix [0,1], Custom loss (MSE + structure penalty)',
            'output_dim': 'K*K',  # Will be replaced with actual K*K value
            'metric': 'mae',  # Use MAE as metric
        },
        'community_detection': {
            'loss_fn': nn.CrossEntropyLoss(),  # Classification loss
            'activation': 'none',  # No activation, use logits for CrossEntropyLoss
            'description': 'K-class node-level classification (inductive)',
            'output_dim': 'K',  # Will be replaced with actual K value
            'metric': 'accuracy',  # Use accuracy as metric instead of MAE
            'task_level': 'node',  # NODE-LEVEL task (different from others!)
        },
    }
    
    config = property_configs[property_name]
    loss_fn_spec = config['loss_fn']
    activation = config['activation']
    metric_type = config['metric']  # 'mae' or 'accuracy'
    
    # Determine output dimension
    output_dim = config['output_dim']
    if output_dim == 'K':
        output_dim = K
    elif output_dim == 'K*K':
        output_dim = K * K
    
    # Create custom loss function for edge_prob_matrix
    if loss_fn_spec == 'custom_matrix_loss':
        # Custom loss that penalizes both element-wise error AND structural inconsistency
        class EdgeProbMatrixLoss(nn.Module):
            def __init__(self, mse_weight=1.0, frobenius_weight=0.5):
                super().__init__()
                self.mse_weight = mse_weight
                self.frobenius_weight = frobenius_weight
                self.mse = nn.MSELoss()
            
            def forward(self, pred, target):
                # Standard MSE loss (element-wise)
                mse_loss = self.mse(pred, target)
                
                # Frobenius norm of difference (penalizes overall matrix structure)
                # This makes it harder to just predict near-mean values
                diff = pred - target
                frobenius_loss = torch.norm(diff.view(diff.size(0), -1), p='fro', dim=1).mean()
                
                return self.mse_weight * mse_loss + self.frobenius_weight * frobenius_loss
        
        criterion = EdgeProbMatrixLoss(mse_weight=1.0, frobenius_weight=0.3)
        loss_name = "Custom (MSE + 0.3×Frobenius)"
    else:
        criterion = loss_fn_spec
        loss_name = criterion.__class__.__name__
    
    print(f"\n{'='*80}")
    print(f"TRAINING PROPERTY: {property_name.upper()}")
    print(f"{'='*80}")
    print(f"  Loss function: {loss_name}")
    print(f"  Output activation: {activation}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Description: {config['description']}")
    print(f"  Metric: {metric_type.upper()}")
    print(f"  Encoder frozen: {freeze_encoder}")
    if baseline_val_mae is not None and metric_type == 'mae':
        print(f"  Baseline val MAE: {baseline_val_mae:.4f} (must beat this!)")
    if property_name in ['community_presence', 'edge_prob_matrix']:
        print(f"  Complex property: output shape will be reshaped in forward pass")
    if property_name == 'community_detection':
        print(f"  NODE-LEVEL classification task: using accuracy metric (not MAE)")
        print(f"  Predicting community labels for each node (inductive)")
        print(f"  No baseline comparison (accuracy starts from 0)")
    elif activation != 'none':
        print(f"  KEY: Loss computed AFTER activation (not on logits!)")
    print(f"{'='*80}\n")
    
    # Create prediction head (Linear: hidden_dim -> output_dim)
    # Output is then passed through activation before computing loss
    prediction_head = nn.Linear(hidden_dim, output_dim).to(device)
    
    # CRITICAL: Initialize with small weights to avoid overwhelming pretrained features
    nn.init.xavier_uniform_(prediction_head.weight, gain=0.01)
    nn.init.zeros_(prediction_head.bias)
    
    # Determine if this is a node-level task
    is_node_level = config.get('task_level', 'graph') == 'node'
    
    # Create model combining encoder + head
    class SinglePropertyModel(nn.Module):
        def __init__(self, encoder, head, freeze_encoder, activation, property_name, K=10, is_node_level=False):
            super().__init__()
            self.encoder = encoder
            self.head = head
            self.freeze_encoder = freeze_encoder
            self.activation = activation
            self.property_name = property_name
            self.K = K
            self.is_node_level = is_node_level
            
            if freeze_encoder:
                for param in encoder.parameters():
                    param.requires_grad = False
                encoder.eval()
        
        def forward(self, batch, apply_activation=True):
            """
            Forward pass with optional activation.
            
            Parameters
            ----------
            apply_activation : bool
                If True, applies activation to output.
                For 'sigmoid_for_pred_only': only applies sigmoid when True (for metrics),
                returns logits when False (for loss computation with BCEWithLogitsLoss).
            """
            if self.freeze_encoder:
                self.encoder.eval()
                with torch.no_grad():
                    if self.is_node_level:
                        # For node-level tasks, get node embeddings
                        features_dict = self.encoder(batch, return_node_features=True)
                        features = features_dict['node']  # (num_nodes, hidden_dim)
                    else:
                        # For graph-level tasks, get graph embeddings
                        features = self.encoder(batch)  # (batch_size, hidden_dim)
            else:
                if self.is_node_level:
                    # For node-level tasks, get node embeddings
                    features_dict = self.encoder(batch, return_node_features=True)
                    features = features_dict['node']  # (num_nodes, hidden_dim)
                else:
                    # For graph-level tasks, get graph embeddings
                    features = self.encoder(batch)  # (batch_size, hidden_dim)
            
            output = self.head(features)
            
            # Apply activation based on property type
            if apply_activation:
                if self.activation == 'sigmoid':
                    output = torch.sigmoid(output)
                elif self.activation == 'sigmoid_for_pred_only':
                    # Apply sigmoid for predictions/metrics
                    output = torch.sigmoid(output)
                elif self.activation == 'softplus':
                    output = torch.nn.functional.softplus(output)
                elif self.activation == 'none':
                    # No activation - return logits (for community_detection with CrossEntropyLoss)
                    pass
            # else: return logits (for BCEWithLogitsLoss or classification)
            
            # Reshape for complex properties (only for graph-level)
            if not self.is_node_level:
                if self.property_name == 'community_presence':
                    # Output is already [batch_size, K]
                    pass
                elif self.property_name == 'edge_prob_matrix':
                    # Reshape from [batch_size, K*K] to [batch_size, K, K]
                    batch_size = output.size(0)
                    output = output.view(batch_size, self.K, self.K)
                    # Make symmetric
                    output = (output + output.transpose(1, 2)) / 2
            # For node-level: output is [num_nodes, K] - logits for classification
            
            return output
        
        def train(self, mode=True):
            super().train(mode)
            if self.freeze_encoder:
                self.encoder.eval()
            return self
    
    model = SinglePropertyModel(encoder, prediction_head, freeze_encoder, activation, property_name, K, is_node_level)
    model = model.to(device)
    
    # Optimizer: only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_val_metric = 0.0 if metric_type == 'accuracy' else float('inf')  # Higher is better for accuracy, lower for MAE
    best_model_state = None
    patience_counter = 0
    
    # History tracking depends on metric type
    metric_name = metric_type  # 'mae' or 'accuracy'
    history = {
        "train_loss": [],
        f"train_{metric_name}": [],
        "val_loss": [],
        f"val_{metric_name}": [],
    }
    
    pbar = tqdm(range(epochs), desc=f"Training {property_name}")
    
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_metric = 0.0  # MAE or accuracy depending on property
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()
            
            # Get target - for node-level community_detection, use batch.y directly
            if property_name == 'community_detection':
                # NODE-LEVEL: target is batch.y with shape (num_nodes,)
                target = batch.y.to(device)
            else:
                # GRAPH-LEVEL: target is pre-computed property
                target = getattr(batch, f'property_{property_name}').to(device)
            
            # Handle different property types
            if property_name == 'community_detection':
                # NODE-LEVEL classification task: use CrossEntropyLoss on logits
                logits = model(batch, apply_activation=True)  # Returns logits (no activation)
                
                # Target is (num_nodes,) with class labels for each node
                # Logits is (num_nodes, K) with K logits per node
                # Loss on logits
                loss = criterion(logits, target.long())
                
                # Compute accuracy (across all nodes in batch)
                with torch.no_grad():
                    pred_classes = logits.argmax(dim=1)
                    metric = (pred_classes == target.long()).float().mean()
                
                # For batch counting, use number of nodes (not number of graphs)
                batch_size = target.size(0)  # num_nodes
                
            elif property_name == 'community_presence':
                # PyG batching flattens [K] tensors to [batch_size * K]
                # Need to reshape to [batch_size, K]
                # For BCEWithLogitsLoss, we need logits (no activation for loss)
                logits = model(batch, apply_activation=False)
                batch_size = logits.size(0)
                K = logits.size(1)
                target = target.view(batch_size, K)
                
                # Loss on logits (BCEWithLogitsLoss)
                loss = criterion(logits, target)
                
                # For MAE, we need probabilities
                with torch.no_grad():
                    pred_probs = torch.sigmoid(logits)
                    metric = torch.abs(pred_probs - target).mean()
            else:
                # Other properties: get prediction WITH activation
                pred = model(batch, apply_activation=True)
                
                if property_name == 'edge_prob_matrix':
                    # PyG batching flattens [K, K] tensors to [batch_size * K * K]
                    batch_size = pred.size(0)
                    K = pred.size(1)
                    target = target.view(batch_size, K, K)
                elif target.dim() == 1:
                    # Simple scalar properties
                    target = target.unsqueeze(1)
                
                # Loss on ACTUAL scale (after activation)
                loss = criterion(pred, target)
                
                # MAE on actual scale
                metric = torch.abs(pred - target).mean()
                
                batch_size = target.size(0)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_total += batch_size
            train_loss += loss.item() * batch_size
            train_metric += metric.item() * batch_size
        
        train_loss /= train_total
        train_metric /= train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metric = 0.0  # MAE or accuracy depending on property
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                
                # Get target - for node-level community_detection, use batch.y directly
                if property_name == 'community_detection':
                    # NODE-LEVEL: target is batch.y with shape (num_nodes,)
                    target = batch.y.to(device)
                else:
                    # GRAPH-LEVEL: target is pre-computed property
                    target = getattr(batch, f'property_{property_name}').to(device)
                
                # Handle different property types
                if property_name == 'community_detection':
                    # NODE-LEVEL classification task
                    logits = model(batch, apply_activation=True)  # Returns logits
                    
                    # Loss on logits
                    loss = criterion(logits, target.long())
                    
                    # Compute accuracy (across all nodes in batch)
                    pred_classes = logits.argmax(dim=1)
                    metric = (pred_classes == target.long()).float().mean()
                    
                    # For batch counting, use number of nodes (not number of graphs)
                    batch_size = target.size(0)  # num_nodes
                    
                elif property_name == 'community_presence':
                    # For BCEWithLogitsLoss, we need logits
                    logits = model(batch, apply_activation=False)
                    batch_size = logits.size(0)
                    K = logits.size(1)
                    target = target.view(batch_size, K)
                    
                    # Loss on logits
                    loss = criterion(logits, target)
                    
                    # For MAE, we need probabilities
                    pred_probs = torch.sigmoid(logits)
                    metric = torch.abs(pred_probs - target).mean()
                else:
                    # Forward - get prediction WITH activation
                    pred = model(batch, apply_activation=True)
                    
                    if property_name == 'edge_prob_matrix':
                        # PyG batching flattens [K, K] tensors to [batch_size * K * K]
                        batch_size = pred.size(0)
                        K = pred.size(1)
                        target = target.view(batch_size, K, K)
                    elif target.dim() == 1:
                        # Simple scalar properties
                        target = target.unsqueeze(1)
                    
                    # Loss on actual scale
                    loss = criterion(pred, target)
                    
                    # MAE on actual scale
                    metric = torch.abs(pred - target).mean()
                    
                    batch_size = pred.size(0)
                
                val_total += batch_size
                val_loss += loss.item() * batch_size
                val_metric += metric.item() * batch_size
        
        val_loss /= val_total
        val_metric /= val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history[f"train_{metric_name}"].append(train_metric)
        history["val_loss"].append(val_loss)
        history[f"val_{metric_name}"].append(val_metric)
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                f"{property_name}/epoch": epoch,
                f"{property_name}/train_loss": train_loss,
                f"{property_name}/train_{metric_name}": train_metric,
                f"{property_name}/val_loss": val_loss,
                f"{property_name}/val_{metric_name}": val_metric,
                f"{property_name}/lr": optimizer.param_groups[0]['lr'],
            })
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Early stopping based on metric (for accuracy, higher is better; for MAE, lower is better)
        if metric_type == 'accuracy':
            is_better = val_metric > best_val_metric
        else:  # mae
            is_better = val_metric < best_val_metric
        
        if is_better:
            best_val_loss = val_loss
            best_val_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            f"val_{metric_name}": f"{val_metric:.4f}",
            f"best_{metric_name}": f"{best_val_metric:.4f}",
        })
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        # Move state dict back to device before loading
        best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state_device)
    
    # Ensure model is in eval mode
    model.eval()
    
    # IMMEDIATELY evaluate on test set to avoid any state issues
    test_metric = 0.0  # MAE or accuracy
    test_total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            
            # Get target - for node-level community_detection, use batch.y directly
            if property_name == 'community_detection':
                # NODE-LEVEL: target is batch.y with shape (num_nodes,)
                target = batch.y.to(device)
            else:
                # GRAPH-LEVEL: target is pre-computed property
                target = getattr(batch, f'property_{property_name}').to(device)
            
            # Handle different property types
            if property_name == 'community_detection':
                # NODE-LEVEL classification: compute accuracy
                logits = model(batch, apply_activation=True)  # Returns logits
                pred_classes = logits.argmax(dim=1)
                metric = (pred_classes == target.long()).float().mean()
                # For batch counting, use number of nodes (not number of graphs)
                batch_size = target.size(0)  # num_nodes
                
            elif property_name == 'community_presence':
                # For community_presence, get probabilities with sigmoid
                pred = model(batch, apply_activation=True)
                batch_size = pred.size(0)
                K = pred.size(1)
                target = target.view(batch_size, K)
                metric = torch.abs(pred - target).mean()
            else:
                # Forward - get prediction WITH activation
                pred = model(batch, apply_activation=True)
                
                if property_name == 'edge_prob_matrix':
                    # PyG batching flattens [K, K] tensors to [batch_size * K * K]
                    batch_size = pred.size(0)
                    K = pred.size(1)
                    target = target.view(batch_size, K, K)
                elif target.dim() == 1:
                    # Simple scalar properties
                    target = target.unsqueeze(1)
                
                batch_size = pred.size(0)
                
                # MAE on actual scale
                metric = torch.abs(pred - target).mean()
            
            test_total += batch_size
            test_metric += metric.item() * batch_size
    
    test_metric /= test_total
    
    # For backwards compatibility, keep test_mae variable (will be accuracy for community_detection)
    test_mae = test_metric
    
    # Report results
    print(f"\n{'='*80}")
    print(f"RESULTS FOR {property_name.upper()}")
    print(f"{'='*80}")
    
    if metric_type == 'accuracy':
        # Classification task: report accuracy
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val accuracy: {best_val_metric:.4f}")
        print(f"  Test accuracy: {test_metric:.4f}")
        print(f"  Note: No baseline for classification (accuracy starts from 0)")
    else:
        # Regression task: report MAE with baseline comparison
        # For simple properties, print train_mean
        if train_mean is not None and isinstance(train_mean, (int, float)):
            print(f"  Train mean: {train_mean:.4f}")
        if baseline_val_mae is not None:
            print(f"  Baseline val MAE: {baseline_val_mae:.4f}")
        if baseline_test_mae is not None:
            print(f"  Baseline test MAE: {baseline_test_mae:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val MAE: {best_val_metric:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        if baseline_test_mae is not None:
            beat_baseline = "✓" if test_mae < baseline_test_mae else "✗"
            improvement = ((baseline_test_mae - test_mae) / baseline_test_mae * 100) if baseline_test_mae > 0 else 0
            print(f"  vs Baseline: {improvement:+.1f}% {beat_baseline}")
    print(f"{'='*80}\n")
    
    return best_val_loss, test_mae, history, model


def _train_property_reconstruction_consecutive(
    model: PropertyReconstructionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,  # Add test loader
    device: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    use_wandb: bool,
    K: int,
    normalization_scales: dict,
    properties_to_include: list[str] | None = None,  # Which properties to train (None = all)
) -> dict:
    """
    Train each property CONSECUTIVELY with individual prediction heads.
    
    Each property starts from the SAME pretrained encoder state.
    """
    print("\n" + "="*80)
    print("CONSECUTIVE PROPERTY TRAINING (NEW DEFAULT)")
    print("="*80)
    print("Each property will be trained separately with its own prediction head.")
    print("Each property starts from the SAME pretrained encoder state.")
    print("="*80)
    
    # Detect if this is a community-only model
    is_community_model = isinstance(model, CommunityPropertyReconstructionModel)
    
    if is_community_model:
        # Community-related properties (3 regression + 1 classification)
        simple_properties = ['homophily']
        complex_properties = ['community_presence', 'edge_prob_matrix']
        classification_properties = ['community_detection']
        all_available_properties = simple_properties + complex_properties + classification_properties
        
        # Filter properties if specified
        if properties_to_include is not None:
            properties = [p for p in all_available_properties if p in properties_to_include]
            print(f"  Model type: CommunityPropertyReconstructionModel ({len(properties)} properties selected)")
            print(f"  Selected properties: {', '.join(properties)}")
        else:
            properties = all_available_properties
            print("  Model type: CommunityPropertyReconstructionModel (4 properties)")
            print("    - 3 regression: homophily, community_presence, edge_prob_matrix")
            print("    - 1 classification: community_detection (accuracy metric)")
    else:
        # Basic properties only (NO complex properties, NO homophily - moved to community)
        simple_properties = ['avg_degree', 'size', 'gini', 'diameter']
        complex_properties = []  # No complex properties for basic reconstruction
        classification_properties = []  # No classification for basic reconstruction
        all_available_properties = simple_properties
        
        # Filter properties if specified
        if properties_to_include is not None:
            properties = [p for p in all_available_properties if p in properties_to_include]
            print(f"  Model type: PropertyReconstructionModel ({len(properties)} properties selected)")
            print(f"  Selected properties: {', '.join(properties)}")
        else:
            properties = all_available_properties
            print("  Model type: PropertyReconstructionModel (4 properties)")
    
    # First, compute baselines using TRAIN mean to predict VAL
    print("\n" + "=" * 80)
    print("COMPUTING BASELINES (Predict TRAIN Mean)")
    print("=" * 80)
    
    train_means_dict = {prop: [] for prop in properties}
    val_targets_dict = {prop: [] for prop in properties}
    test_targets_dict = {prop: [] for prop in properties}
    
    # Collect train, val, and test data
    for batch in train_loader:
        for prop in properties:
            # For community_detection, extract from batch.y (mode of node labels)
            if prop == 'community_detection':
                batch_size = batch.num_graphs
                community_labels = []
                for graph_idx in range(batch_size):
                    node_mask = (batch.batch == graph_idx)
                    graph_node_labels = batch.y[node_mask]
                    mode_label = torch.mode(graph_node_labels).values
                    community_labels.append(mode_label)
                target = torch.stack(community_labels)
            else:
                target = getattr(batch, f'property_{prop}')
            
            if target.dim() == 1:
                target = target.unsqueeze(1)
            train_means_dict[prop].append(target)
    
    for batch in val_loader:
        for prop in properties:
            # For community_detection, extract from batch.y (mode of node labels)
            if prop == 'community_detection':
                batch_size = batch.num_graphs
                community_labels = []
                for graph_idx in range(batch_size):
                    node_mask = (batch.batch == graph_idx)
                    graph_node_labels = batch.y[node_mask]
                    mode_label = torch.mode(graph_node_labels).values
                    community_labels.append(mode_label)
                target = torch.stack(community_labels)
            else:
                target = getattr(batch, f'property_{prop}')
            
            if target.dim() == 1:
                target = target.unsqueeze(1)
            val_targets_dict[prop].append(target)
    
    for batch in test_loader:
        for prop in properties:
            # For community_detection, extract from batch.y (mode of node labels)
            if prop == 'community_detection':
                batch_size = batch.num_graphs
                community_labels = []
                for graph_idx in range(batch_size):
                    node_mask = (batch.batch == graph_idx)
                    graph_node_labels = batch.y[node_mask]
                    mode_label = torch.mode(graph_node_labels).values
                    community_labels.append(mode_label)
                target = torch.stack(community_labels)
            else:
                target = getattr(batch, f'property_{prop}')
            
            if target.dim() == 1:
                target = target.unsqueeze(1)
            test_targets_dict[prop].append(target)
    
    # Compute baselines (skip classification properties)
    train_mean_values = {}
    baseline_val_mae = {}
    baseline_test_mae = {}
    
    # Get classification properties list (if it exists)
    classification_props = classification_properties if is_community_model else []
    
    for prop in properties:
        # Skip classification properties (no MAE baseline)
        if prop in classification_props:
            print(f"{prop:<25}: CLASSIFICATION TASK - no MAE baseline (use accuracy from 0)")
            train_mean_values[prop] = None
            baseline_val_mae[prop] = None
            baseline_test_mae[prop] = None
            continue
        
        train_all = torch.cat(train_means_dict[prop], dim=0)
        val_all = torch.cat(val_targets_dict[prop], dim=0)
        test_all = torch.cat(test_targets_dict[prop], dim=0)
        
        # For simple properties (scalars), compute mean as usual
        # For complex properties (vectors/matrices), compute element-wise mean
        if prop in simple_properties:
            train_mean = train_all.mean().item()
            train_std = train_all.std().item()
            train_mean_values[prop] = train_mean
            
            # Baseline: predict train mean for validation and test
            baseline_val_mae[prop] = torch.abs(val_all - train_mean).mean().item()
            baseline_test_mae[prop] = torch.abs(test_all - train_mean).mean().item()
            
            print(f"{prop:<25}: train_mean={train_mean:>8.4f}, train_std={train_std:>8.4f}, baseline_val_mae={baseline_val_mae[prop]:>8.4f}, baseline_test_mae={baseline_test_mae[prop]:>8.4f}")
        else:
            # For complex properties, compute element-wise statistics
            # train_mean will be a tensor (not a scalar)
            train_mean_tensor = train_all.mean(dim=0)  # Average across batch
            train_std = train_all.std().item()  # Overall std for reference
            train_mean_values[prop] = train_mean_tensor  # Store as tensor
            
            # Baseline: predict train mean (broadcasted) for validation and test
            baseline_val_mae[prop] = torch.abs(val_all - train_mean_tensor).mean().item()
            baseline_test_mae[prop] = torch.abs(test_all - train_mean_tensor).mean().item()
            
            print(f"{prop:<25}: train_std={train_std:>8.4f}, baseline_val_mae={baseline_val_mae[prop]:>8.4f}, baseline_test_mae={baseline_test_mae[prop]:>8.4f} [shape: {list(train_mean_tensor.shape)}]")
    
    print("=" * 80)
    
    # Log baselines to wandb (skip classification properties)
    if use_wandb and WANDB_AVAILABLE:
        for prop in properties:
            if prop not in classification_props:  # Only log for regression properties
                wandb.log({
                    f"{prop}/baseline_train_mean": train_mean_values[prop],
                    f"{prop}/baseline_val_mae": baseline_val_mae[prop],
                })
    
    # Save initial encoder state
    encoder = model.encoder
    freeze_encoder = model.freeze_encoder
    hidden_dim = None
    
    # Detect hidden dim from encoder
    sample_batch = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    sample_batch = prepare_batch_for_topobench(sample_batch)
    
    if freeze_encoder:
        encoder.eval()
        with torch.no_grad():
            sample_out = encoder(sample_batch)
    else:
        encoder.eval()
        with torch.no_grad():
            sample_out = encoder(sample_batch)
        encoder.train()
    
    hidden_dim = sample_out.shape[-1]
    print(f"\nDetected hidden dimension: {hidden_dim}")
    
    # Store initial encoder state
    initial_encoder_state = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
    
    # Train each property
    all_results = {}
    all_test_maes = {}
    all_histories = {}
    trained_models = {}
    
    for prop in properties:
        # CRITICAL: Create a FRESH encoder instance for each property
        # We can't just reset state_dict because all models would share the same encoder object
        # and when one property fine-tunes, it would affect previously trained models!
        
        # Create a fresh encoder by loading initial state into a new instance
        import copy
        encoder_for_this_property = copy.deepcopy(encoder)
        encoder_for_this_property.load_state_dict(initial_encoder_state)
        encoder_for_this_property = encoder_for_this_property.to(device)
        
        print(f"\n{'='*80}")
        print(f"Training property: {prop}")
        print(f"  Using FRESH encoder instance (id: {id(encoder_for_this_property)})")
        print(f"{'='*80}")
        
        # Train this property and immediately evaluate on test
        best_val_loss, test_mae, history, trained_model = _train_single_property(
            encoder=encoder_for_this_property,  # Pass the fresh encoder instance
            property_name=prop,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  # Pass test loader
            hidden_dim=hidden_dim,
            freeze_encoder=freeze_encoder,
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            use_wandb=use_wandb,
            K=K,  # Pass K for complex properties
            baseline_val_mae=baseline_val_mae[prop],
            baseline_test_mae=baseline_test_mae[prop],
            train_mean=train_mean_values[prop],
        )
        
        all_results[prop] = best_val_loss
        all_test_maes[prop] = test_mae
        all_histories[prop] = history
        trained_models[prop] = trained_model
        
        print(f"✓ Model for {prop} saved with encoder id: {id(trained_model.encoder)}")
    
    # Combine histories (handle both MAE and accuracy metrics)
    combined_history = {}
    for prop in properties:
        # Check if this property uses accuracy or MAE
        if prop in classification_properties:
            # Classification: use accuracy
            combined_history[f"train_accuracy_{prop}"] = all_histories[prop]["train_accuracy"]
            combined_history[f"val_accuracy_{prop}"] = all_histories[prop]["val_accuracy"]
            # Also store as "mae" key for backward compatibility (though it's actually accuracy)
            combined_history[f"train_mae_{prop}"] = all_histories[prop]["train_accuracy"]
            combined_history[f"val_mae_{prop}"] = all_histories[prop]["val_accuracy"]
        else:
            # Regression: use MAE
            combined_history[f"train_mae_{prop}"] = all_histories[prop]["train_mae"]
            combined_history[f"val_mae_{prop}"] = all_histories[prop]["val_mae"]
        
        combined_history[f"train_loss_{prop}"] = all_histories[prop]["train_loss"]
        combined_history[f"val_loss_{prop}"] = all_histories[prop]["val_loss"]
        combined_history[f"test_mae_{prop}"] = all_test_maes[prop]  # Store test metric for each property
    
    # Store trained models, train means, and test results for later retrieval
    model._consecutive_models = trained_models
    model._train_mean_values = train_mean_values
    model._test_maes = all_test_maes  # Store test MAEs (computed immediately after training)
    
    # VERIFICATION: Re-evaluate stored models to check save/load integrity
    print("\n" + "="*80)
    print("VERIFYING MODEL SAVE/LOAD INTEGRITY")
    print("="*80)
    print("Re-evaluating stored models to ensure state was preserved correctly...")
    
    verification_passed = True
    for prop in properties:
        prop_model = trained_models[prop].to(device)
        prop_model.eval()
        
        # Re-compute test metric (MAE or accuracy)
        recomputed_metric = 0.0
        test_total = 0
        
        is_classification = prop in classification_properties
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                
                # Forward pass
                pred = prop_model(batch, apply_activation=True)
                
                # Get target - for node-level community_detection, use batch.y directly
                if prop == 'community_detection':
                    # NODE-LEVEL: target is batch.y with shape (num_nodes,)
                    target = batch.y.to(device)
                else:
                    # GRAPH-LEVEL: target is pre-computed property
                    target = getattr(batch, f'property_{prop}').to(device)
                
                # Handle different property shapes
                if prop == 'community_detection':
                    # NODE-LEVEL classification: compute accuracy
                    pred_classes = pred.argmax(dim=1)
                    metric = (pred_classes == target.long()).float().mean()
                    # For batch counting, use number of nodes (not number of graphs)
                    batch_size = target.size(0)  # num_nodes
                elif prop == 'community_presence':
                    # PyG batching flattens [K] tensors
                    batch_size = pred.size(0)
                    K = pred.size(1)
                    target = target.view(batch_size, K)
                    metric = torch.abs(pred - target).mean()
                elif prop == 'edge_prob_matrix':
                    # PyG batching flattens [K, K] tensors
                    batch_size = pred.size(0)
                    K = pred.size(1)
                    target = target.view(batch_size, K, K)
                    metric = torch.abs(pred - target).mean()
                elif target.dim() == 1:
                    # Simple scalar properties
                    target = target.unsqueeze(1)
                    batch_size = pred.size(0)
                    metric = torch.abs(pred - target).mean()
                else:
                    batch_size = pred.size(0)
                    metric = torch.abs(pred - target).mean()
                
                test_total += batch_size
                recomputed_metric += metric.item() * batch_size
        
        recomputed_metric /= test_total
        original_metric = all_test_maes[prop]  # Contains accuracy for classification
        
        # Check if they match (within numerical tolerance)
        matches = abs(recomputed_metric - original_metric) < 1e-4
        status = "✓ PASS" if matches else "✗ FAIL"
        
        metric_name = "Accuracy" if is_classification else "MAE"
        print(f"  {prop:<15}: Original {metric_name}={original_metric:.6f}, Recomputed={recomputed_metric:.6f} {status}")
        
        if not matches:
            verification_passed = False
            print(f"    WARNING: Mismatch detected! Difference: {abs(recomputed_metric - original_metric):.6f}")
    
    if verification_passed:
        print(f"\n✓ ALL MODELS VERIFIED: Stored models produce identical results")
    else:
        print(f"\n✗ VERIFICATION FAILED: Some models have state issues!")
        print(f"  This indicates a bug in model saving/loading logic.")
    
    print("="*80 + "\n")
    
    return combined_history


def train_property_reconstruction(
    model: PropertyReconstructionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,  # Add test loader for immediate evaluation
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 20,
    use_wandb: bool = False,
    K: int = 10,  # Number of communities for GraphPropertyComputer
    normalization_scales: dict = None,  # Normalization scales from GraphUniverse config
    enable_joint_training: bool = False,  # NEW: Enable joint training (old behavior)
    properties_to_include: list[str] | None = None,  # Which properties to train (None = all)
):
    """
    Train the property reconstruction model.
    
    NEW DEFAULT BEHAVIOR (enable_joint_training=False):
    - Trains each property CONSECUTIVELY with individual prediction heads
    - Each property starts from the SAME pretrained encoder (frozen or unfrozen)
    - Uses property-specific loss functions on ORIGINAL scale (no normalization needed)
    - Logs each property's training curve to wandb separately
    
    OLD BEHAVIOR (enable_joint_training=True):
    - Trains all properties jointly with a multi-head predictor
    - Uses weighted multi-task loss with MAE (L1) for all properties
    
    CONSECUTIVE TRAINING (DEFAULT, enable_joint_training=False):
    Each property is trained separately with its own prediction head:
    - homophily [0,1]: BCEWithLogitsLoss (binary cross-entropy for bounded regression)
    - avg_degree [2-10]: MSELoss (continuous, typically small values)
    - size [50-1000+]: MSELoss (integer, large range)
    - gini [0,1]: BCEWithLogitsLoss (binary cross-entropy for bounded regression)
    - diameter [1-100+]: MSELoss (integer, moderate range)
    
    Each property starts from the SAME pretrained encoder state, ensuring fair comparison.
    No normalization or loss weighting needed - each loss function is natural for its property.
    
    JOINT TRAINING (OLD, enable_joint_training=True):
    All properties trained together with weighted multi-task loss.
    Uses inverse-scale weighting to balance gradient magnitudes across properties.
    
    Returns
    -------
    dict
        Training history with per-property metrics.
    """
    if not enable_joint_training:
        # NEW: Consecutive training with individual heads
        return _train_property_reconstruction_consecutive(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  # Pass test loader
            device=device,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            use_wandb=use_wandb,
            K=K,
            normalization_scales=normalization_scales,
            properties_to_include=properties_to_include,
        )
    
    # OLD: Joint training (kept as fallback)
    from graph_properties import GraphPropertyComputer
    
    model = model.to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    # NEW STRATEGY: Don't normalize! Use property-specific losses on ORIGINAL scale
    # Each property gets a loss function designed for its natural range
    
    # Property-specific loss functions (NO normalization)
    # Use Huber loss (smooth L1) for robustness - less sensitive to outliers than MSE
    # delta=1.0 means: MAE for errors > 1, MSE for errors < 1
    criteria = {
        'homophily': nn.SmoothL1Loss(beta=0.1),    # [0,1] → small beta for this range
        'avg_degree': nn.SmoothL1Loss(beta=1.0),   # [0,~10] → beta=1.0
        'size': nn.SmoothL1Loss(beta=5.0),         # [0,~150] → larger beta for larger values
        'gini': nn.SmoothL1Loss(beta=0.1),         # [0,1] → small beta
        'diameter': nn.SmoothL1Loss(beta=1.0),     # [0,~10] → beta=1.0
    }
    
    # Loss weights: Balance the natural scales of different properties
    # Strategy: Weight inversely proportional to typical value range
    # This way all losses contribute roughly equally to the gradient
    
    if normalization_scales is None:
        normalization_scales = {
            'homophily': 1.0,
            'avg_degree': 20.0,
            'size': 200.0,
            'gini': 1.0,
            'diameter': 20.0,
        }
    
    # Compute inverse-scale weights for balanced gradients
    # For property with range [0, R], MSE loss ~ R², so weight = 1/R² to normalize
    # But we also boost important properties (homophily, gini)
    
    # Base weights (inverse square of typical values)
    avg_degree_typical = normalization_scales['avg_degree'] / 2  # Typical value ~5-10, not 20
    diameter_typical = normalization_scales['diameter'] / 10      # Typical value ~5-15, not 150
    size_typical = normalization_scales['size'] / 2               # Typical value ~50-100, not 150
    
    loss_weights = {
        'homophily': 50.0,                                         # [0,1] → VERY high weight (most important)
        'gini': 50.0,                                              # [0,1] → VERY high weight (important)
        'avg_degree': 1.0 / (avg_degree_typical ** 2),            # ~0.04
        'diameter': 1.0 / (diameter_typical ** 2),                # ~0.0044
        'size': 1.0 / (size_typical ** 2),                        # ~0.0004
    }
    
    print(f"\n  Property value ranges (from GraphUniverse config):")
    for prop, scale in normalization_scales.items():
        print(f"    {prop:15s}: [0, ~{scale:.0f}]")
    
    print(f"\n  Loss weights (inverse-scale for balanced gradients):")
    for prop, weight in loss_weights.items():
        print(f"    {prop:15s}: {weight:.6f}")
    
    print(f"\n  → All properties contribute equally to gradient magnitude")
    print(f"  → NO normalization - learning on original scale!")
    
    # Note: Properties should be pre-computed and stored in the dataset
    # We don't need GraphPropertyComputer during training anymore
    
    # Compute baseline: "predict-the-mean" performance
    # CRITICAL: Use TRAIN mean to predict VAL/TEST (realistic baseline)
    print("\n" + "=" * 80)
    print("BASELINE: Predict-the-Mean Performance")
    print("=" * 80)
    print("Computing average property values from TRAIN set...")
    print("Then using TRAIN mean to predict VAL (realistic baseline)")
    
    # Collect all target values
    train_means = {prop: [] for prop in criteria.keys()}
    val_means = {prop: [] for prop in criteria.keys()}
    
    for batch in train_loader:
        targets = extract_property_targets_from_batch(batch)
        for prop in criteria.keys():
            train_means[prop].append(targets[prop])
    
    for batch in val_loader:
        targets = extract_property_targets_from_batch(batch)
        for prop in criteria.keys():
            val_means[prop].append(targets[prop])
    
    # Compute mean values and baselines
    train_mean_values = {}
    baseline_train_mae = {}
    baseline_val_mae = {}
    
    for prop in criteria.keys():
        # Concatenate all batches
        train_all = torch.cat(train_means[prop], dim=0)
        val_all = torch.cat(val_means[prop], dim=0)
        
        # Compute TRAIN mean
        train_mean_values[prop] = train_all.mean().item()
        
        # Baseline: predict TRAIN mean for both train and val
        baseline_train_mae[prop] = torch.abs(train_all - train_mean_values[prop]).mean().item()
        baseline_val_mae[prop] = torch.abs(val_all - train_mean_values[prop]).mean().item()
    
    print("\nTrain set:")
    print(f"{'Property':<15} {'Mean':<12} {'Std':<12} {'Baseline MAE':<15}")
    print("-" * 60)
    for prop in criteria.keys():
        train_all = torch.cat(train_means[prop], dim=0)
        train_std = train_all.std().item()
        print(f"{prop:<15} {train_mean_values[prop]:<12.4f} {train_std:<12.4f} {baseline_train_mae[prop]:<15.4f}")
    
    print("\nValidation set (predicting TRAIN mean):")
    print(f"{'Property':<15} {'TRAIN Mean':<12} {'Baseline MAE':<15}")
    print("-" * 50)
    for prop in criteria.keys():
        print(f"{prop:<15} {train_mean_values[prop]:<12.4f} {baseline_val_mae[prop]:<15.4f}")
    
    print("\n" + "=" * 80)
    print("YOUR MODEL MUST BEAT THESE VALIDATION BASELINE MAE VALUES!")
    print("=" * 80 + "\n")
    
    # Log baselines to wandb
    if use_wandb and WANDB_AVAILABLE:
        baseline_log = {}
        for prop in criteria.keys():
            baseline_log[f"baseline_train/mae_{prop}"] = baseline_train_mae[prop]
            baseline_log[f"baseline_val/mae_{prop}"] = baseline_val_mae[prop]
            baseline_log[f"baseline_train/mean_{prop}"] = train_mean_values[prop]
            # Also log std for reference
            train_all = torch.cat(train_means[prop], dim=0)
            baseline_log[f"baseline_train/std_{prop}"] = train_all.std().item()
        wandb.log(baseline_log)
    
    # Scheduler: minimize weighted loss
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Initialize history
    history = {
        "train_loss": [],
        "train_loss_weighted": [],
        "val_loss": [],
        "val_loss_weighted": [],
    }
    
    # Add per-property metrics to history
    for prop in criteria.keys():
        history[f"train_mae_{prop}"] = []
        history[f"val_mae_{prop}"] = []
    
    pbar = tqdm(range(epochs), desc="Training Property Reconstruction")
    
    for epoch in pbar:
        # Training
        model.train()
        train_losses = {k: 0.0 for k in criteria.keys()}
        train_maes = {k: 0.0 for k in criteria.keys()}
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(batch)
            
            # Extract pre-computed targets
            targets = extract_property_targets_from_batch(batch)
            
            # Move targets to device (NO normalization - using original scale!)
            for k in targets.keys():
                targets[k] = targets[k].to(device)
            
            # Compute per-property losses (on ORIGINAL scale)
            losses = {}
            for prop in criteria.keys():
                pred = predictions[prop]
                target = targets[prop]
                losses[prop] = criteria[prop](pred, target)
            
            # Weighted total loss
            total_loss = sum(loss_weights[k] * losses[k] for k in losses.keys())
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            batch_size = predictions['homophily'].size(0)
            train_total += batch_size
            
            for prop in criteria.keys():
                train_losses[prop] += losses[prop].item() * batch_size
                # Compute MAE on ORIGINAL scale (already on original scale!)
                with torch.no_grad():
                    mae = torch.abs(predictions[prop] - targets[prop]).mean()
                    train_maes[prop] += mae.item() * batch_size
        
        # Average training metrics
        train_loss_avg = {k: v / train_total for k, v in train_losses.items()}
        train_mae_avg = {k: v / train_total for k, v in train_maes.items()}
        train_loss_weighted = sum(loss_weights[k] * train_loss_avg[k] for k in train_loss_avg.keys())
        
        # Validation
        model.eval()
        val_losses = {k: 0.0 for k in criteria.keys()}
        val_maes = {k: 0.0 for k in criteria.keys()}
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                
                # Forward pass
                predictions = model(batch)
                
                # Extract pre-computed targets
                targets = extract_property_targets_from_batch(batch)
                
                # Move targets to device (NO normalization!)
                for k in targets.keys():
                    targets[k] = targets[k].to(device)
                
                # Compute losses (on ORIGINAL scale)
                losses = {}
                for prop in criteria.keys():
                    pred = predictions[prop]
                    target = targets[prop]
                    losses[prop] = criteria[prop](pred, target)
                
                # Accumulate
                batch_size = predictions['homophily'].size(0)
                val_total += batch_size
                
                for prop in criteria.keys():
                    val_losses[prop] += losses[prop].item() * batch_size
                    # Compute MAE on ORIGINAL scale (already on original scale!)
                    mae = torch.abs(predictions[prop] - targets[prop]).mean()
                    val_maes[prop] += mae.item() * batch_size
        
        # Average validation metrics
        val_loss_avg = {k: v / val_total for k, v in val_losses.items()}
        val_mae_avg = {k: v / val_total for k, v in val_maes.items()}
        val_loss_weighted = sum(loss_weights[k] * val_loss_avg[k] for k in val_loss_avg.keys())
        
        # Update history
        history["train_loss_weighted"].append(train_loss_weighted)
        history["val_loss_weighted"].append(val_loss_weighted)
        
        for prop in criteria.keys():
            history[f"train_mae_{prop}"].append(train_mae_avg[prop])
            history[f"val_mae_{prop}"].append(val_mae_avg[prop])
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch,
                "train/loss_weighted": train_loss_weighted,
                "val/loss_weighted": val_loss_weighted,
                "best_val_loss_weighted": best_val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }
            
            # Add per-property metrics
            for prop in criteria.keys():
                log_dict[f"train/loss_{prop}"] = train_loss_avg[prop]
                log_dict[f"train/mae_{prop}"] = train_mae_avg[prop]
                log_dict[f"val/loss_{prop}"] = val_loss_avg[prop]
                log_dict[f"val/mae_{prop}"] = val_mae_avg[prop]
            
            wandb.log(log_dict)
        
        # Scheduler step
        scheduler.step(val_loss_weighted)
        
        # Early stopping
        if val_loss_weighted < best_val_loss:
            best_val_loss = val_loss_weighted
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update progress bar
        pbar.set_postfix({
            "train_loss": f"{train_loss_weighted:.4f}",
            "val_loss": f"{val_loss_weighted:.4f}",
            "best": f"{best_val_loss:.4f}",
        })
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_property_reconstruction(
    model: PropertyReconstructionModel,
    test_loader: DataLoader,
    device: str = "cpu",
    use_wandb: bool = False,
    K: int = 10,
    normalization_scales: dict = None,
) -> dict:
    """
    Evaluate property reconstruction model on test set.
    
    Handles both consecutive (individual heads) and joint (multi-head) models.
    
    Returns per-property MAE and MSE metrics (on original scale).
    """
    from graph_properties import GraphPropertyComputer
    
    model = model.to(device)
    model.eval()
    
    # Use provided normalization scales or defaults
    if normalization_scales is None:
        normalization_scales = {
            'homophily': 1.0,
            'avg_degree': 20.0,
            'size': 200.0,
            'gini': 1.0,
            'diameter': 20.0,
        }
    
    # Check if using consecutive models
    is_consecutive = hasattr(model, '_consecutive_models')
    
    # Define property lists
    simple_properties = ['avg_degree', 'size', 'gini', 'diameter']  # homophily moved to community
    complex_properties = ['community_presence', 'edge_prob_matrix']
    classification_properties = ['community_detection']  # Classification task (accuracy metric)
    
    # Determine which properties were actually trained
    if is_consecutive and hasattr(model, '_consecutive_models'):
        # Use the properties that were actually trained
        trained_properties = list(model._consecutive_models.keys())
        all_properties = trained_properties
    else:
        # Fallback to all properties
        all_properties = simple_properties + complex_properties + classification_properties
    
    # Accumulate predictions and targets
    all_predictions = {prop: [] for prop in all_properties}
    all_targets = {prop: [] for prop in all_properties}
    
    if is_consecutive:
        # Check if test MAEs were already computed during training (NEW approach)
        if hasattr(model, '_test_maes'):
            print("\n" + "="*80)
            print("USING PRE-COMPUTED TEST RESULTS (from immediate evaluation)")
            print("="*80)
            print("✓ Test MAEs were computed immediately after training each property")
            print("  This avoids any model state issues during evaluation")
            
            # Use pre-computed MAEs
            for prop in all_properties:
                # Still need to collect predictions for detailed analysis
                # But MAE is already correctly computed
                pass
            
        # Still run evaluation to collect predictions for analysis (optional)
        print("\n" + "="*80)
        print("COLLECTING PREDICTIONS FOR ANALYSIS")
        print("="*80)
        
        consecutive_models = model._consecutive_models
        properties = all_properties
        
        for prop in properties:
            prop_model = consecutive_models[prop].to(device)
            prop_model.eval()
            
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    batch = prepare_batch_for_topobench(batch)
                    
                    # Get prediction WITH activation (actual scale)
                    pred = prop_model(batch, apply_activation=True)
                    
                    # Get target
                    if prop == 'community_detection':
                        # NODE-LEVEL: use batch.y directly (num_nodes,)
                        target = batch.y
                    else:
                        # GRAPH-LEVEL: use pre-computed property
                        target = getattr(batch, f'property_{prop}')
                    
                    # Handle reshaping for different property types
                    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
                    
                    if prop == 'community_presence':
                        # Community presence: should be [batch_size, K]
                        expected_shape = (batch_size, K)
                        if target.shape != expected_shape:
                            target = target.view(expected_shape)
                    elif prop == 'edge_prob_matrix':
                        # Edge prob matrix: should be [batch_size, K, K]
                        expected_shape = (batch_size, K, K)
                        if target.shape != expected_shape:
                            target = target.view(expected_shape)
                    elif prop == 'community_detection':
                        # NODE-LEVEL: target is (num_nodes,), pred is (num_nodes, K) logits
                        # Don't reshape - keep as-is for node-level evaluation
                        pass
                    elif prop not in complex_properties and target.dim() == 1:
                        # Simple properties: unsqueeze if scalar
                        target = target.unsqueeze(1)
                    
                    all_predictions[prop].append(pred.cpu())
                    all_targets[prop].append(target.cpu())
            
            print(f"  ✓ Collected predictions for {prop}")
    else:
        # Joint model evaluation (original behavior)
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                
                # Forward pass
                predictions = model(batch)
                
                # Extract pre-computed targets
                targets = extract_property_targets_from_batch(batch)
                
                # Accumulate
                for prop in all_predictions.keys():
                    all_predictions[prop].append(predictions[prop].cpu())
                    all_targets[prop].append(targets[prop].cpu())
    
    # Concatenate all batches
    for prop in all_predictions.keys():
        all_predictions[prop] = torch.cat(all_predictions[prop], dim=0)
        all_targets[prop] = torch.cat(all_targets[prop], dim=0)
    
    # Predictions and targets are ALREADY on original scale (no normalization used)
    # Compute metrics directly
    import numpy as np
    
    # Get number of graphs from first available property
    first_prop = list(all_predictions.keys())[0]
    results = {
        "num_graphs": all_predictions[first_prop].size(0),
    }
    
    # Check if we have pre-computed test MAEs (from immediate evaluation)
    use_precomputed = is_consecutive and hasattr(model, '_test_maes')
    
    for prop in all_predictions.keys():
        pred = all_predictions[prop].numpy()
        target = all_targets[prop].numpy()
        
        # Check if this is a classification property
        is_classification = prop in classification_properties
        
        if is_classification:
            # Classification: compute accuracy
            # Predictions are logits, need to take argmax
            if pred.ndim > 1 and pred.shape[1] > 1:
                pred_classes = np.argmax(pred, axis=1)
            else:
                pred_classes = pred.round()  # Fallback for binary classification
            
            current_accuracy = (pred_classes == target).mean()
            
            # Use pre-computed accuracy if available
            if use_precomputed:
                precomputed_metric = model._test_maes[prop]  # Actually accuracy for classification
                
                # VERIFICATION: Check if current accuracy matches pre-computed
                metric_diff = abs(current_accuracy - precomputed_metric)
                matches = metric_diff < 1e-4
                
                if matches:
                    print(f"✓ {prop}: Pre-computed accuracy verified ({precomputed_metric:.6f})")
                    accuracy = precomputed_metric
                else:
                    print(f"⚠ {prop}: Accuracy MISMATCH! Pre-computed={precomputed_metric:.6f}, Current={current_accuracy:.6f}, Diff={metric_diff:.6f}")
                    print(f"  Using pre-computed value (computed immediately after training)")
                    accuracy = precomputed_metric
            else:
                accuracy = current_accuracy
            
            # Store accuracy (not MAE)
            results[f"test_accuracy_{prop}"] = accuracy
            # For compatibility, also store as test_mae (though it's actually accuracy)
            results[f"test_mae_{prop}"] = accuracy
            results[f"test_mse_{prop}"] = 0.0  # Not applicable
            results[f"test_rmse_{prop}"] = 0.0  # Not applicable
            
            # Store predictions and targets
            results[f"predictions_{prop}"] = pred_classes.tolist()
            results[f"targets_{prop}"] = target.tolist()
        else:
            # Regression: compute MAE
            current_mae = np.abs(pred - target).mean()
            
            # Use pre-computed MAE if available (more reliable)
            if use_precomputed:
                precomputed_mae = model._test_maes[prop]
                
                # VERIFICATION: Check if current MAE matches pre-computed
                mae_diff = abs(current_mae - precomputed_mae)
                matches = mae_diff < 1e-4
                
                if matches:
                    print(f"✓ {prop}: Pre-computed MAE verified ({precomputed_mae:.6f})")
                    mae = precomputed_mae  # Use pre-computed (computed immediately after training)
                else:
                    print(f"⚠ {prop}: MAE MISMATCH! Pre-computed={precomputed_mae:.6f}, Current={current_mae:.6f}, Diff={mae_diff:.6f}")
                    print(f"  Using pre-computed value (computed immediately after training)")
                    mae = precomputed_mae  # Still use pre-computed as it's more reliable
            else:
                mae = current_mae
            
            mse = np.power(pred - target, 2).mean()
            rmse = np.sqrt(mse)
            
            results[f"test_mae_{prop}"] = mae
            results[f"test_mse_{prop}"] = mse
            results[f"test_rmse_{prop}"] = rmse
            
            # Also store raw predictions for analysis
            results[f"predictions_{prop}"] = pred.tolist()
            results[f"targets_{prop}"] = target.tolist()
    
    # Compute baseline: test set "predict-the-mean" performance
    # Use TRAIN mean to predict TEST (realistic baseline)
    print("\n" + "=" * 80)
    print("TEST SET BASELINE: Predict TRAIN Mean")
    print("=" * 80)
    
    # Check if we have train means stored from training (consecutive models)
    has_train_means = hasattr(model, '_train_mean_values')
    
    if has_train_means:
        # Use TRAIN means (correct baseline)
        train_mean_values = model._train_mean_values
        print("✓ Using TRAIN means for baseline (correct approach)")
    else:
        # Fallback: compute from test data (not ideal, but better than nothing)
        train_mean_values = {}
        for prop in all_predictions.keys():
            train_mean_values[prop] = all_targets[prop].numpy().mean()
        print("⚠ WARNING: Using TEST means for baseline (train means not available)")
        print("  This happens with joint training mode. Baseline will be overly optimistic.")
    
    # Compute baseline MAE using train means to predict test (skip classification properties)
    baseline_test_mae = {}
    for prop in all_predictions.keys():
        # Skip classification properties (no MAE baseline)
        if prop in classification_properties:
            baseline_test_mae[prop] = None
            continue
        
        target = all_targets[prop].numpy()
        train_mean = train_mean_values[prop]
        
        # Handle complex properties (train_mean might be a tensor/array)
        if isinstance(train_mean, torch.Tensor):
            train_mean = train_mean.numpy()
        
        baseline_test_mae[prop] = np.abs(target - train_mean).mean()
    
    print(f"\n{'Property':<25} {'TRAIN Mean':<18} {'Baseline/Metric':<15} {'Model Metric':<15} {'Improvement':<15}")
    print("-" * 100)
    for prop in all_predictions.keys():
        is_classification = prop in classification_properties
        
        if is_classification:
            # Classification: show accuracy (no baseline)
            accuracy = results[f"test_accuracy_{prop}"]
            print(f"{prop:<25} {'N/A (classif.)':<18} {'N/A':<15} {accuracy:<15.4f} {'N/A (accuracy)':<15}")
        else:
            # Regression: show MAE with baseline
            model_mae = results[f"test_mae_{prop}"]
            baseline_mae = baseline_test_mae[prop]
            improvement = ((baseline_mae - model_mae) / baseline_mae * 100) if baseline_mae > 0 else 0
            beat_baseline = "✓" if model_mae < baseline_mae else "✗"
            
            # Format train_mean display differently for simple vs complex properties
            if prop in simple_properties:
                train_mean_str = f"{train_mean_values[prop]:.4f}"
            else:
                train_mean_str = f"[shape: {list(train_mean_values[prop].shape) if isinstance(train_mean_values[prop], (torch.Tensor, np.ndarray)) else 'scalar'}]"
            
            print(f"{prop:<25} {train_mean_str:<18} {baseline_mae:<15.4f} {model_mae:<15.4f} {improvement:>13.1f}% {beat_baseline}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Add baselines to results (skip classification properties)
    for prop in all_predictions.keys():
        if prop not in classification_properties:
            results[f"baseline_test_mae_{prop}"] = baseline_test_mae[prop]
            results[f"train_mean_{prop}"] = train_mean_values[prop]  # Store train mean used for baseline
            results[f"improvement_{prop}"] = ((baseline_test_mae[prop] - results[f"test_mae_{prop}"]) / baseline_test_mae[prop] * 100) if baseline_test_mae[prop] > 0 else 0
        else:
            # For classification, store None for baseline-related metrics
            results[f"baseline_test_mae_{prop}"] = None
            results[f"train_mean_{prop}"] = None
            results[f"improvement_{prop}"] = None
    
    # Compute weighted average MAE (using same weights as training)
    # Get typical values for inverse-scale weighting
    avg_degree_typical = normalization_scales.get('avg_degree', 20.0) / 2 if normalization_scales else 10.0
    diameter_typical = normalization_scales.get('diameter', 20.0) / 10 if normalization_scales else 2.0
    size_typical = normalization_scales.get('size', 200.0) / 2 if normalization_scales else 100.0
    
    # Define all possible weights
    all_possible_weights = {
        'homophily': 50.0,
        'gini': 50.0,
        'avg_degree': 1.0 / (avg_degree_typical ** 2),
        'diameter': 1.0 / (diameter_typical ** 2),
        'size': 1.0 / (size_typical ** 2),
        'community_presence': 20.0,  # Similar to homophily/gini (bounded [0,1])
        'edge_prob_matrix': 10.0,  # Matrix has K*K entries, so lower weight
        'community_detection': 50.0,  # Classification: weight by (1 - accuracy) as "error"
    }
    
    # Only use weights for properties that are actually present (and skip classification from weighted MAE)
    regression_props = [p for p in all_predictions.keys() if p not in classification_properties]
    loss_weights = {prop: all_possible_weights[prop] for prop in regression_props if prop in all_possible_weights}
    
    weighted_mae = sum(loss_weights[prop] * results[f"test_mae_{prop}"] for prop in regression_props)
    baseline_weighted_mae = sum(loss_weights[prop] * baseline_test_mae[prop] for prop in regression_props if baseline_test_mae[prop] is not None)
    results["test_mae_weighted"] = weighted_mae
    results["baseline_test_mae_weighted"] = baseline_weighted_mae
    results["improvement_weighted"] = ((baseline_weighted_mae - weighted_mae) / baseline_weighted_mae * 100) if baseline_weighted_mae > 0 else 0
    
    # Separately track accuracy for classification properties
    for prop in classification_properties:
        if prop in all_predictions.keys():
            results[f"test_accuracy_{prop}"] = results.get(f"test_accuracy_{prop}", results[f"test_mae_{prop}"])
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        log_dict = {
            "test/mae_weighted": weighted_mae,
            "test/baseline_mae_weighted": baseline_weighted_mae,
            "test/improvement_weighted": results["improvement_weighted"],
            "test/num_graphs": results["num_graphs"],
        }
        
        for prop in all_predictions.keys():
            is_classification = prop in classification_properties
            
            if is_classification:
                # Log accuracy for classification properties
                log_dict[f"test/accuracy_{prop}"] = results[f"test_accuracy_{prop}"]
                # Don't log MAE/MSE/RMSE for classification
            else:
                # Log MAE metrics for regression properties
                log_dict[f"test/mae_{prop}"] = results[f"test_mae_{prop}"]
                log_dict[f"test/mse_{prop}"] = results[f"test_mse_{prop}"]
                log_dict[f"test/rmse_{prop}"] = results[f"test_rmse_{prop}"]
                if baseline_test_mae[prop] is not None:
                    log_dict[f"test/baseline_mae_{prop}"] = baseline_test_mae[prop]
                    log_dict[f"test/improvement_{prop}"] = results[f"improvement_{prop}"]
                if train_mean_values[prop] is not None:
                    log_dict[f"test/train_mean_{prop}"] = train_mean_values[prop]  # Log train mean used for baseline
        
        wandb.log(log_dict)
    
    return results


def train_downstream(
    model: DownstreamModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    task_level: str = "node",
    task_type: str = "classification",
    loss_type: str = "cross_entropy",
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 20,
    use_wandb: bool = False,
):
    """
    Train the downstream model for classification or regression.
    
    Parameters
    ----------
    task_type : str
        "classification" or "regression"
    loss_type : str
        For classification: "cross_entropy"
        For regression: "mae" or "mse"
    
    Returns
    -------
    dict
        Training history with loss and metrics curves.
    """
    model = model.to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    # Determine criterion and metric based on task type
    if task_type == "regression":
        if loss_type == "mse":
            criterion = nn.MSELoss()
        else:  # mae
            criterion = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # Lower is better
        metric_name = loss_type  # "mae" or "mse"
    else:  # classification
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)  # Higher is better
        metric_name = "accuracy"
    
    best_val_metric = float('inf') if task_type == "regression" else 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], f"train_{metric_name}": [], "val_loss": [], f"val_{metric_name}": []}
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_metric_sum = 0.0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()
            
            out = model(batch)

            if task_level == "graph":
                # Graph-level: one prediction per graph
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)  # Shape: (batch_size, 1)
                else:
                    y = batch.y.long()  # Shape: (batch_size,)
                num_samples = y.size(0)  # Number of graphs
            else:
                # Node-level
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()  # Shape: (num_nodes, 1)
                else:
                    y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                num_samples = y.size(0)  # Number of nodes
            
            loss = criterion(out, y)
            
            # Compute metric
            if task_type == "regression":
                if loss_type == "mae":
                    metric = torch.abs(out - y).mean()
                else:  # mse
                    metric = torch.pow(out - y, 2).mean()
            else:  # classification
                pred = out.argmax(dim=1)
                metric = (pred == y).float().mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * num_samples
            train_metric_sum += metric.item() * num_samples
            train_total += num_samples
        
        train_loss /= train_total
        train_metric = train_metric_sum / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_metric_sum = 0.0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                out = model(batch)

                if task_level == "graph":
                    # Graph-level: one prediction per graph
                    if task_type == "regression":
                        y = batch.y.float().view(-1, 1)  # Shape: (batch_size, 1)
                    else:
                        y = batch.y.long()  # Shape: (batch_size,)
                    num_samples = y.size(0)  # Number of graphs
                else:
                    # Node-level
                    if task_type == "regression":
                        y = batch.y.view(-1, 1).float()  # Shape: (num_nodes, 1)
                    else:
                        y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                    num_samples = y.size(0)  # Number of nodes

                loss = criterion(out, y)
                
                # Compute metric
                if task_type == "regression":
                    if loss_type == "mae":
                        metric = torch.abs(out - y).mean()
                    else:  # mse
                        metric = torch.pow(out - y, 2).mean()
                else:  # classification
                    pred = out.argmax(dim=1)
                    metric = (pred == y).float().mean()

                val_loss += loss.item() * num_samples
                val_metric_sum += metric.item() * num_samples
                val_total += num_samples
        
        val_loss /= val_total
        val_metric = val_metric_sum / val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history[f"train_{metric_name}"].append(train_metric)
        history["val_loss"].append(val_loss)
        history[f"val_{metric_name}"].append(val_metric)
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                f"train/{metric_name}": train_metric,
                "val/loss": val_loss,
                f"val/{metric_name}": val_metric,
                f"best_val_{metric_name}": best_val_metric,
                "lr": optimizer.param_groups[0]['lr'],
            })
        
        # Scheduler step
        scheduler.step(val_metric)
        
        # Early stopping (lower is better for regression, higher for classification)
        is_better = (val_metric < best_val_metric) if task_type == "regression" else (val_metric > best_val_metric)
        
        if is_better:
            best_val_metric = val_metric
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            f"train_{metric_name}": f"{train_metric:.4f}",
            f"val_{metric_name}": f"{val_metric:.4f}",
            "best": f"{best_val_metric:.4f}",
        })
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate(
    model: DownstreamModel,
    test_loader: DataLoader,
    num_classes: int,
    task_type: str = "classification",
    loss_type: str = "cross_entropy",
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Evaluate model on test set."""
    model = model.to(device)
    model.eval()
    
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            out = model(batch)
            
            if model.task_level == "graph":
                # Graph-level: one prediction per graph
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)  # Shape: (batch_size, 1)
                else:
                    y = batch.y.long()  # Shape: (batch_size,)
                num_samples = y.size(0)  # Number of graphs
            else:
                # Node-level
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()  # Shape: (num_nodes, 1)
                else:
                    y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                num_samples = y.size(0)  # Number of nodes
            
            if task_type == "classification":
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
            else:  # regression
                all_preds.extend(out.cpu().squeeze().tolist())
                all_labels.extend(y.cpu().squeeze().tolist())
            
            test_total += num_samples
    
    # Compute metrics
    if task_type == "classification":
        import numpy as np
        test_correct = sum([1 for p, l in zip(all_preds, all_labels) if p == l])
        test_acc = test_correct / test_total
        
        result = {
            "test_accuracy": test_acc,
            "predictions": all_preds,
            "labels": all_labels,
            f"num_{model.task_level}s": test_total,
        }
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/accuracy": test_acc,
                f"test/num_{model.task_level}s": test_total,
            })
    else:  # regression
        import numpy as np
        all_preds_np = np.array(all_preds)
        all_labels_np = np.array(all_labels)
        
        mae = np.abs(all_preds_np - all_labels_np).mean()
        mse = np.power(all_preds_np - all_labels_np, 2).mean()
        rmse = np.sqrt(mse)
        
        result = {
            "test_mae": mae,
            "test_mse": mse,
            "test_rmse": rmse,
            "predictions": all_preds,
            "labels": all_labels,
            f"num_{model.task_level}s": test_total,
        }
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/mae": mae,
                "test/mse": mse,
                "test/rmse": rmse,
                f"test/num_{model.task_level}s": test_total,
            })
    
    return result

# =============================================================================
# Main Pipeline
# =============================================================================

def run_downstream_evaluation(
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    n_evaluation_graphs: int = 200,
    n_train: int = 20,
    mode: str = "linear",  # "linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "gpf", "gpf-plus", "gpf-linear", "gpf-plus-linear", "untrained_frozen"
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "downstream_eval",
    subsample_from_train: bool = False,
    n_graphs: int = None,  # DEPRECATED: kept for backward compatibility
    p_num: int = 5,  # Number of basis vectors for GPF-Plus
    graphuniverse_override: dict | None = None,
    classifier_dropout: float = 0.5,  # Dropout rate for classifier
    input_dropout: float = None,  # Dropout on encoder output (None = use classifier_dropout)
    downstream_task: str | None = None,  # Override task for downstream eval (e.g., "triangle_counting")
    readout_type: str = "sum",  # Pooling type for graph-level tasks: "mean", "max", "sum"
    enable_joint_basic_property_reconstruction: bool = False,  # Use joint training for property reconstruction (old behavior)
    pretraining_config: dict | None = None,  # NEW: Pretraining config from wandb for logging
    basic_properties_to_include: list[str] | None = None,  # Which basic properties to train (default: all)
    community_related_properties_to_include: list[str] | None = None,  # Which community properties to train (default: all)
) -> dict:
    """
    Run full downstream evaluation pipeline for NODE-LEVEL classification.
    
    NEW BEHAVIOR (for fair comparison):
    - Generates (n_evaluation_graphs + n_train) graphs TOGETHER with the same seed
    - First n_evaluation_graphs are used for val/test (FIXED, same for all n_train values)
    - Next n_train graphs are used for training (ADDITIONAL, not from evaluation set)
    - All graphs from same generation run ensures consistent distribution
    - This ensures all models are evaluated on the SAME test set regardless of n_train
    
    **CRITICAL FIX**: The `seed` parameter is ONLY used for model initialization and 
    training randomness. For dataset generation, we ALWAYS use the seed from the 
    pretraining config to ensure downstream evaluation uses THE EXACT SAME DATASET
    as pretraining.
    
    Parameters
    ----------
    run_dir : str or Path
        Path to wandb run directory.
    checkpoint_path : str or Path, optional
        Path to checkpoint. If None, reads from wandb-summary.json.
    n_evaluation_graphs : int
        Number of FIXED evaluation graphs (val + test). These are generated once
        with a fixed seed and are the same for all n_train values.
    n_train : int
        Number of training graphs. These are generated ADDITIONALLY (not taken from
        evaluation set) to ensure fair comparison across different n_train values.
    mode : str
        Evaluation mode: 
        - "linear": frozen encoder + linear classifier
        - "mlp": frozen encoder + MLP classifier  
        - "finetune-linear": unfrozen encoder + linear classifier
        - "finetune-mlp": unfrozen encoder + MLP classifier
        - "scratch": random init + MLP classifier
        - "untrained_frozen": random init frozen + MLP classifier
        - "gpf": GPF prompt + MLP classifier (default)
        - "gpf-linear": GPF prompt + linear classifier
        - "gpf-plus": GPF-Plus prompt + MLP classifier (default)
        - "gpf-plus-linear": GPF-Plus prompt + linear classifier
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate.
    batch_size : int
        Batch size for training (number of graphs per batch).
    patience : int
        Early stopping patience (default: 20).
    device : str
        Device to use.
    seed : int
        Random seed.
    use_wandb : bool
        Whether to log to Weights & Biases.
    wandb_project : str
        W&B project name.
    subsample_from_train : bool
        For TUDataset only: whether to subsample from training set.
    n_graphs : int, optional
        DEPRECATED: Use n_evaluation_graphs instead. Kept for backward compatibility.
    p_num : int
        Number of basis vectors for GPF-Plus (default: 5).
    graphuniverse_override : dict, optional
        Override specific GraphUniverse generation parameters for downstream evaluation.
        Example: {'family_parameters': {'homophily_range': [0.9, 1.0], 'max_communities': 10}}
        Will recursively update the generation_parameters from the pretraining config.
        Default: None (use pretraining config as-is).
    classifier_dropout : float
        Dropout rate for classifier hidden layers (default: 0.5).
    input_dropout : float, optional
        Dropout rate applied to encoder output before classifier. 
        If None, uses classifier_dropout for MLP, 0.0 for linear.
        This helps prevent overfitting on pre-trained features.
    downstream_task : str, optional
        Override the task for downstream evaluation. If specified, this task will be used
        instead of the task from the pretraining config. Useful for transfer learning
        (e.g., pretrain on community_detection, evaluate on triangle_counting).
        Options: "community_detection", "triangle_counting"
    readout_type : str
        Pooling type for graph-level tasks: "mean", "max", "sum" (default: "mean")
    enable_joint_basic_property_reconstruction : bool
        For property reconstruction task: use joint training with multi-head predictor (old behavior).
        Default: False (uses consecutive training with individual heads - NEW DEFAULT).
    basic_properties_to_include : list[str], optional
        Which basic properties to train for basic_property_reconstruction task.
        Options: ["avg_degree", "size", "gini", "diameter"]
        Default: None (trains all 4 properties)
    community_related_properties_to_include : list[str], optional
        Which community-related properties to train for community_related_property_reconstruction task.
        Options: ["homophily", "community_presence", "edge_prob_matrix", "community_detection"]
        Default: None (trains all 4 properties)
    
    Returns
    -------
    dict
        Results including test accuracy and training history.
    """
    # Handle backward compatibility
    if n_graphs is not None and n_evaluation_graphs == 200:
        print(f"WARNING: n_graphs parameter is deprecated. Use n_evaluation_graphs instead.")
        n_evaluation_graphs = n_graphs
    run_dir = Path(run_dir)
    
    # Load config
    print("=" * 60)
    print("Loading configuration...")
    print("=" * 60)
    config = load_wandb_config(run_dir)

    # Detect task level
    task_level = detect_task_level(config)
    
    # Print GraphUniverse config for debugging
    dataset_config = config.get("dataset", {})
    loader_target = dataset_config.get("loader", {}).get("_target_", "")
    if "GraphUniverse" in loader_target:
        print("\n" + "-" * 60)
        print("GRAPHUNIVERSE CONFIG FROM PRE-TRAINING RUN:")
        print("-" * 60)
        gen_params = dataset_config.get("loader", {}).get("parameters", {}).get("generation_parameters", {})
        print(f"Family parameters:")
        for k, v in gen_params.get("family_parameters", {}).items():
            print(f"  {k}: {v}")
        print(f"Universe parameters:")
        for k, v in gen_params.get("universe_parameters", {}).items():
            print(f"  {k}: {v}")
        print("-" * 60 + "\n")
    
    # Determine checkpoint
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path_from_summary(run_dir)
        if checkpoint_path is None:
            raise ValueError("No checkpoint path found. Please provide --checkpoint.")
    
    # Get task type and determine number of classes / outputs
    loader_target = config["dataset"]["loader"]["_target_"]
    dataset_params = config["dataset"]["parameters"]
    
    # Check if downstream task is overridden
    if downstream_task is not None:
        print(f"\n🔄 DOWNSTREAM TASK OVERRIDE: {downstream_task}")
        print(f"   (Pretraining config will be ignored for task setup)")
        
        if downstream_task == "triangle_counting":
            task_type = "regression"
            loss_type = "mae"
            num_classes = 1  # Regression output
            actual_task_level = "graph"  # Triangle counting is graph-level
            print(f"   Task: Triangle Counting (Graph-level Regression)")
            print(f"   Output dimension: {num_classes}")
        elif downstream_task == "community_detection":
            task_type = "classification"
            loss_type = "cross_entropy"
            actual_task_level = "node"  # Community detection is node-level
            # Get K from pretraining config
            if "GraphUniverse" in loader_target:
                gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
                universe_params = gen_params.get("universe_parameters", {})
                num_classes = universe_params.get("K", 10)
            else:
                num_classes = dataset_params.get("num_classes", 2)
            print(f"   Task: Community Detection (Node-level Classification)")
            print(f"   Number of classes (K communities): {num_classes}")
        elif downstream_task == "basic_property_reconstruction":
            task_type = "property_reconstruction"
            loss_type = "weighted_mse_mae"
            num_classes = 4  # 4 properties (homophily moved to community task)
            actual_task_level = "graph"  # Property reconstruction is graph-level
            print(f"   Task: Basic Property Reconstruction (Graph-level Multi-output Regression)")
            print(f"   Properties: avg_degree, size, gini, diameter")
        elif downstream_task == "community_related_property_reconstruction":
            task_type = "community_property_reconstruction"
            loss_type = "weighted_bce_mse"
            num_classes = 4  # 4 properties (3 regression + 1 classification)
            actual_task_level = "graph"  # Property reconstruction is graph-level
            print(f"   Task: Community-Related Property Reconstruction (Graph-level Multi-output)")
            print(f"   Properties: homophily, community_presence, edge_prob_matrix, community_detection")
            print(f"   Number of outputs: {num_classes} (3 regression + 1 classification)")
        else:
            raise ValueError(f"Unknown downstream_task: {downstream_task}")
    else:
        # Use task from pretraining config (but could still be overridden by downstream_task arg)
        # Note: If downstream_task was set above, we've already configured task_type/num_classes
        # This else block is for when downstream_task is None
        task_type = dataset_params.get("task", "classification")  # "classification" or "regression"
        loss_type = dataset_params.get("loss_type", "cross_entropy")  # "cross_entropy", "mae", "mse"
        actual_task_level = task_level  # Use detected task level from config
        
        if "GraphUniverse" in loader_target:
            # For GraphUniverse: check if it's triangle counting or community detection
            gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
            task_name = gen_params.get("task", "community_detection")
            
            if task_name == "triangle_counting":
                num_classes = 1  # Regression output
                print(f"Task: Triangle Counting (Graph-level Regression)")
                print(f"Output dimension: {num_classes}")
            else:
                # Community detection
                universe_params = gen_params.get("universe_parameters", {})
                num_classes = universe_params.get("K", 10)
                print(f"Task: Community Detection (Node-level Classification)")
                print(f"Number of classes (K communities): {num_classes}")
        else:
            # For other datasets: use num_classes from config
            num_classes = dataset_params.get("num_classes", 2)
            print(f"Number of classes: {num_classes}")
            print(f"Task type: {task_type}")
            if task_type == "regression":
                print(f"Loss type: {loss_type}")
    
    # Create dataset with FIXED evaluation set + ADDITIONAL training graphs
    print("\n" + "=" * 60)
    print("Creating dataset with FIXED evaluation set...")
    print("=" * 60)
    
    # Determine if GraphUniverse
    loader_target = config["dataset"]["loader"]["_target_"]
    is_graph_universe = "GraphUniverse" in loader_target
    
    # CRITICAL: Extract pretraining dataset seed from config
    # This ensures downstream evaluation uses THE SAME dataset
    pretraining_seed = seed  # Default to function arg
    if is_graph_universe:
        gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
        family_params = gen_params.get("family_parameters", {})
        universe_params = gen_params.get("universe_parameters", {})
        
        # Try to get seed from family_parameters first, then universe_parameters
        pretraining_seed = family_params.get("seed") or universe_params.get("seed") or seed
        
        print("\n" + "=" * 80)
        print("SEED VERIFICATION (GraphUniverse)")
        print("=" * 80)
        print(f"🔍 PRETRAINING DATASET SEED:  {pretraining_seed}")
        print(f"🔍 DOWNSTREAM EVAL SEED:      {pretraining_seed}  (SAME - using pretraining seed)")
        print(f"   (Function arg seed={seed} was {'IGNORED' if pretraining_seed != seed else 'same'} for dataset generation)")
        print(f"   ✓ Ensures downstream evaluation uses THE EXACT SAME DATASET as pretraining!")
        print("=" * 80)
    
    if is_graph_universe:
        # V2 APPROACH: Generate eval and train datasets INDEPENDENTLY with different family seeds
        # This ensures eval set is ALWAYS identical regardless of train set size
        print("\n" + "=" * 80)
        print("GENERATING DATASETS WITH INDEPENDENT SEEDS (v2)")
        print("=" * 80)
        print(f"Base seed: {pretraining_seed}")
        print(f"  Eval:  universe_seed={pretraining_seed}, family_seed={pretraining_seed + 1}")
        print(f"  Train: universe_seed={pretraining_seed}, family_seed={pretraining_seed + 2}")
        print("\n✓ Eval set will be IDENTICAL across all train set sizes")
        print("✓ Train set generation is INDEPENDENT of eval set")
        print("=" * 80)
        
        # Generate EVAL set (FIXED, always the same)
        eval_dataset, eval_data_dir, eval_dataset_info = create_dataset_from_config_v2(
            config,
            n_graphs=n_evaluation_graphs,
            seed=pretraining_seed,
            dataset_purpose="eval",
            graphuniverse_override=graphuniverse_override,
            downstream_task=downstream_task,
        )
        
        # Apply transforms to eval set
        transforms_config = config.get("transforms")
        eval_preprocessor = apply_transforms(eval_dataset, eval_data_dir, transforms_config)
        eval_data_list = eval_preprocessor.data_list
        
        print(f"✓ Generated {len(eval_data_list)} EVAL graphs (family_seed={pretraining_seed + 1})")
        
        # Generate TRAIN set (INDEPENDENT)
        train_dataset, train_data_dir, train_dataset_info = create_dataset_from_config_v2(
            config,
            n_graphs=n_train,
            seed=pretraining_seed,
            dataset_purpose="train",
            graphuniverse_override=graphuniverse_override,
            downstream_task=downstream_task,
        )
        
        # Apply transforms to train set
        train_preprocessor = apply_transforms(train_dataset, train_data_dir, transforms_config)
        train_data = train_preprocessor.data_list
        
        print(f"✓ Generated {len(train_data)} TRAIN graphs (family_seed={pretraining_seed + 2})")
        
        # Split eval data into val/test (50/50)
        import random
        random.seed(pretraining_seed)
        eval_indices = list(range(len(eval_data_list)))
        random.shuffle(eval_indices)
        n_val = len(eval_indices) // 2
        val_indices = eval_indices[:n_val]
        test_indices = eval_indices[n_val:]
        val_data = [eval_data_list[i] for i in val_indices]
        test_data = [eval_data_list[i] for i in test_indices]
        
        print(f"\nFinal split:")
        print(f"  Training: {len(train_data)} graphs (INDEPENDENT generation)")
        print(f"  Validation: {len(val_data)} graphs (from FIXED eval set)")
        print(f"  Test: {len(test_data)} graphs (from FIXED eval set)")
        print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} graphs")
        
        # Pre-compute properties for property reconstruction tasks (v2 approach - already split)
        if task_type in ["property_reconstruction", "community_property_reconstruction"]:
            from graph_properties import add_properties_to_dataset
            
            print("\n" + "=" * 60)
            print("Pre-computing graph properties...")
            print("=" * 60)
            
            # Get K from config
            gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
            universe_params = gen_params.get("universe_parameters", {})
            K = universe_params.get("K", 10)
            
            # Determine which properties to compute
            if task_type == "community_property_reconstruction":
                include_complex = True
                print("  Computing: homophily, community_presence, edge_prob_matrix")
            else:
                include_complex = False
                print("  Computing: avg_degree, size, gini, diameter")
            
            # Pre-compute properties for each split separately
            print(f"  Processing training set ({len(train_data)} graphs)...")
            train_data = add_properties_to_dataset(
                train_data, 
                K=K, 
                include_complex=include_complex,
                verbose=False,
            )
            print(f"  Processing validation set ({len(val_data)} graphs)...")
            val_data = add_properties_to_dataset(
                val_data, 
                K=K, 
                include_complex=include_complex,
                verbose=False,
            )
            print(f"  Processing test set ({len(test_data)} graphs)...")
            test_data = add_properties_to_dataset(
                test_data, 
                K=K, 
                include_complex=include_complex,
                verbose=False,
            )
            print(f"✓ Properties pre-computed for all splits")
        
    else:
        # For non-GraphUniverse: use original approach
        print("Non-GraphUniverse dataset: using original splitting approach...")
        dataset, data_dir, dataset_info = create_dataset_from_config(
            config, 
            n_graphs=n_evaluation_graphs, 
            n_train=n_train,
            subsample_train=subsample_from_train,
            seed=pretraining_seed,  # Use pretraining seed here too
            graphuniverse_override=graphuniverse_override,
            downstream_task=downstream_task,
        )

        # Apply transforms
        transforms_config = config.get("transforms")
        preprocessor = apply_transforms(dataset, data_dir, transforms_config)
        
        # Get data list
        data_list = preprocessor.data_list
        print(f"Dataset size: {len(data_list)} graphs")
        
        # Pre-compute properties for property reconstruction tasks
        if task_type in ["property_reconstruction", "community_property_reconstruction"]:
            from graph_properties import add_properties_to_dataset
            
            print("\n" + "=" * 60)
            print("Pre-computing graph properties...")
            print("=" * 60)
            
            # Get K from config
            if "GraphUniverse" in loader_target:
                gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
                universe_params = gen_params.get("universe_parameters", {})
                K = universe_params.get("K", 10)
            else:
                K = 10  # Default
            
            # Determine which properties to compute
            if task_type == "community_property_reconstruction":
                # Need complex properties (community_presence, edge_prob_matrix) + homophily
                include_complex = True
                print("  Computing: homophily, community_presence, edge_prob_matrix")
            else:
                # Basic property reconstruction: only simple properties (no complex ones, no homophily)
                include_complex = False
                print("  Computing: avg_degree, size, gini, diameter")
            
            # Pre-compute properties (modifies data_list in-place)
            data_list = add_properties_to_dataset(
                data_list, 
                K=K, 
                include_complex=include_complex,
                verbose=True,
            )
            print(f"✓ Properties pre-computed for all {len(data_list)} graphs")
        
        # Create splits
        train_data, val_data, test_data = create_data_splits(
            data_list, 
            n_train, 
            dataset_info=dataset_info,
            seed=pretraining_seed  # Use pretraining seed for consistency
        )
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Load encoder (pre-trained, random init, or prompted)
    print("\n" + "=" * 60)
    
    # Check if using prompt-based methods
    is_prompt_method = mode in ["gpf", "gpf-plus", "gpf-linear", "gpf-plus-linear"]
    
    # IMPORTANT: Prompt methods are NOT supported for property reconstruction yet
    if is_prompt_method and task_type in ["property_reconstruction", "community_property_reconstruction"]:
        raise ValueError(
            f"Prompt methods (GPF, GPF-Plus) are not yet supported for property reconstruction. "
            f"Please use mode='linear', 'mlp', 'finetune', or 'scratch' instead."
        )
    
    if is_prompt_method:
        # Import prompt tuning utilities
        from prompt_tuning import create_prompted_model
        
        # Determine prompt type and classifier type from mode
        if mode == "gpf" or mode == "gpf-linear":
            prompt_type = "gpf"
            classifier_type = "linear" if mode == "gpf-linear" else "mlp"
        else:  # gpf-plus or gpf-plus-linear
            prompt_type = "gpf-plus"
            classifier_type = "linear" if mode == "gpf-plus-linear" else "mlp"
        
        print(f"Creating prompted model ({mode.upper()}: {prompt_type} + {classifier_type})...")
        print("=" * 60)
        
        # Create prompted model (handles encoder + prompt + classifier)
        downstream_model, prompt_info = create_prompted_model(
            config=config,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            prompt_type=prompt_type,
            p_num=p_num,
            task_level=actual_task_level,  # Use actual_task_level (may differ from pretraining)
            device=device,
            classifier_type=classifier_type,
            mode="finetune",  # Always use pre-trained for prompts
            readout_type=readout_type,  # Pass readout type for graph-level tasks
            task_type=task_type,  # Pass task type (classification vs regression)
        )
        
        # Skip the manual encoder/classifier creation below
        skip_manual_model_creation = True
        
    else:
        skip_manual_model_creation = False
        
        if mode == "scratch":
            print("Creating randomly initialized encoder (scratch baseline)...")
            print("=" * 60)
            feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
        elif mode == "untrained_frozen":
            print("Creating randomly initialized encoder (untrained frozen baseline)...")
            print("  This tests: random weights + frozen encoder + MLP classifier")
            print("=" * 60)
            feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
        else:
            print("Loading pre-trained encoder...")
            print("=" * 60)
            feature_encoder, backbone, hidden_dim = load_pretrained_encoder(
                config, checkpoint_path, device=device
            )

        # Create appropriate encoder wrapper based on actual task level
        # (use actual_task_level which may differ from pretraining if downstream_task is set)
        if actual_task_level == "graph":
            encoder = GraphLevelEncoder(
                feature_encoder=feature_encoder,
                backbone=backbone,
                readout_type=readout_type,
            )
            print(f"  Using GraphLevelEncoder with {readout_type} pooling")
        else:
            encoder = PretrainedEncoder(
                feature_encoder=feature_encoder,
                backbone=backbone,
            )
            print(f"  Using PretrainedEncoder (node-level)")
    
    # Verify encoder produces meaningful outputs (skip for prompt methods, verified internally)
    if not is_prompt_method:
        print("\n" + "=" * 60)
        print("Verifying encoder outputs...")
        print("=" * 60)
        verify_result = verify_encoder_outputs(encoder, train_loader, device=device)
        print(f"  Status: {verify_result['status']}")
        print(f"  Output shape: {verify_result.get('shape', 'N/A')}")
        print(f"  Mean: {verify_result.get('mean', 'N/A'):.4f}, Std: {verify_result.get('std', 'N/A'):.4f}")
        print(f"  Range: [{verify_result.get('min', 'N/A'):.4f}, {verify_result.get('max', 'N/A'):.4f}]")
        if verify_result.get('issues'):
            print(f"  ISSUES: {verify_result['issues']}")
    else:
        verify_result = {"status": "SKIPPED (prompt method)"}
    
    # Initialize wandb if requested
    if use_wandb:
        if not WANDB_AVAILABLE:
            print("WARNING: wandb not installed, skipping logging")
            use_wandb = False
        else:
            wandb_config = {
                "mode": mode,
                "task_type": task_type,
                "task_level": task_level,
                "loss_type": loss_type,
                "n_evaluation_graphs": n_evaluation_graphs,
                "n_val": len(val_data),
                "n_test": len(test_data),
                "n_train": n_train,
                "n_train_actual": len(train_data),
                "n_total_generated": n_evaluation_graphs + n_train if is_graph_universe else len(data_list),
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": seed,
                "num_classes": num_classes,
                "pretraining_run": str(run_dir),
                "encoder_verify": verify_result,
                "fixed_eval_set": is_graph_universe,
                "downstream_task": downstream_task,
                "readout_type": readout_type,
                "classifier_dropout": classifier_dropout,
                "input_dropout": input_dropout,
            }
            
            # Add pretraining configuration for plotting/filtering
            if pretraining_config is not None:
                # Helper function to flatten nested dicts (reuse existing one)
                def flatten_dict(d, parent_key='', sep='/'):
                    """Flatten nested dict for wandb plotting."""
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                # Extract dataset config (from pretraining)
                if "dataset" in pretraining_config:
                    dataset_config = pretraining_config["dataset"]
                    if isinstance(dataset_config, dict) and "value" in dataset_config:
                        dataset_config = dataset_config["value"]
                    
                    # Flatten and add with prefix
                    flattened_dataset = flatten_dict(dataset_config)
                    for key, value in flattened_dataset.items():
                        # Add prefix to avoid conflicts with downstream config
                        wandb_config[f"pretrain/dataset/{key}"] = value
                
                # Extract model config (from pretraining)
                if "model" in pretraining_config:
                    model_config = pretraining_config["model"]
                    if isinstance(model_config, dict) and "value" in model_config:
                        model_config = model_config["value"]
                    
                    # Flatten and add with prefix
                    flattened_model = flatten_dict(model_config)
                    for key, value in flattened_model.items():
                        # Add prefix to avoid conflicts with downstream config
                        wandb_config[f"pretrain/model/{key}"] = value
                
                # Extract other useful pretraining parameters
                for param_key in ["optimizer", "seed", "trainer", "loss"]:
                    if param_key in pretraining_config:
                        param_config = pretraining_config[param_key]
                        if isinstance(param_config, dict) and "value" in param_config:
                            param_config = param_config["value"]
                        
                        # Handle scalar values (like seed) vs dict values
                        if isinstance(param_config, dict):
                            # Flatten and add with prefix
                            flattened_param = flatten_dict(param_config)
                            for key, value in flattened_param.items():
                                wandb_config[f"pretrain/{param_key}/{key}"] = value
                        else:
                            # Direct scalar value
                            wandb_config[f"pretrain/{param_key}"] = param_config
            
            # Add GraphUniverse override info for plotting/filtering
            if graphuniverse_override is not None:
                # Store the full override as nested dict
                wandb_config["graphuniverse_override"] = graphuniverse_override
                
                # Also flatten override parameters for easier plotting in wandb
                # This allows you to plot/filter by specific override values
                def flatten_dict(d, parent_key='', sep='/'):
                    """Flatten nested dict for wandb plotting."""
                    items = []
                    for k, v in d.items():
                        new_key = f"{parent_key}{sep}{k}" if parent_key else k
                        if isinstance(v, dict):
                            items.extend(flatten_dict(v, new_key, sep=sep).items())
                        else:
                            items.append((new_key, v))
                    return dict(items)
                
                flattened = flatten_dict(graphuniverse_override)
                for key, value in flattened.items():
                    # Add prefix to avoid conflicts
                    wandb_config[f"override/{key}"] = value
                
                # Add a hash for easy identification of override configs
                import hashlib
                override_hash = hashlib.md5(
                    json.dumps(graphuniverse_override, sort_keys=True).encode()
                ).hexdigest()[:8]
                wandb_config["override_hash"] = override_hash
                wandb_config["has_override"] = True
                
                # Add human-readable labels for common overrides
                # This makes plotting/grouping easier in wandb
                if "family_parameters" in graphuniverse_override:
                    family_params = graphuniverse_override["family_parameters"]
                    
                    # Homophily label
                    if "homophily_range" in family_params:
                        h_range = family_params["homophily_range"]
                        if h_range[0] >= 0.9:
                            wandb_config["override_label/homophily"] = "high"
                        elif h_range[1] <= 0.1:
                            wandb_config["override_label/homophily"] = "low"
                        else:
                            wandb_config["override_label/homophily"] = f"{h_range[0]:.1f}-{h_range[1]:.1f}"
                    
                    # Other family params
                    if "min_n_nodes" in family_params:
                        wandb_config["override_label/graph_size"] = f"{family_params['min_n_nodes']}-{family_params.get('max_n_nodes', 'N/A')}"
                    if "max_communities" in family_params:
                        wandb_config["override_label/max_communities"] = family_params["max_communities"]
            else:
                wandb_config["graphuniverse_override"] = None
                wandb_config["has_override"] = False
                wandb_config["override_hash"] = "none"
                wandb_config["override_label/homophily"] = "pretraining_config"
            
            # Add prompt-specific config
            if is_prompt_method:
                wandb_config["hidden_dim"] = prompt_info["hidden_dim"]
                wandb_config["prompt_type"] = prompt_info["prompt_type"]
                wandb_config["classifier_type"] = prompt_info["classifier_type"]
                wandb_config["p_num"] = prompt_info["p_num"]
                wandb_config["prompt_params"] = prompt_info["prompt_params"]
            else:
                wandb_config["hidden_dim"] = hidden_dim
            
            wandb.init(project=wandb_project, config=wandb_config)
    
    # Create classifier/regressor based on mode (skip if already created by prompt method)
    if not skip_manual_model_creation:
        print("\n" + "=" * 60)
        print(f"Setting up downstream evaluation (mode: {mode})...")
        print("=" * 60)
        print(f"  Hidden dim: {hidden_dim}")
        
        if task_type == "property_reconstruction":
            print(f"  Multi-property reconstruction: {num_classes} properties")
            print(f"  Properties: avg_degree, size, gini, diameter")
            
            # Extract K from config for complex properties
            from graph_properties import extract_K_from_config
            try:
                K = extract_K_from_config(config)
                print(f"  K (number of communities): {K}")
            except:
                K = 10  # Default fallback
                print(f"  K (number of communities): {K} (default fallback)")
            
            # Determine if encoder should be frozen
            freeze_encoder = mode in ["linear", "mlp", "untrained_frozen"]
            
            # Determine head type based on mode
            # finetune-linear and linear: use linear heads
            # finetune-mlp and mlp: use MLP heads
            use_mlp_heads = mode in ["mlp", "finetune-mlp", "scratch"]
            
            property_regressor = MultiPropertyRegressor(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                dropout=classifier_dropout,
                input_dropout=input_dropout,
                use_mlp_heads=use_mlp_heads,
                K=K,  # Pass K for complex properties
            )
            
            head_type = "MLP" if use_mlp_heads else "Linear"
            encoder_status = "frozen" if freeze_encoder else "unfrozen"
            print(f"  Using {head_type} heads for each property")
            print(f"  Encoder: {encoder_status}")
            
            # Create property reconstruction model
            downstream_model = PropertyReconstructionModel(
                encoder=encoder,
                property_regressor=property_regressor,
                freeze_encoder=freeze_encoder,
            )
        
        elif task_type == "community_property_reconstruction":
            print(f"  Community-related property reconstruction: {num_classes} properties")
            print(f"  Properties: homophily, community_presence, edge_prob_matrix, community_detection")
            print(f"    - 3 regression (MAE metric): homophily, community_presence, edge_prob_matrix")
            print(f"    - 1 classification (accuracy metric): community_detection")
            
            # Extract K from config for complex properties
            from graph_properties import extract_K_from_config
            try:
                K = extract_K_from_config(config)
                print(f"  K (number of communities): {K}")
            except:
                K = 10  # Default fallback
                print(f"  K (number of communities): {K} (default fallback)")
            
            # Determine if encoder should be frozen
            freeze_encoder = mode in ["linear", "mlp", "untrained_frozen"]
            
            # Determine head type based on mode
            use_mlp_heads = mode in ["mlp", "finetune-mlp", "scratch"]
            
            property_regressor = CommunityPropertyRegressor(
                input_dim=hidden_dim,
                hidden_dim=hidden_dim // 2,
                dropout=classifier_dropout,
                input_dropout=input_dropout,
                use_mlp_heads=use_mlp_heads,
                K=K,
            )
            
            head_type = "MLP" if use_mlp_heads else "Linear"
            encoder_status = "frozen" if freeze_encoder else "unfrozen"
            print(f"  Using {head_type} heads for each property")
            print(f"  Encoder: {encoder_status}")
            
            # Create community property reconstruction model
            downstream_model = CommunityPropertyReconstructionModel(
                encoder=encoder,
                property_regressor=property_regressor,
                freeze_encoder=freeze_encoder,
            )
            
        elif task_type == "regression":
            print(f"  Output dim: {num_classes} (regression)")
            
            # Determine if encoder should be frozen
            freeze_encoder = mode in ["linear", "mlp", "untrained_frozen"]
            
            # Determine classifier type based on mode
            if mode in ["linear", "finetune-linear"]:
                # Linear head
                dropout_rate = input_dropout if input_dropout is not None else 0.0
                classifier = LinearRegressor(hidden_dim, num_classes, dropout=dropout_rate)
                encoder_status = "frozen" if freeze_encoder else "unfrozen"
                print(f"  Linear regressor with input dropout: {dropout_rate}")
                print(f"  Encoder: {encoder_status}")
            else:  # mlp, finetune-mlp, scratch, untrained_frozen
                # MLP head
                classifier = MLPRegressor(
                    hidden_dim, 
                    num_classes, 
                    dropout=classifier_dropout,
                    input_dropout=input_dropout,
                )
                final_input_dropout = input_dropout if input_dropout is not None else classifier_dropout
                encoder_status = "frozen" if freeze_encoder else "unfrozen"
                print(f"  MLP regressor with input dropout: {final_input_dropout}, hidden dropout: {classifier_dropout}")
                print(f"  Encoder: {encoder_status}")
            
            # Create downstream model
            downstream_model = DownstreamModel(
                encoder=encoder,
                classifier=classifier,
                freeze_encoder=freeze_encoder,
                task_level=actual_task_level,
            )
        else:
            print(f"  Num classes: {num_classes}")
            
            # Determine if encoder should be frozen
            freeze_encoder = mode in ["linear", "mlp", "untrained_frozen"]
            
            # Determine classifier type based on mode
            if mode in ["linear", "finetune-linear"]:
                # Linear head
                dropout_rate = input_dropout if input_dropout is not None else 0.0
                classifier = LinearClassifier(hidden_dim, num_classes, dropout=dropout_rate)
                encoder_status = "frozen" if freeze_encoder else "unfrozen"
                print(f"  Linear classifier with input dropout: {dropout_rate}")
                print(f"  Encoder: {encoder_status}")
            else:  # mlp, finetune-mlp, scratch, untrained_frozen
                # MLP head
                classifier = MLPClassifier(
                    hidden_dim, 
                    num_classes, 
                    dropout=classifier_dropout,
                    input_dropout=input_dropout,
                )
                final_input_dropout = input_dropout if input_dropout is not None else classifier_dropout
                encoder_status = "frozen" if freeze_encoder else "unfrozen"
                print(f"  MLP classifier with input dropout: {final_input_dropout}, hidden dropout: {classifier_dropout}")
                print(f"  Encoder: {encoder_status}")
            
            # Create downstream model
            downstream_model = DownstreamModel(
                encoder=encoder,
                classifier=classifier,
                freeze_encoder=freeze_encoder,
                task_level=actual_task_level,
            )
    
    # Set model to train mode before checking state
    downstream_model.train()
    
    # Get detailed model state info
    if not is_prompt_method:
        state_info = downstream_model.get_model_state_info()
        total_params = sum(p.numel() for p in downstream_model.parameters())
        trainable_params = sum(p.numel() for p in downstream_model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Encoder frozen: {freeze_encoder}")
        print(f"  - Encoder params: {state_info['encoder_params']:,} ({state_info['encoder_trainable']:,} trainable)")
        print(f"  - Encoder in train mode: {state_info['encoder_in_train_mode']} (should be {not freeze_encoder})")
        
        # Different models have different names for the prediction head
        if 'classifier_params' in state_info:
            print(f"  - Classifier params: {state_info['classifier_params']:,}")
        elif 'regressor_params' in state_info:
            print(f"  - Regressor params: {state_info['regressor_params']:,}")
    else:
        # Already printed by create_prompted_model
        pass
    
    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    
    if task_type == "property_reconstruction":
        # Get K for property computer
        if "GraphUniverse" in loader_target:
            gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
            universe_params = gen_params.get("universe_parameters", {})
            K = universe_params.get("K", 10)
        else:
            K = 10  # Default
        
        # Extract normalization scales from config
        normalization_scales = extract_normalization_scales_from_config(config)
        
        history = train_property_reconstruction(
            model=downstream_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,  # Pass test loader for immediate evaluation
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            use_wandb=use_wandb,
            K=K,
            normalization_scales=normalization_scales,
            enable_joint_training=enable_joint_basic_property_reconstruction,
            properties_to_include=basic_properties_to_include,
        )
    
    elif task_type == "community_property_reconstruction":
        # Get K for property computer
        if "GraphUniverse" in loader_target:
            gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
            universe_params = gen_params.get("universe_parameters", {})
            K = universe_params.get("K", 10)
        else:
            K = 10  # Default
        
        # Extract normalization scales from config (needed for evaluation)
        normalization_scales = extract_normalization_scales_from_config(config)
        
        # For community properties, we can use the same training function
        # but we need to create a wrapper that uses the community property extraction
        history = train_property_reconstruction(
            model=downstream_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            use_wandb=use_wandb,
            K=K,
            normalization_scales=normalization_scales,  # Pass for weighted MAE computation
            enable_joint_training=enable_joint_basic_property_reconstruction,
            properties_to_include=community_related_properties_to_include,
        )
    
    else:
        history = train_downstream(
            model=downstream_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_classes=num_classes,
            task_type=task_type,
            loss_type=loss_type,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            use_wandb=use_wandb,
            task_level=actual_task_level,
        )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    if task_type == "property_reconstruction":
        results = evaluate_property_reconstruction(
            model=downstream_model,
            test_loader=test_loader,
            device=device,
            use_wandb=use_wandb,
            K=K,
            normalization_scales=normalization_scales,
        )
    elif task_type == "community_property_reconstruction":
        # For community property reconstruction, call evaluate_property_reconstruction
        # This will compute test metrics and log them to wandb
        results = evaluate_property_reconstruction(
            model=downstream_model,
            test_loader=test_loader,
            device=device,
            use_wandb=use_wandb,
            K=K,
            normalization_scales=normalization_scales,
        )
        print("✓ Community property reconstruction evaluation completed")
    else:
        results = evaluate(
            model=downstream_model,
            test_loader=test_loader,
            num_classes=num_classes,
            task_type=task_type,
            loss_type=loss_type,
            device=device,
            use_wandb=use_wandb,
        )
    
    # Print results based on task type
    if task_type == "property_reconstruction":
        print(f"\n{'='*60}")
        print("PROPERTY RECONSTRUCTION RESULTS")
        print('='*60)
        print(f"Weighted MAE: {results['test_mae_weighted']:.4f}")
        print("\nPer-property MAE:")
        for prop in ['avg_degree', 'size', 'gini', 'diameter']:
            print(f"  {prop:15s}: {results[f'test_mae_{prop}']:.4f} (RMSE: {results[f'test_rmse_{prop}']:.4f})")
        print('='*60)
    elif task_type == "community_property_reconstruction":
        print(f"\n{'='*60}")
        print("COMMUNITY PROPERTY RECONSTRUCTION RESULTS")
        print('='*60)
        print(f"Weighted MAE (regression only): {results['test_mae_weighted']:.4f}")
        print(f"Baseline MAE: {results['baseline_test_mae_weighted']:.4f}")
        print(f"Improvement: {results['improvement_weighted']:.2f}%")
        print("\nPer-property Results:")
        # Determine which properties were tested
        tested_properties = [prop.replace('test_mae_', '') for prop in results.keys() if prop.startswith('test_mae_') and not prop.endswith('_weighted')]
        classification_props_tested = [p for p in tested_properties if p in ['community_detection']]
        regression_props_tested = [p for p in tested_properties if p not in classification_props_tested]
        
        # Show regression properties
        if regression_props_tested:
            print("  Regression (MAE):")
            for prop in regression_props_tested:
                mae = results[f'test_mae_{prop}']
                rmse = results.get(f'test_rmse_{prop}', 0)
                baseline = results.get(f'baseline_test_mae_{prop}', 0)
                improvement = results.get(f'improvement_{prop}', 0)
                if baseline and baseline > 0:
                    print(f"    {prop:20s}: MAE={mae:.4f}, RMSE={rmse:.4f}, Baseline={baseline:.4f}, Improvement={improvement:.1f}%")
                else:
                    print(f"    {prop:20s}: MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        # Show classification properties
        if classification_props_tested:
            print("  Classification (Accuracy):")
            for prop in classification_props_tested:
                accuracy = results.get(f'test_accuracy_{prop}', results.get(f'test_mae_{prop}', 0))
                print(f"    {prop:20s}: Accuracy={accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print('='*60)
    elif task_type == "regression":
        print(f"\nTest MAE: {results['test_mae']:.4f}")
        print(f"Test MSE: {results['test_mse']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
    else:
        print(f"\nTest Accuracy: {results['test_accuracy']:.4f}")
    
    if task_type not in ["property_reconstruction", "community_property_reconstruction"]:
        print(f"Total test {actual_task_level}s: {results[f'num_{actual_task_level}s']}")
    
    # Close wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Add metadata to results
    results["mode"] = mode
    results["task_type"] = task_type  # Add this so we can properly detect the task type later
    results["n_evaluation_graphs"] = n_evaluation_graphs
    results["n_train"] = n_train
    results["n_train_actual"] = len(train_data)
    results["n_val"] = len(val_data)
    results["n_test"] = len(test_data)
    results["num_classes"] = num_classes
    results["history"] = history
    results["from_scratch"] = (mode == "scratch")
    # Determine if encoder was frozen (for prompt methods, check mode; otherwise use freeze_encoder variable)
    if is_prompt_method:
        results["encoder_frozen"] = True  # Prompt methods always freeze encoder
    else:
        results["encoder_frozen"] = freeze_encoder
    results["fixed_eval_set"] = is_graph_universe
    results["config"] = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "hidden_dim": prompt_info["hidden_dim"] if is_prompt_method else hidden_dim,
    }
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    if is_graph_universe:
        print(f"Dataset approach: FIXED eval set (first {n_evaluation_graphs} graphs) + ADDITIONAL train (next {n_train} graphs)")
        print(f"All graphs generated together with seed={pretraining_seed}")
    else:
        print(f"Dataset approach: Traditional split")
    print(f"Training graphs: {len(train_data)}")
    print(f"Validation graphs: {len(val_data)}")
    print(f"Test graphs: {len(test_data)}")
    
    if task_type == "property_reconstruction":
        print(f"Weighted MAE: {results['test_mae_weighted']:.4f}")
        print("Per-property MAE:")
        for prop in ['avg_degree', 'size', 'gini', 'diameter']:
            print(f"  {prop}: {results[f'test_mae_{prop}']:.4f}")
    elif task_type == "community_property_reconstruction":
        print("✓ Community-related property reconstruction completed")
        print("  (See detailed results above for each property)")
    elif task_type == "regression":
        print(f"Test MAE: {results['test_mae']:.4f}")
        print(f"Test MSE: {results['test_mse']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
    else:
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print("=" * 60)
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Downstream evaluation of pre-trained inductive graph models."
    )
    parser.add_argument(
        "--run_dir", type=str, default="data/outputs/wandb/run-20260128_145853-1m1efyqb",
        help="Path to wandb run directory"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--n_evaluation_graphs", type=int, default=200,
        help="Number of FIXED evaluation graphs (val+test, same for all n_train). Default: 200"
    )
    parser.add_argument(
        "--n_train", type=int, default=50,
        help="Number of ADDITIONAL training graphs (generated separately). Default: 50"
    )
    parser.add_argument(
        "--n_graphs", type=int, default=None,
        help="DEPRECATED: Use --n_evaluation_graphs instead"
    )
    parser.add_argument(
        "--mode", type=str, default="finetune-linear",
        choices=["linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "gpf", "gpf-plus", "gpf-linear", "gpf-plus-linear", "untrained_frozen"],
        help="Evaluation mode: linear (frozen encoder + linear), mlp (frozen encoder + MLP), finetune-linear (unfrozen encoder + linear), finetune-mlp (unfrozen encoder + MLP), scratch (random init + MLP), untrained_frozen (random init frozen + MLP), gpf/gpf-plus (prompt + MLP), gpf-linear/gpf-plus-linear (prompt + linear)"
    )
    parser.add_argument(
        "--p_num", type=int, default=5,
        help="Number of basis vectors for GPF-Plus (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size in graphs (default: 32)"
    )
    parser.add_argument(
        "--patience", type=int, default=100,
        help="Early stopping patience (default: 50)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (default: cpu)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--wandb", action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="test_community_related_property_reconstruction",
        help="W&B project name (default: downstream_eval)"
    )

    # Add this new argument
    parser.add_argument(
        "--subsample_from_train", action="store_true",
        help="For non-GraphUniverse datasets: subsample n_train from the original training set"
    )
    
    parser.add_argument(
        "--graphuniverse_override", type=str, default=None,
        help="JSON string to override GraphUniverse generation parameters. "
             "Example: '{\"family_parameters\": {\"homophily_range\": [0.9, 1.0], \"max_communities\": 10}}'. "
             "If not specified, uses GRAPHUNIVERSE_OVERRIDE_DEFAULT from the script."
    )
    
    parser.add_argument(
        "--classifier_dropout", type=float, default=0.3,
        help="Dropout rate for classifier hidden layers (default: 0.3)"
    )
    
    parser.add_argument(
        "--input_dropout", type=float, default=None,
        help="Dropout rate applied to encoder output before classifier. "
             "If not specified, uses classifier_dropout for MLP, 0.0 for linear. "
             "Helps prevent overfitting on pre-trained features."
    )
    
    parser.add_argument(
        "--downstream_task", type=str, default="community_related_property_reconstruction",
        choices=["basic_property_reconstruction", "community_related_property_reconstruction"],
        help="Override the downstream task (for transfer learning). "
             "Example: pretrain on community_detection, finetune on triangle_counting. "
             "Options: community_detection (node-level), triangle_counting (graph-level), "
             "basic_property_reconstruction (graph-level multi-output regression). "
             "If not specified, uses the task from the pretraining config."
    )
    
    parser.add_argument(
        "--readout_type", type=str, default="sum",
        choices=["mean", "max", "sum"],
        help="Pooling type for graph-level tasks (default: mean)"
    )
    
    parser.add_argument(
        "--enable_joint_basic_property_reconstruction", action="store_true",
        help="For property reconstruction: use joint training with multi-head predictor (old behavior). "
             "Default: False (consecutive training with individual heads - NEW DEFAULT)"
    )
    
    parser.add_argument(
        "--basic_properties_to_include", type=str, nargs="+", default=None,
        choices=["avg_degree", "size", "gini", "diameter"],
        help="Which basic properties to train for basic_property_reconstruction. "
             "Default: None (trains all 4 properties)"
    )
    
    parser.add_argument(
        "--community_related_properties_to_include", type=str, nargs="+", default=None,
        choices=["homophily", "community_presence", "edge_prob_matrix", "community_detection"],
        help="Which community-related properties to train for community_related_property_reconstruction. "
             "Default: None (trains all 4 properties)"
    )
    
    args = parser.parse_args()
    
    # Determine graphuniverse_override: CLI arg takes precedence over default
    graphuniverse_override = None
    if args.graphuniverse_override:
        # CLI argument provided - parse JSON string
        try:
            graphuniverse_override = json.loads(args.graphuniverse_override)
            print(f"\n✓ Using GraphUniverse override from CLI argument")
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --graphuniverse_override: {e}")
            sys.exit(1)
    elif GRAPHUNIVERSE_OVERRIDE_DEFAULT is not None and len(GRAPHUNIVERSE_OVERRIDE_DEFAULT) > 0:
        # Use default from top of file
        graphuniverse_override = GRAPHUNIVERSE_OVERRIDE_DEFAULT
        print(f"\n✓ Using GraphUniverse override from GRAPHUNIVERSE_OVERRIDE_DEFAULT")
        print(f"  Override: {json.dumps(graphuniverse_override, indent=2)}")
    else:
        print(f"\n✓ No GraphUniverse override specified - using pretraining config")
    
    results = run_downstream_evaluation(
        run_dir=args.run_dir,
        checkpoint_path=args.checkpoint,
        n_evaluation_graphs=args.n_evaluation_graphs,
        n_train=args.n_train,
        subsample_from_train=args.subsample_from_train,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        n_graphs=args.n_graphs,  # Backward compatibility
        p_num=args.p_num,
        graphuniverse_override=graphuniverse_override,
        downstream_task=args.downstream_task,
        readout_type=args.readout_type,
        classifier_dropout=args.classifier_dropout,
        input_dropout=args.input_dropout,
        enable_joint_basic_property_reconstruction=args.enable_joint_basic_property_reconstruction,
        basic_properties_to_include=args.basic_properties_to_include,
        community_related_properties_to_include=args.community_related_properties_to_include,
    )
    
    # Determine task type from results
    task_type = results.get('task_type', 'classification')
    is_property_reconstruction = task_type == "property_reconstruction"
    is_community_property_reconstruction = task_type == "community_property_reconstruction"
    is_regression = task_type == "regression"
    
    print("\n" + "=" * 60)
    if is_property_reconstruction:
        print("FINAL RESULTS (PROPERTY RECONSTRUCTION)")
    elif is_community_property_reconstruction:
        print("FINAL RESULTS (COMMUNITY PROPERTY RECONSTRUCTION)")
    elif is_regression:
        print("FINAL RESULTS (GRAPH-LEVEL REGRESSION)")
    else:
        print("FINAL RESULTS (NODE-LEVEL CLASSIFICATION)")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    if results['mode'] == 'scratch':
        print("  (Random initialization baseline - no pre-trained weights)")
    elif results['mode'] == 'finetune':
        print("  (Fine-tuning pre-trained encoder)")
    else:
        print("  (Frozen pre-trained encoder)")
    
    if is_property_reconstruction:
        print(f"\nWeighted MAE: {results['test_mae_weighted']:.4f}")
        print("\nPer-property MAE:")
        for prop in ['avg_degree', 'size', 'gini', 'diameter']:
            print(f"  {prop:15s}: {results[f'test_mae_{prop}']:.4f}")
    elif is_community_property_reconstruction:
        print("\n✓ Results were printed during consecutive training")
        print("  (See above for homophily, community_presence, edge_prob_matrix)")
    elif is_regression:
        print(f"Test MAE: {results['test_mae']:.4f}")
        print(f"Test MSE: {results['test_mse']:.4f}")
        print(f"Test RMSE: {results['test_rmse']:.4f}")
    else:
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"Number of classes (K): {results['num_classes']}")
    
    print(f"\nTraining graphs: {results['n_train_actual']}")
    print(f"Validation graphs: {results['n_val']}")
    print(f"Test graphs: {results['n_test']}")
    
    # Print additional metrics
    if not is_property_reconstruction and not is_community_property_reconstruction:
        task_level_key = f"num_graphs" if is_regression else f"num_nodes"
        if task_level_key in results:
            if is_regression:
                print(f"Test graphs: {results['num_graphs']}")
            else:
                print(f"Test nodes: {results['num_nodes']}")
    
    if results.get('fixed_eval_set'):
        print(f"\nNote: Using FIXED evaluation set approach for fair comparison")
    
    return results


if __name__ == "__main__":
    main()


