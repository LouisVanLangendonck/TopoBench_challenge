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
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# TopoBench imports
from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import DataloadDataset
from graph_universe import GraphUniverseDataset


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
    universe_seed: int = 42,
    family_seed: int = 43,
    dataset_purpose: str = "eval",  # "eval" or "train"
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
    data_dir: str | None = None,
) -> tuple:
    """
    Create/load a GraphUniverse dataset.
    
    Parameters
    ----------
    config : dict, optional
        Configuration from wandb run.
    n_graphs : int, optional
        Number of graphs to generate
    universe_seed : int
        Seed for the universe parameters
    family_seed : int
        Seed for the family parameters
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

    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    # Store original for diff printing
    original_gen_params = deepcopy(gen_params)
    
    # Override task
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
    
    # Override seeds (Mainly for the family seed to have completely new graphs for downstream evaluation, but keeping general behaviour of the universe seed)
    gen_params["family_parameters"]["seed"] = family_seed
    gen_params["universe_parameters"]["seed"] = universe_seed
    print(f"{dataset_purpose} set: universe_seed={universe_seed}, family_seed={family_seed}")
    
    # Apply GraphUniverse overrides if specified
    has_overrides = graphuniverse_override is not None and len(graphuniverse_override) > 0
    if has_overrides:
        print("\n" + "=" * 80)
        print("APPLYING GRAPHUNIVERSE OVERRIDES (for out-of-distribution fine-tuning evaluation)")
        print("=" * 80)
        _deep_update(gen_params, graphuniverse_override)
    
    # Override data directory if specified
    root_dir = data_dir if data_dir else params["data_dir"]
    
    # Add suffix to distinguish from original dataset
    if n_graphs is not None:
        root_dir = f"{root_dir}_downstream_{n_graphs}graphs_universe_seed{universe_seed}_family_seed{family_seed}"
    
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

    data_dir = dataset.raw_dir
    
    return dataset, data_dir, {"type": "GraphUniverse", "subsample_info": None}
    

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
        One of: 'dgi', 'graphmae', 'graphmaev2', 'grace', 'dgmae', 'graphcl', 'linkpred', 's2gae', 'higmae', 'supervised', 'unknown'
    """
    model_config = config.get("model", {})
    wrapper_config = model_config.get("backbone_wrapper", {})
    
    if wrapper_config:
        wrapper_target = wrapper_config.get("_target_", "")
        if "DGI" in wrapper_target:
            return "dgi"
        elif "GraphMAEv2" in wrapper_target:
            return "graphmaev2"
        elif "GRACE" in wrapper_target or "grace" in wrapper_target.lower():
            return "grace"
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
    elif "GRACE" in loss_target or "grace" in loss_target.lower():
        return "grace"
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
    
    IMPORTANT: This explicitly re-initializes ALL weights to ensure true randomness.
    This is critical for fair comparison - we want to test if pretraining helps,
    not if PyTorch's default initialization happens to work well.
    
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
    import torch.nn.init as init
    
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
    
    # EXPLICITLY RE-INITIALIZE ALL WEIGHTS TO ENSURE RANDOMNESS
    # This is critical because:
    # 1. Some models might have special initialization in __init__
    # 2. We want to ensure truly random weights for fair baseline comparison
    # 3. With structural features (PSEs), even random projections can perform well
    
    print("\n" + "=" * 80)
    print("RANDOM INITIALIZATION VERIFICATION")
    print("=" * 80)
    
    # Store initial weight stats BEFORE re-initialization
    initial_stats = {}
    for name, param in backbone.named_parameters():
        if 'weight' in name:
            initial_stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
            }
    
    def reinitialize_weights(module):
        """Re-initialize all parameters with Xavier uniform (standard init)."""
        if hasattr(module, 'reset_parameters'):
            # Use module's own reset if available
            module.reset_parameters()
        else:
            # Otherwise, reinitialize manually
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    if 'weight' in name:
                        if param.ndim >= 2:
                            init.xavier_uniform_(param)
                        else:
                            init.uniform_(param, -0.1, 0.1)
                    elif 'bias' in name:
                        init.zeros_(param)
    
    # Apply re-initialization to all modules
    feature_encoder.apply(reinitialize_weights)
    backbone.apply(reinitialize_weights)
    
    # Verify weights changed after re-initialization
    print("\n📊 Weight Statistics (sample of first 3 backbone layers):")
    num_shown = 0
    for name, param in backbone.named_parameters():
        if 'weight' in name and num_shown < 3:
            new_mean = param.data.mean().item()
            new_std = param.data.std().item()
            old_mean = initial_stats.get(name, {}).get('mean', 0)
            old_std = initial_stats.get(name, {}).get('std', 0)
            changed = abs(new_mean - old_mean) > 1e-6 or abs(new_std - old_std) > 1e-6
            status = "✓ CHANGED" if changed else "✗ UNCHANGED"
            print(f"  {name[:50]:<50} → mean={new_mean:>7.4f}, std={new_std:>7.4f} {status}")
            num_shown += 1
    
    # Count total parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\n📈 Backbone parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Move to device
    feature_encoder = feature_encoder.to(device)
    backbone = backbone.to(device)
    
    # Wrap backbone in GNNWrapper for consistency with pretrained loading
    # This ensures the same interface for both scratch and pretrained encoders
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper
    
    # Get wrapper config parameters
    wrapper_config = model_config.get("backbone_wrapper", {})
    out_channels = wrapper_config.get("out_channels", hidden_dim)
    num_cell_dimensions = wrapper_config.get("num_cell_dimensions", 0)
    
    wrapped_backbone = GNNWrapper(
        backbone=backbone,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dimensions,
    ).to(device)
    
    print(f"\n✓ Created random encoder: {type(backbone).__name__}")
    print(f"  Wrapped in: GNNWrapper (for consistent interface)")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  ⚠️  CAUTION: Model receives PSE features (structural encodings)")
    print(f"      Even random encoders can perform well on these rich features!")
    print("=" * 80)
    
    return feature_encoder, wrapped_backbone, hidden_dim

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
    Load the pre-trained feature encoder and backbone for downstream tasks.
    
    STREAMLINED APPROACH (GraphMAEv2 and GRACE only):
    1. Fully instantiate the pretrained model (feature_encoder + backbone + wrapper)
    2. Extract the pretrained feature_encoder
    3. Extract the pretrained backbone (GNN encoder)
    4. Wrap the backbone in a clean GNNWrapper for downstream use
    
    This ensures we load the exact architecture and weights from pretraining,
    then use a clean wrapper without any pretraining-specific components.
    
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
        (feature_encoder, backbone_wrapped, hidden_dim)
        - feature_encoder: The pretrained feature encoding layer
        - backbone_wrapped: The pretrained backbone wrapped in GNNWrapper
        - hidden_dim: Output dimension of the backbone
    """
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper
    
    model_config = config["model"]
    pretraining_method = detect_pretraining_method(config)
    print(f"Detected pre-training method: {pretraining_method}")
    
    # Validate pretraining method
    if pretraining_method not in ["graphmaev2", "grace"]:
        raise ValueError(
            f"Unsupported pretraining method '{pretraining_method}'. "
            f"Only 'graphmaev2' and 'grace' are supported for downstream finetuning. "
            f"Please use one of these methods for pretraining."
        )
    
    print(f"\n🔧 STREAMLINED MODEL LOADING FOR DOWNSTREAM FINETUNING")
    print(f"   Method: {pretraining_method.upper()}")
    print(f"   Strategy: Full model instantiation → Extract components → Clean wrapper")
    
    # Step 1: Build the complete pretrained model architecture
    print(f"\n📦 Step 1: Instantiating full pretrained model...")
    
    # Build feature encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
        print(f"   ✓ Feature encoder: {type(feature_encoder).__name__}")
    else:
        print("   ⚠ No feature encoder found, using identity!")
        feature_encoder = nn.Identity()
    
    # Build backbone (the actual GNN encoder)
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    print(f"   ✓ Backbone (GNN): {type(backbone).__name__}")
    
    # Build the pretraining wrapper (we'll load weights then discard it)
    wrapper_config = model_config.get("backbone_wrapper", {})
    if wrapper_config:
        # Need to pass the backbone to the wrapper
        wrapper_config_copy = dict(wrapper_config)
        wrapper_config_copy.pop("_partial_", None)  # Remove _partial_ flag
        wrapper = instantiate_from_config(wrapper_config_copy, backbone=backbone)
        print(f"   ✓ Pretraining wrapper: {type(wrapper).__name__}")
    else:
        raise ValueError(f"No wrapper config found for {pretraining_method}")
    
    # Get hidden dimension
    hidden_dim = (
        backbone_config.get("hidden_dim") or 
        backbone_config.get("hidden_channels") or 
        backbone_config.get("out_channels") or
        model_config.get("feature_encoder", {}).get("out_channels") or
        128
    )
    print(f"   ✓ Hidden dimension: {hidden_dim}")
    
    # Step 2: Load checkpoint weights into the full model
    print(f"\n💾 Step 2: Loading checkpoint weights...")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Load feature encoder weights
    encoder_state = {k.replace("feature_encoder.", ""): v for k, v in state_dict.items() if k.startswith("feature_encoder.")}
    if encoder_state:
        feature_encoder.load_state_dict(encoder_state, strict=True)
        print(f"   ✓ Loaded feature encoder: {len(encoder_state)} parameters")
    
    # Load wrapper weights (which contains the backbone)
    wrapper_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    if wrapper_state:
        wrapper.load_state_dict(wrapper_state, strict=False)  # strict=False because wrapper has extra params
        print(f"   ✓ Loaded wrapper with backbone: {len(wrapper_state)} parameters")
    
    # Step 3: Extract the pretrained backbone from the wrapper
    print(f"\n🎯 Step 3: Extracting pretrained components...")
    pretrained_backbone = wrapper.backbone  # The actual GNN encoder with loaded weights
    print(f"   ✓ Extracted backbone: {type(pretrained_backbone).__name__}")
    
    # Step 4: Create clean GNNWrapper for downstream use
    print(f"\n🔄 Step 4: Creating clean GNNWrapper for downstream...")
    # Get required parameters from original wrapper config
    num_cell_dimensions = wrapper_config.get("num_cell_dimensions", 1)
    out_channels = wrapper_config.get("out_channels", hidden_dim)
    
    clean_wrapper = GNNWrapper(
        backbone=pretrained_backbone,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dimensions
    )
    print(f"   ✓ Clean wrapper created: GNNWrapper")
    print(f"   ✓ No pretraining-specific components (no masks, projectors, EMA, etc.)")
    
    # Move to device
    feature_encoder = feature_encoder.to(device)
    clean_wrapper = clean_wrapper.to(device)
    
    # Verification
    print(f"\n✅ Model loading complete!")
    print(f"   - Feature encoder: {type(feature_encoder).__name__} (pretrained)")
    print(f"   - Backbone: {type(pretrained_backbone).__name__} (pretrained, wrapped in GNNWrapper)")
    print(f"   - Hidden dimension: {hidden_dim}")
    print(f"   - Ready for downstream finetuning!")
    
    return feature_encoder, clean_wrapper, hidden_dim


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


def verify_gradient_flow(model: nn.Module, data_loader: DataLoader, device: str = "cpu", freeze_encoder: bool = True):
    """
    Verify that gradients are (not) flowing through the encoder as expected.
    
    This is CRITICAL for frozen encoder baselines - we must ensure:
    1. Encoder parameters have requires_grad=False
    2. No gradients actually flow through encoder during backprop
    3. Encoder stays in eval mode
    
    Parameters
    ----------
    model : nn.Module
        The complete model (encoder + classifier).
    data_loader : DataLoader
        Data loader to get a test batch.
    device : str
        Device to use.
    freeze_encoder : bool
        Whether encoder should be frozen.
    
    Returns
    -------
    dict
        Diagnostic information about gradient flow.
    """
    import torch
    
    # Ensure model is on correct device
    model = model.to(device)
    model.train()  # Put model in train mode (encoder should stay eval if frozen)
    
    # Get one batch
    try:
        batch = next(iter(data_loader))
        batch = batch.to(device)
        batch = prepare_batch_for_topobench(batch)
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': f'Failed to get batch: {str(e)}',
        }
    
    # Check requires_grad status BEFORE forward pass
    encoder = model.encoder if hasattr(model, 'encoder') else model
    encoder_params_status = []
    for name, param in encoder.named_parameters():
        encoder_params_status.append({
            'name': name,
            'requires_grad': param.requires_grad,
            'grad': param.grad is not None,
        })
    
    # Do a forward + backward pass
    try:
        # Forward
        output = model(batch)
        
        # Create a dummy loss (we just need something to backprop)
        if isinstance(output, dict):
            # Property reconstruction: sum all outputs
            loss = torch.tensor(0.0, device=device, requires_grad=True)
            for v in output.values():
                if isinstance(v, torch.Tensor) and v.requires_grad:
                    loss = loss + v.sum()
        else:
            loss = output.sum()
        
        # Only backward if loss requires grad
        if loss.requires_grad:
            loss.backward()
        else:
            # If no gradients, that's actually fine for frozen encoder
            pass
        
        # Check if gradients actually flowed through encoder
        encoder_grads = []
        for name, param in encoder.named_parameters():
            encoder_grads.append({
                'name': name,
                'has_grad': param.grad is not None,
                'grad_norm': param.grad.norm().item() if param.grad is not None else 0.0,
            })
        
        # Check encoder mode
        encoder_in_eval = not encoder.training
        
        # Clean up
        model.zero_grad()
        
        # Determine issues
        issues = []
        if freeze_encoder:
            # Encoder should be frozen
            if not encoder_in_eval:
                issues.append("Encoder not in eval mode (should be for frozen encoder)")
            
            params_with_grad = [p for p in encoder_params_status if p['requires_grad']]
            if params_with_grad:
                issues.append(f"{len(params_with_grad)} encoder params have requires_grad=True (should be False)")
            
            params_with_computed_grad = [g for g in encoder_grads if g['has_grad']]
            if params_with_computed_grad:
                issues.append(f"{len(params_with_computed_grad)} encoder params received gradients (should be 0)")
        
        status = "OK" if not issues else "FAILED"
        
        return {
            'status': status,
            'encoder_in_eval': encoder_in_eval,
            'encoder_params_total': len(encoder_params_status),
            'encoder_params_requiring_grad': sum(1 for p in encoder_params_status if p['requires_grad']),
            'encoder_params_with_grad': sum(1 for g in encoder_grads if g['has_grad']),
            'issues': issues,
            'sample_grads': encoder_grads[:3],  # Show first 3 for inspection
        }
        
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
        }


def verify_encoder_outputs(encoder: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    """
    Verify that the encoder produces meaningful outputs.
    
    Checks:
    1. Outputs are not all zeros
    2. Outputs have variance (not constant)
    3. Outputs have reasonable magnitude
    
    Also reports input feature statistics to diagnose if PSEs are providing structural information.
    
    Returns a dict with diagnostic info.
    """
    encoder.eval()
    encoder = encoder.to(device)
    
    # Get one batch
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = prepare_batch_for_topobench(batch)
    
    # Check INPUT features (to diagnose PSE effect)
    input_features = batch.x_0 if hasattr(batch, 'x_0') else batch.x
    input_mean = input_features.mean().item()
    input_std = input_features.std().item()
    input_dim = input_features.shape[1]
    
    with torch.no_grad():
        try:
            features = encoder(batch)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    # Compute OUTPUT statistics
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
        # Input diagnostics (to understand PSE effect)
        "input_mean": input_mean,
        "input_std": input_std,
        "input_dim": input_dim,
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
        
        # Backbone encoding - the backbone is wrapped in GNNWrapper which expects a batch
        # Update batch with the encoded features
        batch.x_0 = x
        
        # Call the GNNWrapper (or other wrapper) with the batch
        # It returns a dictionary with 'x_0' containing node-level features
        output = self.backbone(batch)
        
        # Extract node features from the output dictionary
        if isinstance(output, dict):
            node_features = output['x_0']
        else:
            # Fallback if backbone returns tensor directly (shouldn't happen with GNNWrapper)
            node_features = output
        
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
        readout_type: str = "mean",  # "mean", "max", "sum"
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
        
        # Update the batch with encoded features
        # The GNNWrapper expects a batch object, not individual arguments
        batch_encoded.x_0 = batch_encoded.x_0 if hasattr(batch_encoded, 'x_0') else (
            batch_encoded.get('x_0', batch_encoded.get('x')) if isinstance(batch_encoded, dict) else batch_encoded
        )
        
        # Backbone encoding - GNNWrapper expects a batch object and returns a dict
        model_out_from_wrapper = self.backbone(batch_encoded)
        
        # Extract node features from the wrapper's output dict
        node_features = model_out_from_wrapper["x_0"]
        
        # Get batch indices for pooling
        batch_indices = batch_encoded.batch_0 if hasattr(batch_encoded, 'batch_0') else batch_encoded.batch
        
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
    baseline_val_mae: float | None = None,  # DEPRECATED: Not used anymore
    baseline_test_mae: float | None = None,  # Fixed baseline (test mean on test)
    train_mean: float | None = None,  # Actually test_mean now (for reporting)
) -> tuple:
    """
    Train a single property prediction head and evaluate on test set.
    
    NOTE: baseline_test_mae is computed using TEST mean on TEST set for a fixed
    baseline that's independent of training set size (n_train).
    
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
        # For simple properties, print test_mean (used for fixed baseline)
        if train_mean is not None and isinstance(train_mean, (int, float)):
            print(f"  Test mean (baseline): {train_mean:.4f}")
        if baseline_test_mae is not None:
            print(f"  Baseline test MAE (fixed): {baseline_test_mae:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best val MAE: {best_val_metric:.4f}")
        print(f"  Test MAE: {test_mae:.4f}")
        if baseline_test_mae is not None:
            beat_baseline = "✓" if test_mae < baseline_test_mae else "✗"
            improvement = ((baseline_test_mae - test_mae) / baseline_test_mae * 100) if baseline_test_mae > 0 else 0
            print(f"  vs Baseline (fixed): {improvement:+.1f}% {beat_baseline}")
    print(f"{'='*80}\n")
    
    return best_val_loss, test_mae, history, model

def train_property_reconstruction(
    model: PropertyReconstructionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,  # Add test loader
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    use_wandb: bool,
    K: int,
    normalization_scales: dict,
    properties_to_include: list[str] | None = None,  # Which properties to train (None = all)
    weight_decay: float = 0.0,
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
    
    # First, compute baselines using TEST mean to predict TEST (FIXED baseline)
    # This ensures baseline is independent of training set size (n_train)
    # and comparable across different experiments
    print("\n" + "=" * 80)
    print("COMPUTING BASELINES (Predict TEST Mean on TEST Set)")
    print("=" * 80)
    print("Using TEST set mean as baseline for fair comparison across n_train values")
    print("=" * 80)
    
    test_targets_dict = {prop: [] for prop in properties}
    
    # Collect test data
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
    test_mean_values = {}
    baseline_test_mae = {}
    
    # Get classification properties list (if it exists)
    classification_props = classification_properties if is_community_model else []
    
    for prop in properties:
        # Skip classification properties (no MAE baseline)
        if prop in classification_props:
            print(f"{prop:<25}: CLASSIFICATION TASK - no MAE baseline (use accuracy from 0)")
            test_mean_values[prop] = None
            baseline_test_mae[prop] = None
            continue
        
        test_all = torch.cat(test_targets_dict[prop], dim=0)
        
        # For simple properties (scalars), compute mean as usual
        # For complex properties (vectors/matrices), compute element-wise mean
        if prop in simple_properties:
            test_mean = test_all.mean().item()
            test_std = test_all.std().item()
            test_mean_values[prop] = test_mean
            
            # Baseline: predict test mean on test set (approximately equal to test std)
            baseline_test_mae[prop] = torch.abs(test_all - test_mean).mean().item()
            
            print(f"{prop:<25}: test_mean={test_mean:>8.4f}, test_std={test_std:>8.4f}, baseline_test_mae={baseline_test_mae[prop]:>8.4f}")
        else:
            # For complex properties, compute element-wise statistics
            # test_mean will be a tensor (not a scalar)
            test_mean_tensor = test_all.mean(dim=0)  # Average across batch
            test_std = test_all.std().item()  # Overall std for reference
            test_mean_values[prop] = test_mean_tensor  # Store as tensor
            
            # Baseline: predict test mean (broadcasted) on test set
            baseline_test_mae[prop] = torch.abs(test_all - test_mean_tensor).mean().item()
            
            print(f"{prop:<25}: test_std={test_std:>8.4f}, baseline_test_mae={baseline_test_mae[prop]:>8.4f} [shape: {list(test_mean_tensor.shape)}]")
    
    print("=" * 80)
    
    # Log baselines to wandb (skip classification properties)
    if use_wandb and WANDB_AVAILABLE:
        for prop in properties:
            if prop not in classification_props:  # Only log for regression properties
                wandb.log({
                    f"{prop}/baseline_test_mean": test_mean_values[prop],
                    f"{prop}/baseline_test_mae": baseline_test_mae[prop],
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
            baseline_val_mae=None,  # No longer compute val baseline (not needed)
            baseline_test_mae=baseline_test_mae[prop],
            train_mean=test_mean_values[prop],  # Pass test mean (fixed baseline)
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
    
    # Store trained models, test means, and test results for later retrieval
    model._consecutive_models = trained_models
    model._test_mean_values = test_mean_values  # Changed from train to test mean for fixed baseline
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
    # Use TEST mean to predict TEST (FIXED baseline independent of n_train)
    print("\n" + "=" * 80)
    print("TEST SET BASELINE: Predict TEST Mean (FIXED)")
    print("=" * 80)
    
    # Check if we have test means stored from training (consecutive models)
    has_test_means = hasattr(model, '_test_mean_values')
    
    if has_test_means:
        # Use TEST means (correct fixed baseline)
        test_mean_values = model._test_mean_values
        print("✓ Using TEST means for baseline (fixed baseline approach)")
    else:
        # Compute from test data directly
        test_mean_values = {}
        for prop in all_predictions.keys():
            test_mean_values[prop] = all_targets[prop].numpy().mean()
        print("✓ Computing TEST means from test data (fixed baseline approach)")
    
    # Compute baseline MAE using test means to predict test (skip classification properties)
    baseline_test_mae = {}
    for prop in all_predictions.keys():
        # Skip classification properties (no MAE baseline)
        if prop in classification_properties:
            baseline_test_mae[prop] = None
            continue
        
        target = all_targets[prop].numpy()
        test_mean = test_mean_values[prop]
        
        # Handle complex properties (test_mean might be a tensor/array)
        if isinstance(test_mean, torch.Tensor):
            test_mean = test_mean.numpy()
        
        baseline_test_mae[prop] = np.abs(target - test_mean).mean()
    
    print(f"\n{'Property':<25} {'TEST Mean':<18} {'Baseline/Metric':<15} {'Model Metric':<15} {'Improvement':<15}")
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
            
            # Format test_mean display differently for simple vs complex properties
            if prop in simple_properties:
                test_mean_str = f"{test_mean_values[prop]:.4f}"
            else:
                test_mean_str = f"[shape: {list(test_mean_values[prop].shape) if isinstance(test_mean_values[prop], (torch.Tensor, np.ndarray)) else 'scalar'}]"
            
            print(f"{prop:<25} {test_mean_str:<18} {baseline_mae:<15.4f} {model_mae:<15.4f} {improvement:>13.1f}% {beat_baseline}")
    
    print("\n" + "=" * 80 + "\n")
    
    # Add baselines to results (skip classification properties)
    for prop in all_predictions.keys():
        if prop not in classification_properties:
            results[f"baseline_test_mae_{prop}"] = baseline_test_mae[prop]
            results[f"test_mean_{prop}"] = test_mean_values[prop]  # Store test mean used for fixed baseline
            results[f"improvement_{prop}"] = ((baseline_test_mae[prop] - results[f"test_mae_{prop}"]) / baseline_test_mae[prop] * 100) if baseline_test_mae[prop] > 0 else 0
        else:
            # For classification, store None for baseline-related metrics
            results[f"baseline_test_mae_{prop}"] = None
            results[f"test_mean_{prop}"] = None
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
                if test_mean_values[prop] is not None:
                    log_dict[f"test/test_mean_{prop}"] = test_mean_values[prop]  # Log test mean used for fixed baseline
        
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
    n_evaluation_graphs: int = 200,
    n_train: int = 20,
    mode: str = "linear",  # "linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "gpf", "gpf-plus", "gpf-linear", "gpf-plus-linear", "scratch_frozen"
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "downstream_eval",
    graphuniverse_override: dict | None = None,
    classifier_dropout: float = 0.5,  # Dropout rate for classifier
    input_dropout: float = None,  # Dropout on encoder output (None = use classifier_dropout)
    downstream_task: str | None = None,  # Override task for downstream eval (e.g., "triangle_counting")
    readout_type: str = "mean",  # Pooling type for graph-level tasks: "mean", "max", "sum"
    pretraining_config: dict | None = None,  # NEW: Pretraining config from wandb for logging
    basic_properties_to_include: list[str] | None = None,  # Which basic properties to train (default: all)
    community_related_properties_to_include: list[str] | None = None,  # Which community properties to train (default: all)
) -> dict:
    """
    Run full downstream evaluation pipeline for NODE-LEVEL classification.:
    
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
        - "scratch_frozen": random init frozen + MLP classifier
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
    run_dir = Path(run_dir)
    
    # Load config
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
    
    
    checkpoint_path = get_checkpoint_path_from_summary(run_dir)
    if checkpoint_path is None:
        raise ValueError("No checkpoint path found in wandb-summary.json in run directory. Please check if the run directory is correct.")
    
    # Get task type and determine number of classes / outputs
    loader_target = config["dataset"]["loader"]["_target_"]
    dataset_params = config["dataset"]["parameters"]
    
    # Check if downstream task is overridden
    if downstream_task is not None:
        
        if downstream_task == "basic_property_reconstruction":
            task_type = "property_reconstruction"
            loss_type = "weighted_mse_mae"
            num_classes = 4  # 4 properties (homophily moved to community task)
            actual_task_level = "graph"  # Property reconstruction is graph-level
            print(f"   Task: Basic Property Reconstruction (Graph-level Multi-output Regression)")
            print(f"   Properties: {basic_properties_to_include}")
        elif downstream_task == "community_related_property_reconstruction":
            task_type = "community_property_reconstruction"
            loss_type = "weighted_bce_mse"
            num_classes = 4  # 4 properties (3 regression + 1 classification)
            actual_task_level = "graph"  # Property reconstruction is graph-level
            print(f"   Task: Community-Related Property Reconstruction (Graph-level Multi-output)")
            print(f"   Properties: {community_related_properties_to_include}")
        else:
            raise ValueError(f"Unknown downstream_task: {downstream_task}")
        
    
        gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
        task_name = gen_params.get("task", "community_detection")
        universe_params = gen_params.get("universe_parameters", {})
        num_classes = universe_params.get("K", 10)
       
    # Create downstream dataset
    print("\n" + "=" * 60)
    print("Creating downstream evaluation dataset...")
    print("=" * 60)
    
    # Determine if GraphUniverse
    loader_target = config["dataset"]["loader"]["_target_"]
    is_graph_universe = "GraphUniverse" in loader_target
    
    # Get pretraining seeds
    gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
    family_params = gen_params.get("family_parameters", {})
    universe_params = gen_params.get("universe_parameters", {})
    pretraining_universe_seed = universe_params["seed"]
    pretraining_family_seed = family_params["seed"]

    # Add 1 to family seed for evaluation and 2 for training for family seeds (enforces completely NEW graphs for evaluation and training from pretraining)
    family_evaluation_seed = pretraining_family_seed + 1
    family_training_seed = pretraining_family_seed + 2
    
    # Generate EVAL set
    eval_dataset, eval_data_dir, eval_dataset_info = create_dataset_from_config(
        config,
        n_graphs=n_evaluation_graphs,
        universe_seed=pretraining_universe_seed,
        family_seed=family_evaluation_seed,
        dataset_purpose="eval",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )
    
    # Apply transforms to eval set
    transforms_config = config.get("transforms")
    eval_preprocessor = apply_transforms(eval_dataset, eval_data_dir, transforms_config)
    eval_data_list = eval_preprocessor.data_list
    
    print(f"Generated {len(eval_data_list)} EVAL graphs (family_seed={family_evaluation_seed})")
    
    # Generate TRAIN set (INDEPENDENT)
    train_dataset, train_data_dir, train_dataset_info = create_dataset_from_config(
        config,
        n_graphs=n_train,
        universe_seed=pretraining_universe_seed,
        family_seed=family_training_seed,
        dataset_purpose="train",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )
    
    # Apply transforms to train set
    train_preprocessor = apply_transforms(train_dataset, train_data_dir, transforms_config)
    train_data = train_preprocessor.data_list
    
    print(f"Generated {len(train_data)} TRAIN graphs (family_seed={family_training_seed})")
    
    # Split eval data into val/test (50/50)
    import random
    random.seed(pretraining_universe_seed)
    eval_indices = list(range(len(eval_data_list)))
    random.shuffle(eval_indices)
    n_val = len(eval_indices) // 2
    val_indices = eval_indices[:n_val]
    test_indices = eval_indices[n_val:]
    val_data = [eval_data_list[i] for i in val_indices]
    test_data = [eval_data_list[i] for i in test_indices]
    
    print(f"\nFinal split:")
    print(f"  Training: {len(train_data)} graphs")
    print(f"  Validation: {len(val_data)} graphs")
    print(f"  Test: {len(test_data)} graphs")
    print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} graphs")
    
    # Pre-compute properties for property reconstruction tasks
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


    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Encoder creation
    if mode == "scratch":
        print("Creating randomly initialized encoder (scratch baseline)...")
        print("=" * 60)
        feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
    elif mode == "scratch_frozen":
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

    # Verify encoder produces meaningful outputs
    print("\n" + "=" * 60)
    print("Verifying encoder outputs...")
    print("=" * 60)
    verify_result = verify_encoder_outputs(encoder, train_loader, device=device)
    print(f"  Status: {verify_result['status']}")
    print(f"  Output shape: {verify_result.get('shape', 'N/A')}")
    
    # Only format as float if the value is actually a number
    mean_val = verify_result.get('mean', 'N/A')
    std_val = verify_result.get('std', 'N/A')
    min_val = verify_result.get('min', 'N/A')
    max_val = verify_result.get('max', 'N/A')
    
    if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
        print(f"  Mean: {mean_val:.4f}, Std: {std_val:.4f}")
    else:
        print(f"  Mean: {mean_val}, Std: {std_val}")
    
    if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
        print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    else:
        print(f"  Range: [{min_val}, {max_val}]")
    
    # Show input feature statistics to diagnose PSE effect
    if 'input_dim' in verify_result:
        print(f"\n  📊 Input Features (before encoder):")
        print(f"     Dimension: {verify_result['input_dim']}")
        print(f"     Mean: {verify_result.get('input_mean', 0):.4f}, Std: {verify_result.get('input_std', 0):.4f}")
        if verify_result['input_dim'] > 50:
            print(f"     ⚠️  HIGH-DIMENSIONAL INPUT: Likely includes PSEs (structural encodings)")
            print(f"         PSEs contain graph structure info → even random encoders can perform well!")
        
    if verify_result.get('issues'):
        print(f"  ISSUES: {verify_result['issues']}")
    
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
    
    # Downstream model creation
    print("\n" + "=" * 60)
    print(f"Setting up downstream evaluation (mode: {mode})...")
    print("=" * 60)
    
    if task_type == "property_reconstruction":
        print(f"  Multi-property reconstruction: {num_classes} properties")
        print(f"  Properties: avg_degree, size, gini, diameter")
        
        # Extract K from config for complex properties
        from graph_properties import extract_K_from_config

        K = extract_K_from_config(config)
        print(f"  K (number of communities): {K}")
        
        # Determine if encoder should be frozen
        freeze_encoder = mode in ["linear", "mlp", "scratch_frozen"]
        use_mlp_heads = mode in ["mlp", "finetune-mlp"]
        
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
        K = extract_K_from_config(config)
        print(f"  K (number of communities): {K}")
        
        # Determine if encoder should be frozen
        freeze_encoder = mode in ["linear", "mlp", "scratch_frozen"]
        
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
        
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    # Set model to train mode before checking state
    downstream_model.train()
    
    # Get detailed model state info
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
        
    # Verify gradient flow if encoder should be frozen
    if freeze_encoder:
        print("\n" + "=" * 80)
        print("VERIFYING GRADIENT FLOW (CRITICAL FOR FROZEN ENCODER)")
        print("=" * 80)
        grad_check = verify_gradient_flow(downstream_model, train_loader, device=device, freeze_encoder=freeze_encoder)
        print(f"  Status: {grad_check['status']}")
        
        if grad_check['status'] == 'ERROR':
            print(f"  ⚠️  ERROR during gradient check: {grad_check.get('error', 'Unknown')}")
            print(f"      Skipping gradient verification (check manually)")
        else:
            print(f"  Encoder in eval mode: {grad_check.get('encoder_in_eval', 'N/A')}")
            print(f"  Encoder params total: {grad_check.get('encoder_params_total', 'N/A')}")
            print(f"  Encoder params with requires_grad=True: {grad_check.get('encoder_params_requiring_grad', 'N/A')}")
            print(f"  Encoder params that received gradients: {grad_check.get('encoder_params_with_grad', 'N/A')}")
            
            if grad_check['status'] == 'FAILED':
                print(f"\n  ❌ GRADIENT FLOW ISSUES DETECTED:")
                for issue in grad_check.get('issues', []):
                    print(f"     - {issue}")
                print(f"\n  ⚠️  THIS MEANS THE ENCODER IS NOT ACTUALLY FROZEN!")
                print(f"      Results will be invalid - encoder is learning during training!")
            elif grad_check['status'] == 'OK':
                print(f"\n  ✓ GRADIENT FLOW VERIFIED: Encoder is properly frozen")
        
        print("=" * 80)
    
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
            properties_to_include=basic_properties_to_include,
        )
    
    elif task_type == "community_property_reconstruction":
        # Get K for property computer
        gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
        universe_params = gen_params.get("universe_parameters", {})
        K = universe_params.get("K", 10)
        
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
            properties_to_include=community_related_properties_to_include,
        )
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
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
        raise ValueError(f"Unknown task type: {task_type}")
        
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
    else:
        raise ValueError(f"Unknown task type: {task_type}")
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
    results["encoder_frozen"] = freeze_encoder
    results["fixed_eval_set"] = is_graph_universe
    results["config"] = {
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "seed": seed,
        "hidden_dim": hidden_dim,
    }
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
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
        help="Path to wandb run directory corresponding to a pretrained model"
    )
    parser.add_argument(
        "--downstream_task", type=str, default="community_related_property_reconstruction",
        choices=["basic_property_reconstruction", "community_related_property_reconstruction"],
        help="Which family of downstream tasks for few-shot learning. "
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
    parser.add_argument(
        "--n_evaluation_graphs", type=int, default=200,
        help="Number of FIXED evaluation graphs (val+test). Default: 200"
    )
    parser.add_argument(
        "--n_train", type=int, default=50,
        help="Number of few-shot training graphs. Default: 50"
    )
    parser.add_argument(
        "--mode", type=str, default="finetune-linear",
        choices=["linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "scratch_frozen"],
        help="Evaluation mode: linear (frozen encoder + linear), mlp (frozen encoder + MLP), finetune-linear (unfrozen encoder + linear), finetune-mlp (unfrozen encoder + MLP), scratch (random init + MLP), scratch_frozen (random init frozen + MLP)"
    )
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Training epochs (default: 300)"
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
        "--patience", type=int, default=50,
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
        "--readout_type", type=str, default="mean",
        choices=["mean", "max", "sum"],
        help="Pooling type for graph-level tasks (default: mean)"
    )
    
    args = parser.parse_args()
    
    # Determine graphuniverse_override: CLI arg takes precedence over default
    graphuniverse_override = None
    if args.graphuniverse_override:
        # CLI argument provided - parse JSON string
        try:
            graphuniverse_override = json.loads(args.graphuniverse_override)
            print(f"\nUsing GraphUniverse override from CLI argument")
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --graphuniverse_override: {e}")
            sys.exit(1)
    elif GRAPHUNIVERSE_OVERRIDE_DEFAULT is not None and len(GRAPHUNIVERSE_OVERRIDE_DEFAULT) > 0:
        # Use default from top of file
        graphuniverse_override = GRAPHUNIVERSE_OVERRIDE_DEFAULT
        print(f"\nUsing GraphUniverse override from GRAPHUNIVERSE_OVERRIDE_DEFAULT")
        print(f"  Override: {json.dumps(graphuniverse_override, indent=2)}")
    else:
        print(f"\nNo GraphUniverse override specified - using pretraining config")
    
    results = run_downstream_evaluation(
        run_dir=args.run_dir,
        n_evaluation_graphs=args.n_evaluation_graphs,
        n_train=args.n_train,
        mode=args.mode,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        device=args.device,
        seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        graphuniverse_override=graphuniverse_override,
        downstream_task=args.downstream_task,
        readout_type=args.readout_type,
        classifier_dropout=args.classifier_dropout,
        input_dropout=args.input_dropout,
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
    if results['mode'] in ['scratch', 'scratch_frozen']:
        print("  (Random initialization baseline - no pre-trained weights)")
    elif results['mode'] == 'finetune':
        print("  (Full finetuning of unfrozen, pre-trained encoder and fresh classifier)")
    else:
        print("  (Linear Probing of frozen pre-trained encoder using fresh classifier)")
    
    if is_property_reconstruction:
        print(f"\nWeighted MAE: {results['test_mae_weighted']:.4f}")
        print("\nPer-property MAE:")
        for prop in basic_properties_to_include:
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


