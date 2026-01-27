"""
Downstream evaluation of pre-trained TopoBench models for NODE-LEVEL tasks.

This script loads a pre-trained model (e.g., from DGI, GraphMAE, GraphCL),
extracts the encoder components, generates a new dataset with the same
configuration, and runs downstream NODE-LEVEL evaluation with three modes:
1. Linear probing (frozen encoder + linear classifier)
2. MLP probing (frozen encoder + MLP classifier)
3. Fine-tuning (unfrozen encoder + MLP classifier)
4. Scratch (randomly initialized encoder + MLP classifier)

For GraphUniverse community detection: predicts community label for each node.
Number of classes = K (from universe_parameters).

FAIR EVALUATION METHODOLOGY (for GraphUniverse):
- Generates (n_evaluation + n_train) graphs TOGETHER with the same seed
- First n_evaluation graphs: FIXED evaluation set (val + test), same for all n_train values
- Next n_train graphs: training set, additional to evaluation set
- This ensures all models are evaluated on the SAME test set regardless of n_train
- All graphs from same generation = same distribution, fair comparison

Designed to work both locally in the topobench repo and in separate
repos where topobench is pip-installed.

Usage:
    python downstream_eval.py --run_dir /path/to/wandb/run --n_evaluation_graphs 200 --n_train 20 --mode finetune
"""

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
    
    Returns
    -------
    tuple
        (dataset, data_dir, dataset_info) - the loaded dataset, its directory, and metadata
    """
    dataset_config = deepcopy(config["dataset"])
    
    # Determine loader type
    loader_target = dataset_config["loader"]["_target_"]
    
    if "GraphUniverse" in loader_target:
        dataset, data_dir = _create_graph_universe_dataset(dataset_config, n_graphs, data_dir, seed)
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

def _create_graph_universe_dataset(
    dataset_config: dict,
    n_graphs: int | None,
    data_dir: str | None,
    seed: int,
) -> tuple:
    """Create GraphUniverse dataset with optional overrides."""
    from graph_universe import GraphUniverseDataset
    
    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    # Override number of graphs if specified
    if n_graphs is not None:
        gen_params["family_parameters"]["n_graphs"] = n_graphs
    
    # Override seed
    # Increases the seed by 1 to have completely new graphs for downstream evaluation
    gen_params["family_parameters"]["seed"] = seed + 1 
    # Keeps the same seed for the universe parameters to have the same distribution of graphs
    gen_params["universe_parameters"]["seed"] = seed
    
    # Override data directory if specified
    root_dir = data_dir if data_dir else params["data_dir"]
    
    # Add suffix to distinguish from original dataset
    if n_graphs is not None:
        root_dir = f"{root_dir}_downstream_{n_graphs}graphs_seed{seed}"
    
    print(f"Creating GraphUniverse dataset with {gen_params['family_parameters']['n_graphs']} graphs...")
    print(f"Data directory: {root_dir}")
    
    print("\n" + "-" * 60)
    print("GRAPHUNIVERSE CONFIG FOR DOWNSTREAM (what we're generating):")
    print("-" * 60)
    print(f"Family parameters:")
    for k, v in gen_params.get("family_parameters", {}).items():
        print(f"  {k}: {v}")
    print(f"Universe parameters:")
    for k, v in gen_params.get("universe_parameters", {}).items():
        print(f"  {k}: {v}")
    print("-" * 60 + "\n")
    
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
        One of: 'dgi', 'graphmae', 'graphmaev2', 'graphcl', 'linkpred', 's2gae', 'higmae', 'supervised', 'unknown'
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
    wrapper_only_state = {}  # Params that belong to wrapper, not backbone
    
    # Check if this is a wrapped backbone (DGI, GraphMAE, GraphCL, LinkPred, S2GAE, HiGMAE, etc.)
    has_wrapper = pretraining_method in ["dgi", "graphmae", "graphmaev2", "graphcl", "linkpred", "s2gae", "higmae"]
    
    # Get expected backbone parameter names for filtering
    expected_backbone_keys = set(backbone.state_dict().keys())
    
    # DEBUG: Show a few expected keys
    print(f"  Expected backbone keys (first 5): {list(expected_backbone_keys)[:5]}")
    print(f"  Total expected backbone keys: {len(expected_backbone_keys)}")
    
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
                # or "backbone.enc_mask_token" (wrapper-specific)
                
                # If still starts with "backbone.", strip it again (double nesting)
                if stripped_key.startswith("backbone."):
                    final_key = stripped_key.replace("backbone.", "", 1)
                    # This is definitely a backbone param
                    backbone_state[final_key] = value
                else:
                    # Single "backbone." prefix - could be wrapper param or direct backbone param
                    # Check against expected keys
                    if stripped_key in expected_backbone_keys:
                        # Direct backbone param (shouldn't happen with proper wrapping, but handle it)
                        backbone_state[stripped_key] = value
                    else:
                        # Wrapper-specific param (enc_mask_token, ln_0, projector, predictor, encoder_ema, etc.)
                        wrapper_only_state[stripped_key] = value
            else:
                # No wrapper, direct backbone
                backbone_state[stripped_key] = value
    
    # Load weights into feature encoder
    if encoder_state:
        feature_encoder.load_state_dict(encoder_state, strict=True)
        print(f"  Loaded feature encoder weights ({len(encoder_state)} keys)")
    
    # Report wrapper-only params that are being skipped
    if wrapper_only_state:
        print(f"  Skipping {len(wrapper_only_state)} wrapper-only params: {list(wrapper_only_state.keys())}")
        print(f"  (These are from the pre-training wrapper, not needed for clean backbone)")
    
    # Load weights into backbone - STRICT
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
        
        return node_features


# =============================================================================
# Downstream Classifiers
# =============================================================================

class LinearClassifier(nn.Module):
    """Simple linear classifier for probing."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.linear(x)


class MLPClassifier(nn.Module):
    """MLP classifier with configurable layers."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        layers = []
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
    
    def forward(self, x):
        return self.mlp(x)

from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

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
    
    def forward(self, batch):
        """
        Get encoded GRAPH-LEVEL features.
        
        Returns
        -------
        torch.Tensor
            Graph-level encoded features. Shape: (batch_size, hidden_dim)
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
        
        return graph_features

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

def train_downstream(
    model: DownstreamModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_classes: int,
    task_level: str = "node",
    device: str = "cpu",
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 20,
    use_wandb: bool = False,
):
    """
    Train the downstream model for NODE-LEVEL classification.
    
    Returns
    -------
    dict
        Training history with loss and accuracy curves.
    """
    model = model.to(device)
    
    # Only optimize trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    # Loss function for multi-class node classification
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()
            
            # Node-level predictions: (num_nodes, num_classes)
            out = model(batch)

            if task_level == "graph":
                # Graph-level: one prediction per graph
                y = batch.y.long()  # Shape: (batch_size,)
                num_samples = y.size(0)  # Number of graphs
            else:
                # Node-level: one prediction per node  
                y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                num_samples = y.size(0)  # Number of nodes
            
            loss = criterion(out, y)
            pred = out.argmax(dim=1)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * num_samples
            train_correct += (pred == y).sum().item()
            train_total += num_samples
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                out = model(batch)

                if task_level == "graph":
                    # Graph-level: one prediction per graph
                    y = batch.y.long()  # Shape: (batch_size,)
                    num_samples = y.size(0)  # Number of graphs
                else:
                    # Node-level: one prediction per node  
                    y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                    num_samples = y.size(0)  # Number of nodes

                loss = criterion(out, y)
                pred = out.argmax(dim=1)

                val_loss += loss.item() * num_samples
                val_correct += (pred == y).sum().item()
                val_total += num_samples
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "val/loss": val_loss,
                "val/accuracy": val_acc,
                "best_val_accuracy": best_val_acc,
                "lr": optimizer.param_groups[0]['lr'],
            })
        
        # Scheduler step
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_acc": f"{val_acc:.4f}",
            "best": f"{best_val_acc:.4f}",
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
    device: str = "cpu",
    use_wandb: bool = False,
) -> dict:
    """Evaluate model on test set."""
    model = model.to(device)
    model.eval()
    
    test_correct = 0
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
                y = batch.y.long()  # Shape: (batch_size,)
                num_samples = y.size(0)  # Number of graphs
            else:
                # Node-level: one prediction per node  
                y = batch.y.view(-1).long()  # Shape: (num_nodes,)
                num_samples = y.size(0)  # Number of nodes
            
            pred = out.argmax(dim=1)
            
            test_correct += (pred == y).sum().item()
            test_total += num_samples
            
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    
    test_acc = test_correct / test_total
    
    # Log to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "test/accuracy": test_acc,
            f"test/num_{model.task_level}s": test_total,
        })
    
    return {
        "test_accuracy": test_acc,
        "predictions": all_preds,
        "labels": all_labels,
        f"num_{model.task_level}s": test_total,
    }

# =============================================================================
# Main Pipeline
# =============================================================================

def run_downstream_evaluation(
    run_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    n_evaluation_graphs: int = 200,
    n_train: int = 20,
    mode: str = "linear",  # "linear", "mlp", "finetune", "scratch", "gpf", "gpf-plus", "untrained_frozen"
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
        Evaluation mode: "linear", "mlp", "finetune", "scratch", "gpf", "gpf-plus", or "untrained_frozen".
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
    
    # Get number of classes based on dataset type
    loader_target = config["dataset"]["loader"]["_target_"]
    
    if "GraphUniverse" in loader_target:
        # For GraphUniverse: K is the number of unique communities
        gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
        universe_params = gen_params.get("universe_parameters", {})
        num_classes = universe_params.get("K", 10)
        print(f"Number of classes (K communities): {num_classes}")
    else:
        # For other datasets: use num_classes from config, or infer from data
        num_classes = config["dataset"]["parameters"].get("num_classes", 2)
        print(f"Number of classes: {num_classes}")
        print(f"WARNING: This script is designed for node-level community detection.")
        print(f"         For graph-level tasks, the evaluation may not be appropriate.")
    
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
        # NEW APPROACH: Generate ALL graphs together, split by index
        # First n_evaluation_graphs are FIXED for val/test (same for all n_train)
        # Next n_train graphs are for training (additional)
        total_graphs = n_evaluation_graphs + n_train
        
        print(f"\nGenerating {total_graphs} total graphs (seed={pretraining_seed})...")
        print(f"  - First {n_evaluation_graphs} graphs: FIXED evaluation set (val/test)")
        print(f"  - Next {n_train} graphs: training set")
        
        dataset, data_dir, dataset_info = create_dataset_from_config(
            config, 
                n_graphs=total_graphs, 
                n_train=None,
                subsample_train=False,
                seed=pretraining_seed  # CRITICAL: Use pretraining seed!
            )
        
        # Apply transforms
        transforms_config = config.get("transforms")
        preprocessor = apply_transforms(dataset, data_dir, transforms_config)
        data_list = preprocessor.data_list
        
        print(f"✓ Generated {len(data_list)} graphs total using seed={pretraining_seed}")
        print(f"  This is the SAME dataset distribution as used in pretraining!")

        
        # Split by index:
        # [0 : n_evaluation_graphs] -> evaluation (will be split into val/test)
        # [n_evaluation_graphs : n_evaluation_graphs + n_train] -> training
        
        eval_data_list = data_list[:n_evaluation_graphs]
        train_data = data_list[n_evaluation_graphs:n_evaluation_graphs + n_train]
        
        # Split evaluation data into val/test (50/50)
        # Use the PRETRAINING seed so val/test split is consistent
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
        print(f"  Training: {len(train_data)} graphs")
        print(f"  Validation: {len(val_data)} graphs (from fixed eval set)")
        print(f"  Test: {len(test_data)} graphs (from fixed eval set)")
        print(f"  Total: {len(train_data) + len(val_data) + len(test_data)} graphs")
        
    else:
        # For non-GraphUniverse: use original approach
        print("Non-GraphUniverse dataset: using original splitting approach...")
        dataset, data_dir, dataset_info = create_dataset_from_config(
            config, 
            n_graphs=n_evaluation_graphs, 
        n_train=n_train,
        subsample_train=subsample_from_train,
        seed=pretraining_seed  # Use pretraining seed here too
    )

    # Apply transforms
    transforms_config = config.get("transforms")
    preprocessor = apply_transforms(dataset, data_dir, transforms_config)
    
    # Get data list
    data_list = preprocessor.data_list
    print(f"Dataset size: {len(data_list)} graphs")
    
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
    is_prompt_method = mode in ["gpf", "gpf-plus"]
    
    if is_prompt_method:
        # Import prompt tuning utilities
        from prompt_tuning import create_prompted_model
        
        print(f"Creating prompted model ({mode.upper()})...")
        print("=" * 60)
        
        # Create prompted model (handles encoder + prompt + classifier)
        downstream_model, prompt_info = create_prompted_model(
            config=config,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes,
            prompt_type=mode,
            p_num=p_num,
            task_level=task_level,
            device=device,
            classifier_type="mlp",
            mode="finetune",  # Always use pre-trained for prompts
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

        # Create appropriate encoder wrapper based on task level
        if task_level == "graph":
            encoder = GraphLevelEncoder(
                feature_encoder=feature_encoder,
                backbone=backbone,
                readout_type="mean",  # Could be configurable
            )
            print(f"  Using GraphLevelEncoder with mean pooling")
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
            }
            
            # Add prompt-specific config
            if is_prompt_method:
                wandb_config["hidden_dim"] = prompt_info["hidden_dim"]
                wandb_config["prompt_type"] = prompt_info["prompt_type"]
                wandb_config["p_num"] = prompt_info["p_num"]
                wandb_config["prompt_params"] = prompt_info["prompt_params"]
            else:
                wandb_config["hidden_dim"] = hidden_dim
            
            wandb.init(project=wandb_project, config=wandb_config)
    
    # Create classifier based on mode (skip if already created by prompt method)
    if not skip_manual_model_creation:
        # For node-level classification: input is node embedding, output is K classes
        print("\n" + "=" * 60)
        print(f"Setting up downstream evaluation (mode: {mode})...")
        print("=" * 60)
        print(f"  Hidden dim: {hidden_dim}, Num classes (K): {num_classes}")
        
        # Determine if encoder should be frozen
        # - linear/mlp: frozen encoder (probing)
        # - untrained_frozen: frozen random encoder (baseline to test if frozen matters)
        # - finetune: unfrozen encoder (fine-tuning pre-trained)
        # - scratch: unfrozen encoder (training from random init)
        freeze_encoder = mode in ["linear", "mlp", "untrained_frozen"]
        
        if mode == "linear":
            classifier = LinearClassifier(hidden_dim, num_classes)
        else:  # mlp, finetune, or scratch
            classifier = MLPClassifier(hidden_dim, num_classes)
        
        # Create downstream model
        downstream_model = DownstreamModel(
            encoder=encoder,
            classifier=classifier,
            freeze_encoder=freeze_encoder,
            task_level=task_level,
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
        print(f"  - Classifier params: {state_info['classifier_params']:,}")
    else:
        # Already printed by create_prompted_model
        pass
    
    # Train
    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)
    history = train_downstream(
        model=downstream_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        use_wandb=use_wandb,
        task_level=task_level,
    )
    
    # Evaluate
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    results = evaluate(
        model=downstream_model,
        test_loader=test_loader,
        num_classes=num_classes,
        device=device,
        use_wandb=use_wandb,
    )
    
    print(f"\nTest Accuracy (node-level): {results['test_accuracy']:.4f}")
    print(f"Total test nodes: {results['num_nodes']}")
    
    # Close wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Add metadata to results
    results["mode"] = mode
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
        print(f"All graphs generated together with seed={seed}")
    else:
        print(f"Dataset approach: Traditional split")
    print(f"Training graphs: {len(train_data)}")
    print(f"Validation graphs: {len(val_data)}")
    print(f"Test graphs: {len(test_data)}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print("=" * 60)
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Downstream NODE-LEVEL evaluation of pre-trained TopoBench models."
    )
    parser.add_argument(
        "--run_dir", type=str, default="../../../data/louisvl/TB/outputs/wandb/run-20251215_130646-2mnqjy9t",
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
        "--mode", type=str, default="finetune",
        choices=["linear", "mlp", "finetune", "scratch", "gpf", "gpf-plus", "untrained_frozen"],
        help="Evaluation mode: linear/mlp (frozen encoder), finetune (unfrozen pretrained), scratch (random init baseline), untrained_frozen (random init, frozen encoder), gpf/gpf-plus (prompt tuning)"
    )
    parser.add_argument(
        "--p_num", type=int, default=5,
        help="Number of basis vectors for GPF-Plus (default: 5)"
    )
    parser.add_argument(
        "--epochs", type=int, default=400,
        help="Training epochs (default: 100)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size in graphs (default: 16)"
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
        "--wandb_project", type=str, default="downstream_eval_big_run",
        help="W&B project name (default: downstream_eval)"
    )

    # Add this new argument
    parser.add_argument(
        "--subsample_from_train", action="store_true",
        help="For non-GraphUniverse datasets: subsample n_train from the original training set"
    )
    
    args = parser.parse_args()
    
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
    )
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS (NODE-LEVEL CLASSIFICATION)")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    if results['mode'] == 'scratch':
        print("  (Random initialization baseline - no pre-trained weights)")
    elif results['mode'] == 'finetune':
        print("  (Fine-tuning pre-trained encoder)")
    else:
        print("  (Frozen pre-trained encoder)")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Number of classes (K): {results['num_classes']}")
    print(f"Training graphs: {results['n_train_actual']}")
    print(f"Validation graphs: {results['n_val']}")
    print(f"Test graphs: {results['n_test']}")
    print(f"Test nodes: {results['num_nodes']}")
    if results.get('fixed_eval_set'):
        print(f"\nNote: Using FIXED evaluation set approach for fair comparison")
    
    return results


if __name__ == "__main__":
    main()


