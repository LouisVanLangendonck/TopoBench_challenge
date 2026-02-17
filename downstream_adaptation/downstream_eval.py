"""Downstream evaluation for INDUCTIVE learning setting.

This module handles evaluation on multiple graphs where entire graphs are split
into train/val/test sets. For TRANSDUCTIVE evaluation (single graph with node
masks), use downstream_eval_transductive.py instead.

Key differences:
- Inductive: Multiple graphs, graph-level splits, supports graph-level tasks
- Transductive: Single graph, node-level masks, node-level tasks only
"""

# =============================================================================
# GraphUniverse Override Configuration
# =============================================================================
GRAPHUNIVERSE_OVERRIDE_DEFAULT = None

import argparse
import json
import sys
from pathlib import Path
from copy import deepcopy

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

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from topobench.data.preprocessor import PreProcessor
from topobench.dataloader import DataloadDataset
from graph_universe import GraphUniverseDataset


# =============================================================================
# Configuration Loading
# =============================================================================

def load_wandb_config(run_dir: str | Path) -> dict:
    """Load config from wandb run directory."""
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
    dataset_purpose: str = "eval",
    graphuniverse_override: dict | None = None,
    downstream_task: str | None = None,
    data_dir: str | None = None,
) -> tuple:
    """Create/load GraphUniverse dataset with optional overrides."""
    dataset_config = deepcopy(config["dataset"])
    loader_target = dataset_config["loader"]["_target_"]
    params = dataset_config["loader"]["parameters"]
    gen_params = deepcopy(params["generation_parameters"])
    
    # Override task if specified
    if downstream_task is not None:
        if downstream_task in ["basic_property_reconstruction", "community_related_property_reconstruction"]:
            gen_params["task"] = "community_detection"
        else:
            gen_params["task"] = downstream_task
    
    # Override number of graphs and seeds
    if n_graphs is not None:
        gen_params["family_parameters"]["n_graphs"] = n_graphs
    gen_params["family_parameters"]["seed"] = family_seed
    gen_params["universe_parameters"]["seed"] = universe_seed
    
    # Apply GraphUniverse overrides
    if graphuniverse_override is not None and len(graphuniverse_override) > 0:
        _deep_update(gen_params, graphuniverse_override)
    
    # Set data directory
    root_dir = data_dir if data_dir else params["data_dir"]
    if n_graphs is not None:
        root_dir = f"{root_dir}_downstream_{n_graphs}graphs_universe_seed{universe_seed}_family_seed{family_seed}"
    if downstream_task is not None:
        root_dir = f"{root_dir}_task_{downstream_task}"
    if graphuniverse_override is not None and len(graphuniverse_override) > 0:
        import hashlib
        override_hash = hashlib.md5(json.dumps(graphuniverse_override, sort_keys=True).encode()).hexdigest()[:8]
        root_dir = f"{root_dir}_override_{override_hash}"
    
    dataset = GraphUniverseDataset(root=root_dir, parameters=gen_params)
    return dataset, dataset.raw_dir, {"type": "GraphUniverse", "subsample_info": None}


def _deep_update(base_dict: dict, update_dict: dict) -> dict:
    """Recursively update base_dict with values from update_dict."""
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def apply_transforms(
    dataset,
    data_dir: str,
    transforms_config: dict | None,
) -> PreProcessor:
    """Apply transforms from config."""
    if transforms_config:
        transforms_omega = OmegaConf.create(transforms_config)
    else:
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
    """Detect pre-training method from config."""
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
    """Create encoder with random initialization."""
    import torch.nn.init as init
    
    model_config = config["model"]
    
    # Build feature encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
    else:
        feature_encoder = nn.Identity()
    
    # Build backbone
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    
    # Get hidden dimension
    hidden_dim = (
        backbone_config.get("hidden_dim") or 
        backbone_config.get("hidden_channels") or 
        backbone_config.get("out_channels") or
        model_config.get("feature_encoder", {}).get("out_channels") or
        64
    )
    
    # Re-initialize all weights
    def reinitialize_weights(module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
        else:
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    if 'weight' in name:
                        if param.ndim >= 2:
                            init.xavier_uniform_(param)
                        else:
                            init.uniform_(param, -0.1, 0.1)
                    elif 'bias' in name:
                        init.zeros_(param)
    
    feature_encoder.apply(reinitialize_weights)
    backbone.apply(reinitialize_weights)
    
    # Move to device
    feature_encoder = feature_encoder.to(device)
    backbone = backbone.to(device)
    
    # Wrap backbone in GNNWrapper
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper
    
    wrapper_config = model_config.get("backbone_wrapper", {})
    out_channels = wrapper_config.get("out_channels", hidden_dim)
    num_cell_dimensions = wrapper_config.get("num_cell_dimensions", 0)
    
    wrapped_backbone = GNNWrapper(
        backbone=backbone,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dimensions,
    ).to(device)
    
    return feature_encoder, wrapped_backbone, hidden_dim


def detect_task_level(config: dict) -> str:
    """Detect task level from config."""
    dataset_params = config.get("dataset", {}).get("parameters", {})
    task_level = dataset_params.get("task_level", "node")
    return task_level


def load_pretrained_encoder(
    config: dict,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, nn.Module, int]:
    """Load pre-trained encoder from checkpoint."""
    from topobench.nn.wrappers.graph.gnn_wrapper import GNNWrapper
    
    model_config = config["model"]
    pretraining_method = detect_pretraining_method(config)
    
    if pretraining_method not in ["graphmaev2", "grace", "linkpred", "dgi"]:
        raise ValueError(
            f"Unsupported pretraining method '{pretraining_method}'. "
            f"Only 'graphmaev2', 'grace', 'linkpred', and 'dgi' are supported."
        )
    
    # Build feature encoder
    feature_encoder_config = model_config.get("feature_encoder")
    if feature_encoder_config:
        feature_encoder = instantiate_from_config(feature_encoder_config)
    else:
        feature_encoder = nn.Identity()
    
    # Build backbone
    backbone_config = model_config["backbone"]
    backbone = instantiate_from_config(backbone_config)
    
    # Build wrapper
    wrapper_config = model_config.get("backbone_wrapper", {})
    if wrapper_config:
        wrapper_config_copy = dict(wrapper_config)
        wrapper_config_copy.pop("_partial_", None)
        wrapper = instantiate_from_config(wrapper_config_copy, backbone=backbone)
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
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    
    # Load feature encoder weights
    encoder_state = {k.replace("feature_encoder.", ""): v for k, v in state_dict.items() if k.startswith("feature_encoder.")}
    if encoder_state:
        feature_encoder.load_state_dict(encoder_state, strict=True)
    
    # Load wrapper weights
    wrapper_state = {k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")}
    if wrapper_state:
        wrapper.load_state_dict(wrapper_state, strict=False)
    
    # Extract backbone
    pretrained_backbone = wrapper.backbone
    
    # Create clean GNNWrapper
    num_cell_dimensions = wrapper_config.get("num_cell_dimensions", 1)
    out_channels = wrapper_config.get("out_channels", hidden_dim)
    
    clean_wrapper = GNNWrapper(
        backbone=pretrained_backbone,
        out_channels=out_channels,
        num_cell_dimensions=num_cell_dimensions
    )
    
    # Move to device
    feature_encoder = feature_encoder.to(device)
    clean_wrapper = clean_wrapper.to(device)
    
    return feature_encoder, clean_wrapper, hidden_dim


def prepare_batch_for_topobench(batch):
    """Ensure batch has required TopoBench attributes."""
    if not hasattr(batch, 'x_0') and hasattr(batch, 'x'):
        batch.x_0 = batch.x
    if not hasattr(batch, 'batch_0') and hasattr(batch, 'batch'):
        batch.batch_0 = batch.batch
    return batch


def verify_encoder_outputs(encoder: nn.Module, data_loader: DataLoader, device: str = "cpu"):
    """Verify encoder produces meaningful outputs."""
    encoder.eval()
    encoder = encoder.to(device)
    
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = prepare_batch_for_topobench(batch)
    
    input_features = batch.x_0 if hasattr(batch, 'x_0') else batch.x
    input_mean = input_features.mean().item()
    input_std = input_features.std().item()
    input_dim = input_features.shape[1]
    
    with torch.no_grad():
        try:
            features = encoder(batch)
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}
    
    mean = features.mean().item()
    std = features.std().item()
    min_val = features.min().item()
    max_val = features.max().item()
    num_zeros = (features == 0).sum().item()
    total_elements = features.numel()
    
    issues = []
    if std < 1e-6:
        issues.append("Very low variance")
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
        "input_mean": input_mean,
        "input_std": input_std,
        "input_dim": input_dim,
    }


class PretrainedEncoder(nn.Module):
    """Wrapper combining feature encoder + backbone for node-level features."""
    
    def __init__(self, feature_encoder: nn.Module, backbone: nn.Module):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
    
    def forward(self, batch):
        """Get encoded node-level features."""
        batch = prepare_batch_for_topobench(batch)
        
        batch_encoded = self.feature_encoder(batch)
        
        if hasattr(batch_encoded, 'x_0'):
            x = batch_encoded.x_0
        elif isinstance(batch_encoded, dict):
            x = batch_encoded.get('x_0', batch_encoded.get('x'))
        else:
            x = batch_encoded
        
        batch.x_0 = x
        output = self.backbone(batch)
        
        if isinstance(output, dict):
            node_features = output['x_0']
        else:
            node_features = output
        
        return node_features


# =============================================================================
# Downstream Classifiers
# =============================================================================

class LinearClassifier(nn.Module):
    """Linear classifier for probing."""
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class MLPClassifier(nn.Module):
    """MLP classifier with configurable layers."""
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] = None,
        dropout: float = 0.5,
        input_dropout: float = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        if input_dropout is None:
            input_dropout = dropout
        
        layers = []
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
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)


class LinearRegressor(nn.Module):
    """Linear regressor for probing."""
    
    def __init__(self, input_dim: int, output_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear = nn.Linear(input_dim, output_dim)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.01)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class MLPRegressor(nn.Module):
    """MLP regressor with configurable layers."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] = None,
        dropout: float = 0.5,
        input_dropout: float = None,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]
        
        if input_dropout is None:
            input_dropout = dropout
        
        layers = []
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
        self._init_weights()
    
    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.mlp(x)


# =============================================================================
# Property Reconstruction Components
# =============================================================================

class MultiPropertyRegressor(nn.Module):
    """Multi-head regressor for basic graph property reconstruction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        input_dropout: float = None,
        use_mlp_heads: bool = True,
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)
        
        if input_dropout is None:
            input_dropout = dropout
        
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        if use_mlp_heads:
            self.head_avg_degree = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            
            self.head_gini = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            self.head_diameter = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.head_avg_degree = nn.Linear(input_dim, 1)
            self.head_gini = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
            self.head_diameter = nn.Linear(input_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Predict all properties from graph-level embeddings."""
        x = self.input_dropout(x)
        
        return {
            'avg_degree': self.head_avg_degree(x),
            'gini': self.head_gini(x),
            'diameter': self.head_diameter(x),
        }


class CommunityPropertyRegressor(nn.Module):
    """Multi-head regressor for community-related properties."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.3,
        input_dropout: float = None,
        use_mlp_heads: bool = True,
        K: int = 10,
    ):
        super().__init__()
        
        self.K = K
        
        if hidden_dim is None:
            hidden_dim = max(input_dim // 2, 32)
        
        if input_dropout is None:
            input_dropout = dropout
        
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 else nn.Identity()
        
        if use_mlp_heads:
            self.head_homophily = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            
            self.head_community_presence = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, K),
            )
            
            self.head_community_detection = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, K),
            )
        else:
            self.head_homophily = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())
            self.head_community_presence = nn.Linear(input_dim, K)
            self.head_community_detection = nn.Linear(input_dim, K)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x_graph=None, x_node=None):
        """Predict community-related properties."""
        result = {}
        
        if x_graph is not None:
            x_graph_dropout = self.input_dropout(x_graph)
            result['homophily'] = self.head_homophily(x_graph_dropout)
            result['community_presence'] = self.head_community_presence(x_graph_dropout)
        
        if x_node is not None:
            x_node_dropout = self.input_dropout(x_node)
            result['community_detection'] = self.head_community_detection(x_node_dropout)
        
        return result


def extract_normalization_scales_from_config(config: dict) -> dict:
    """Extract normalization scales from GraphUniverse config."""
    dataset_config = config.get("dataset", {})
    loader_target = dataset_config.get("loader", {}).get("_target_", "")
    
    if "GraphUniverse" in loader_target:
        gen_params = dataset_config.get("loader", {}).get("parameters", {}).get("generation_parameters", {})
        family_params = gen_params.get("family_parameters", {})
        
        max_n_nodes = family_params.get("n_nodes_range", [200, 200])[1]
        avg_degree_range = family_params.get("avg_degree_range", [2.0, 10.0])
        max_avg_degree = max(avg_degree_range)
        
        return {
            'homophily': 1.0,
            'avg_degree': max_avg_degree,
            'size': float(max_n_nodes),
            'gini': 1.0,
            'diameter': float(max_n_nodes),
        }
    else:
        return {
            'homophily': 1.0,
            'avg_degree': 20.0,
            'size': 200.0,
            'gini': 1.0,
            'diameter': 20.0,
        }


def extract_property_targets_from_batch(batch):
    """Extract pre-computed property targets from batch."""
    if not hasattr(batch, 'property_avg_degree'):
        raise ValueError("Graph properties not found. Please pre-compute properties first.")
    
    properties = {}
    for prop_name in ['avg_degree', 'gini', 'diameter']:
        prop_tensor = getattr(batch, f'property_{prop_name}')
        if prop_tensor.dim() == 1:
            prop_tensor = prop_tensor.unsqueeze(1)
        properties[prop_name] = prop_tensor
    
    return properties


def extract_community_property_targets_from_batch(batch):
    """Extract pre-computed community-related property targets from batch."""
    if not hasattr(batch, 'property_homophily'):
        raise ValueError("Graph properties not found. Please pre-compute properties first.")
    
    properties = {}
    
    homophily = batch.property_homophily
    if homophily.dim() == 1:
        homophily = homophily.unsqueeze(1)
    properties['homophily'] = homophily
    
    if hasattr(batch, 'property_community_presence'):
        properties['community_presence'] = batch.property_community_presence
    
    if hasattr(batch, 'y'):
        properties['community_detection'] = batch.y
    
    return properties


class GraphLevelEncoder(nn.Module):
    """Wrapper for graph-level tasks: encoder + pooling."""
    
    def __init__(
        self,
        feature_encoder: nn.Module,
        backbone: nn.Module,
        readout_type: str = "mean",
    ):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
        self.readout_type = readout_type
        
        if readout_type == "mean":
            self.pool = global_mean_pool
        elif readout_type == "max":
            self.pool = global_max_pool
        elif readout_type == "sum":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown readout type: {readout_type}")
    
    def forward(self, batch, return_node_features=False):
        """Get encoded features."""
        batch = prepare_batch_for_topobench(batch)
        
        batch_encoded = self.feature_encoder(batch)
        batch_encoded.x_0 = batch_encoded.x_0 if hasattr(batch_encoded, 'x_0') else (
            batch_encoded.get('x_0', batch_encoded.get('x')) if isinstance(batch_encoded, dict) else batch_encoded
        )
        
        model_out_from_wrapper = self.backbone(batch_encoded)
        node_features = model_out_from_wrapper["x_0"]
        
        batch_indices = batch_encoded.batch_0 if hasattr(batch_encoded, 'batch_0') else batch_encoded.batch
        graph_features = self.pool(node_features, batch_indices)
        
        if return_node_features:
            return {
                'node': node_features,
                'graph': graph_features,
            }
        else:
            return graph_features


class CommunityPropertyReconstructionModel(nn.Module):
    """Model for community-related property reconstruction."""
    
    def __init__(
        self,
        encoder: nn.Module,
        property_regressor: CommunityPropertyRegressor,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.property_regressor = property_regressor
        self.freeze_encoder = freeze_encoder
        self.task_level = "mixed"
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
    def forward(self, batch):
        """Forward pass."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder(batch, return_node_features=True)
        else:
            features = self.encoder(batch, return_node_features=True)
        
        return self.property_regressor(
            x_graph=features['graph'],
            x_node=features['node'],
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
    """Model for multi-property reconstruction."""
    
    def __init__(
        self,
        encoder: nn.Module,
        property_regressor: MultiPropertyRegressor,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.property_regressor = property_regressor
        self.freeze_encoder = freeze_encoder
        self.task_level = "graph"
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
    def forward(self, batch):
        """Forward pass."""
        if self.freeze_encoder:
            self.encoder.eval()
            with torch.no_grad():
                graph_features = self.encoder(batch)
        else:
            graph_features = self.encoder(batch)
        
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
    """Complete downstream model: encoder + classifier."""
    
    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        freeze_encoder: bool = True,
        task_level: str = "node",
    ):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        self.freeze_encoder = freeze_encoder
        self.task_level = task_level
        
        if freeze_encoder:
            self._freeze_encoder()
    
    def _freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()
        
        for module in self.encoder.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
    
    def forward(self, batch):
        """Forward pass."""
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
            self.encoder.eval()
        return self
    
    def get_model_state_info(self) -> dict:
        """Get diagnostic info about model state."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        encoder_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
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
    """Split data into train/val/test sets."""
    import random
    random.seed(seed)
    
    if dataset_info is None or dataset_info.get("type") == "GraphUniverse":
        indices = list(range(len(data_list)))
        random.shuffle(indices)
        
        train_indices = indices[:n_train]
        remaining = indices[n_train:]
        
        n_val = int(len(remaining) * val_ratio)
        val_indices = remaining[:n_val]
        test_indices = remaining[n_val:]
        
    else:
        total_size = len(data_list)
        indices = list(range(total_size))
        random.shuffle(indices)
        
        train_prop = 0.5
        base_train_size = int(total_size * train_prop)
        
        base_train_indices = indices[:base_train_size]
        remaining = indices[base_train_size:]
        
        if dataset_info.get("subsample_train") and dataset_info.get("n_train_requested"):
            n_train_actual = min(dataset_info["n_train_requested"], len(base_train_indices))
            train_indices = base_train_indices[:n_train_actual]
        else:
            train_indices = base_train_indices
        
        n_val = int(len(remaining) * val_ratio)
        val_indices = remaining[:n_val]
        test_indices = remaining[n_val:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    return train_data, val_data, test_data


def train_property_reconstruction(
    model: PropertyReconstructionModel | CommunityPropertyReconstructionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    use_wandb: bool,
    K: int,
    normalization_scales: dict,
    properties_to_include: list[str] | None = None,
    weight_decay: float = 0.0,
) -> dict:
    """Train property reconstruction with joint optimization."""
    
    model = model.to(device)
    is_community_model = isinstance(model, CommunityPropertyReconstructionModel)
    
    if is_community_model:
        properties = ['homophily', 'community_presence', 'community_detection']
    else:
        properties = ['avg_degree', 'gini', 'diameter']
    
    # Compute baselines on test set
    baseline_metrics = {}
    for prop in properties:
        if prop == 'community_detection':
            continue
        
        targets = []
        for batch in test_loader:
            if prop == 'community_presence':
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
                target = batch.property_community_presence.view(batch_size, K)
            else:
                target = getattr(batch, f'property_{prop}')
            targets.append(target)
        
        targets_tensor = torch.cat(targets, dim=0)
        test_mean = targets_tensor.mean(dim=0) if targets_tensor.dim() > 1 else targets_tensor.mean()
        baseline_mae = torch.abs(targets_tensor - test_mean).mean().item()
        
        baseline_metrics[prop] = {
            'mean': test_mean,
            'baseline_mae': baseline_mae,
        }
    
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
    }
    
    for prop in properties:
        history[f"train_loss_{prop}"] = []
        history[f"val_loss_{prop}"] = []
        if prop == 'community_detection':
            history[f"train_accuracy_{prop}"] = []
            history[f"val_accuracy_{prop}"] = []
        else:
            history[f"train_mae_{prop}"] = []
            history[f"val_mae_{prop}"] = []
    
    pbar = tqdm(range(epochs), desc="Training")
    
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        train_total = 0
        train_prop_losses = {prop: 0.0 for prop in properties}
        train_prop_metrics = {prop: 0.0 for prop in properties}
        
        for batch in train_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            optimizer.zero_grad()
            
            if is_community_model:
                encoder_output = model.encoder(batch, return_node_features=True)
                graph_embeddings = encoder_output['graph']
                node_features = encoder_output['node']
                predictions = model.property_regressor(x_graph=graph_embeddings, x_node=node_features)
                
                # Homophily loss (scaled x10)
                loss_homophily = mse_loss(predictions['homophily'], batch.property_homophily.unsqueeze(1).to(device))
                
                # Community presence loss (scaled x10)
                batch_size = graph_embeddings.size(0)
                community_presence_target = batch.property_community_presence.view(batch_size, K).to(device)
                loss_community_presence = bce_loss(predictions['community_presence'], community_presence_target)
                
                # Community detection loss (no scaling)
                loss_community_detection = ce_loss(predictions['community_detection'], batch.y.long().to(device))
                
                loss = 10.0 * loss_homophily + loss_community_presence + loss_community_detection
                
                train_prop_losses['homophily'] += loss_homophily.item() * batch_size
                train_prop_losses['community_presence'] += loss_community_presence.item() * batch_size
                train_prop_losses['community_detection'] += loss_community_detection.item() * batch_size
                
                # Compute metrics
                with torch.no_grad():
                    train_prop_metrics['homophily'] += torch.abs(predictions['homophily'] - batch.property_homophily.unsqueeze(1).to(device)).mean().item() * batch_size
                    train_prop_metrics['community_presence'] += torch.abs(torch.sigmoid(predictions['community_presence']) - community_presence_target).mean().item() * batch_size
                    pred_classes = predictions['community_detection'].argmax(dim=1)
                    train_prop_metrics['community_detection'] += (pred_classes == batch.y.long().to(device)).float().mean().item() * batch_size
            else:
                embeddings = model.encoder(batch)
                predictions = model.property_regressor(embeddings)
                
                # Individual losses
                loss_avg_degree = mse_loss(predictions['avg_degree'], batch.property_avg_degree.unsqueeze(1).to(device))
                loss_gini = mse_loss(predictions['gini'], batch.property_gini.unsqueeze(1).to(device))
                loss_diameter = mse_loss(predictions['diameter'], batch.property_diameter.unsqueeze(1).to(device))
                
                loss = loss_avg_degree + loss_gini + loss_diameter
                
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
                train_prop_losses['avg_degree'] += loss_avg_degree.item() * batch_size
                train_prop_losses['gini'] += loss_gini.item() * batch_size
                train_prop_losses['diameter'] += loss_diameter.item() * batch_size
                
                # Compute MAE metrics
                with torch.no_grad():
                    train_prop_metrics['avg_degree'] += torch.abs(predictions['avg_degree'] - batch.property_avg_degree.unsqueeze(1).to(device)).mean().item() * batch_size
                    train_prop_metrics['gini'] += torch.abs(predictions['gini'] - batch.property_gini.unsqueeze(1).to(device)).mean().item() * batch_size
                    train_prop_metrics['diameter'] += torch.abs(predictions['diameter'] - batch.property_diameter.unsqueeze(1).to(device)).mean().item() * batch_size
            
            loss.backward()
            optimizer.step()
            
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
            train_total += batch_size
            train_loss += loss.item() * batch_size
        
        train_loss /= train_total
        for prop in properties:
            train_prop_losses[prop] /= train_total
            train_prop_metrics[prop] /= train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        val_prop_losses = {prop: 0.0 for prop in properties}
        val_prop_metrics = {prop: 0.0 for prop in properties}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                batch = prepare_batch_for_topobench(batch)
                
                if is_community_model:
                    encoder_output = model.encoder(batch, return_node_features=True)
                    graph_embeddings = encoder_output['graph']
                    node_features = encoder_output['node']
                    predictions = model.property_regressor(x_graph=graph_embeddings, x_node=node_features)
                    
                    loss_homophily = mse_loss(predictions['homophily'], batch.property_homophily.unsqueeze(1).to(device))
                    
                    batch_size = graph_embeddings.size(0)
                    community_presence_target = batch.property_community_presence.view(batch_size, K).to(device)
                    loss_community_presence = bce_loss(predictions['community_presence'], community_presence_target)
                    
                    loss_community_detection = ce_loss(predictions['community_detection'], batch.y.long().to(device))
                    
                    loss = 10.0 * loss_homophily + loss_community_presence + loss_community_detection
                    
                    val_prop_losses['homophily'] += loss_homophily.item() * batch_size
                    val_prop_losses['community_presence'] += loss_community_presence.item() * batch_size
                    val_prop_losses['community_detection'] += loss_community_detection.item() * batch_size
                    
                    val_prop_metrics['homophily'] += torch.abs(predictions['homophily'] - batch.property_homophily.unsqueeze(1).to(device)).mean().item() * batch_size
                    val_prop_metrics['community_presence'] += torch.abs(torch.sigmoid(predictions['community_presence']) - community_presence_target).mean().item() * batch_size
                    pred_classes = predictions['community_detection'].argmax(dim=1)
                    val_prop_metrics['community_detection'] += (pred_classes == batch.y.long().to(device)).float().mean().item() * batch_size
                else:
                    embeddings = model.encoder(batch)
                    predictions = model.property_regressor(embeddings)
                    
                    loss_avg_degree = mse_loss(predictions['avg_degree'], batch.property_avg_degree.unsqueeze(1).to(device))
                    loss_gini = mse_loss(predictions['gini'], batch.property_gini.unsqueeze(1).to(device))
                    loss_diameter = mse_loss(predictions['diameter'], batch.property_diameter.unsqueeze(1).to(device))
                    
                    loss = loss_avg_degree + loss_gini + loss_diameter
                    
                    batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
                    val_prop_losses['avg_degree'] += loss_avg_degree.item() * batch_size
                    val_prop_losses['gini'] += loss_gini.item() * batch_size
                    val_prop_losses['diameter'] += loss_diameter.item() * batch_size
                    
                    val_prop_metrics['avg_degree'] += torch.abs(predictions['avg_degree'] - batch.property_avg_degree.unsqueeze(1).to(device)).mean().item() * batch_size
                    val_prop_metrics['gini'] += torch.abs(predictions['gini'] - batch.property_gini.unsqueeze(1).to(device)).mean().item() * batch_size
                    val_prop_metrics['diameter'] += torch.abs(predictions['diameter'] - batch.property_diameter.unsqueeze(1).to(device)).mean().item() * batch_size
                
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.batch.max().item() + 1
                val_total += batch_size
                val_loss += loss.item() * batch_size
        
        val_loss /= val_total
        for prop in properties:
            val_prop_losses[prop] /= val_total
            val_prop_metrics[prop] /= val_total
        
        # Update history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        
        for prop in properties:
            history[f"train_loss_{prop}"].append(train_prop_losses[prop])
            history[f"val_loss_{prop}"].append(val_prop_losses[prop])
            
            if prop == 'community_detection':
                history[f"train_accuracy_{prop}"].append(train_prop_metrics[prop])
                history[f"val_accuracy_{prop}"].append(val_prop_metrics[prop])
            else:
                history[f"train_mae_{prop}"].append(train_prop_metrics[prop])
                history[f"val_mae_{prop}"].append(val_prop_metrics[prop])
        
        # W&B logging
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "epoch": epoch,
                "train/loss_total": train_loss,
                "val/loss_total": val_loss,
                "lr": optimizer.param_groups[0]['lr'],
            }
            
            for prop in properties:
                log_dict[f"train/loss_{prop}"] = train_prop_losses[prop]
                log_dict[f"val/loss_{prop}"] = val_prop_losses[prop]
                
                # Add scaled losses for community model
                if is_community_model:
                    if prop == 'homophily':
                        log_dict[f"train/loss_{prop}_scaled"] = train_prop_losses[prop] * 10.0
                        log_dict[f"val/loss_{prop}_scaled"] = val_prop_losses[prop] * 10.0
                    elif prop == 'community_presence':
                        log_dict[f"train/loss_{prop}_scaled"] = train_prop_losses[prop]
                        log_dict[f"val/loss_{prop}_scaled"] = val_prop_losses[prop]
                    elif prop == 'community_detection':
                        log_dict[f"train/loss_{prop}_scaled"] = train_prop_losses[prop] * 1.0
                        log_dict[f"val/loss_{prop}_scaled"] = val_prop_losses[prop] * 1.0
                
                if prop == 'community_detection':
                    log_dict[f"train/accuracy_{prop}"] = train_prop_metrics[prop]
                    log_dict[f"val/accuracy_{prop}"] = val_prop_metrics[prop]
                else:
                    log_dict[f"train/mae_{prop}"] = train_prop_metrics[prop]
                    log_dict[f"val/mae_{prop}"] = val_prop_metrics[prop]
                    
                    if prop in baseline_metrics:
                        baseline_mae = baseline_metrics[prop]['baseline_mae']
                        log_dict[f"baseline/mae_{prop}"] = baseline_mae
                        improvement = ((baseline_mae - val_prop_metrics[prop]) / baseline_mae * 100) if baseline_mae > 0 else 0
                        log_dict[f"val/improvement_{prop}"] = improvement
            
            wandb.log(log_dict)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        pbar.set_postfix({"train_loss": f"{train_loss:.4f}", "val_loss": f"{val_loss:.4f}"})
        
        if patience_counter >= patience:
            break
    
    if best_model_state is not None:
        best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state_device)
    
    # Store baselines for evaluation
    model._baseline_metrics = baseline_metrics
    
    return history


def evaluate_property_reconstruction(
    model: PropertyReconstructionModel | CommunityPropertyReconstructionModel,
    test_loader: DataLoader,
    device: str = "cpu",
    use_wandb: bool = False,
    K: int = 10,
    normalization_scales: dict = None,
) -> dict:
    """Evaluate property reconstruction model on test set."""
    
    model = model.to(device)
    model.eval()
    
    is_community_model = isinstance(model, CommunityPropertyReconstructionModel)
    
    if is_community_model:
        properties = ['homophily', 'community_presence', 'community_detection']
    else:
        properties = ['avg_degree', 'gini', 'diameter']
    
    all_predictions = {prop: [] for prop in properties}
    all_targets = {prop: [] for prop in properties}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            batch = prepare_batch_for_topobench(batch)
            
            if is_community_model:
                encoder_output = model.encoder(batch, return_node_features=True)
                graph_embeddings = encoder_output['graph']
                node_features = encoder_output['node']
                
                predictions = model.property_regressor(x_graph=graph_embeddings, x_node=node_features)
                
                all_predictions['homophily'].append(predictions['homophily'].cpu())
                all_targets['homophily'].append(batch.property_homophily.unsqueeze(1).cpu())
                
                # Reshape community_presence from [batch_size * K] to [batch_size, K]
                batch_size = graph_embeddings.size(0)
                community_presence_target = batch.property_community_presence.view(batch_size, K).cpu()
                all_predictions['community_presence'].append(torch.sigmoid(predictions['community_presence']).cpu())
                all_targets['community_presence'].append(community_presence_target)
                
                all_predictions['community_detection'].append(predictions['community_detection'].argmax(dim=1).cpu())
                all_targets['community_detection'].append(batch.y.cpu())
            else:
                embeddings = model.encoder(batch)
                predictions = model.property_regressor(embeddings)
                
                all_predictions['avg_degree'].append(predictions['avg_degree'].cpu())
                all_targets['avg_degree'].append(batch.property_avg_degree.unsqueeze(1).cpu())
                
                all_predictions['gini'].append(predictions['gini'].cpu())
                all_targets['gini'].append(batch.property_gini.unsqueeze(1).cpu())
                
                all_predictions['diameter'].append(predictions['diameter'].cpu())
                all_targets['diameter'].append(batch.property_diameter.unsqueeze(1).cpu())
    
    results = {}
    total_graphs = 0
    
    # Get baselines if available
    baseline_metrics = getattr(model, '_baseline_metrics', {})
    
    for prop in properties:
        pred_tensor = torch.cat(all_predictions[prop], dim=0)
        target_tensor = torch.cat(all_targets[prop], dim=0)
        
        total_graphs = len(target_tensor)
        
        if prop == 'community_detection':
            accuracy = (pred_tensor == target_tensor).float().mean().item()
            results[f'test_accuracy_{prop}'] = accuracy
            results[f'test_mae_{prop}'] = accuracy
        else:
            mae = torch.abs(pred_tensor - target_tensor).mean().item()
            rmse = torch.sqrt(((pred_tensor - target_tensor) ** 2).mean()).item()
            
            results[f'test_mae_{prop}'] = mae
            results[f'test_rmse_{prop}'] = rmse
            
            # Compute baseline and improvement
            if prop in baseline_metrics:
                baseline_mae = baseline_metrics[prop]['baseline_mae']
                results[f'baseline_test_mae_{prop}'] = baseline_mae
                improvement = ((baseline_mae - mae) / baseline_mae * 100) if baseline_mae > 0 else 0
                results[f'improvement_{prop}'] = improvement
    
    # Compute weighted MAE for regression properties
    mae_props = [p for p in properties if p != 'community_detection']
    if mae_props:
        weighted_mae = sum(results[f'test_mae_{prop}'] for prop in mae_props) / len(mae_props)
        results['test_mae_weighted'] = weighted_mae
        
        # Weighted baseline
        if all(prop in baseline_metrics for prop in mae_props):
            baseline_weighted = sum(baseline_metrics[prop]['baseline_mae'] for prop in mae_props) / len(mae_props)
            results['baseline_test_mae_weighted'] = baseline_weighted
            improvement_weighted = ((baseline_weighted - weighted_mae) / baseline_weighted * 100) if baseline_weighted > 0 else 0
            results['improvement_weighted'] = improvement_weighted
    
    results['num_graphs'] = total_graphs
    
    if use_wandb and WANDB_AVAILABLE:
        log_dict = {}
        for prop in properties:
            if prop == 'community_detection':
                log_dict[f"test/accuracy_{prop}"] = results[f'test_accuracy_{prop}']
            else:
                log_dict[f"test/mae_{prop}"] = results[f'test_mae_{prop}']
                log_dict[f"test/rmse_{prop}"] = results[f'test_rmse_{prop}']
                
                if f'baseline_test_mae_{prop}' in results:
                    log_dict[f"test/baseline_mae_{prop}"] = results[f'baseline_test_mae_{prop}']
                    log_dict[f"test/improvement_{prop}"] = results[f'improvement_{prop}']
        
        if mae_props:
            log_dict["test/mae_weighted"] = results['test_mae_weighted']
            if 'baseline_test_mae_weighted' in results:
                log_dict["test/baseline_mae_weighted"] = results['baseline_test_mae_weighted']
                log_dict["test/improvement_weighted"] = results['improvement_weighted']
        
        log_dict["test/num_graphs"] = total_graphs
        
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
    """Train downstream model."""
    model = model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    
    if task_type == "regression":
        if loss_type == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        metric_name = loss_type
    else:
        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        metric_name = "accuracy"
    
    best_val_metric = float('inf') if task_type == "regression" else 0.0
    best_model_state = None
    patience_counter = 0
    history = {"train_loss": [], f"train_{metric_name}": [], "val_loss": [], f"val_{metric_name}": []}
    
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
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
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)
                else:
                    y = batch.y.long()
                num_samples = y.size(0)
            else:
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()
                else:
                    y = batch.y.view(-1).long()
                num_samples = y.size(0)
            
            loss = criterion(out, y)
            
            if task_type == "regression":
                if loss_type == "mae":
                    metric = torch.abs(out - y).mean()
                else:
                    metric = torch.pow(out - y, 2).mean()
            else:
                pred = out.argmax(dim=1)
                metric = (pred == y).float().mean()
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * num_samples
            train_metric_sum += metric.item() * num_samples
            train_total += num_samples
        
        train_loss /= train_total
        train_metric = train_metric_sum / train_total
        
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
                    if task_type == "regression":
                        y = batch.y.float().view(-1, 1)
                    else:
                        y = batch.y.long()
                    num_samples = y.size(0)
                else:
                    if task_type == "regression":
                        y = batch.y.view(-1, 1).float()
                    else:
                        y = batch.y.view(-1).long()
                    num_samples = y.size(0)

                loss = criterion(out, y)
                
                if task_type == "regression":
                    if loss_type == "mae":
                        metric = torch.abs(out - y).mean()
                    else:
                        metric = torch.pow(out - y, 2).mean()
                else:
                    pred = out.argmax(dim=1)
                    metric = (pred == y).float().mean()

                val_loss += loss.item() * num_samples
                val_metric_sum += metric.item() * num_samples
                val_total += num_samples
        
        val_loss /= val_total
        val_metric = val_metric_sum / val_total
        
        history["train_loss"].append(train_loss)
        history[f"train_{metric_name}"].append(train_metric)
        history["val_loss"].append(val_loss)
        history[f"val_{metric_name}"].append(val_metric)
        
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
        
        scheduler.step(val_metric)
        
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
            break
    
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
                if task_type == "regression":
                    y = batch.y.float().view(-1, 1)
                else:
                    y = batch.y.long()
                num_samples = y.size(0)
            else:
                if task_type == "regression":
                    y = batch.y.view(-1, 1).float()
                else:
                    y = batch.y.view(-1).long()
                num_samples = y.size(0)
            
            if task_type == "classification":
                pred = out.argmax(dim=1)
                all_preds.extend(pred.cpu().tolist())
                all_labels.extend(y.cpu().tolist())
            else:
                all_preds.extend(out.cpu().squeeze().tolist())
                all_labels.extend(y.cpu().squeeze().tolist())
            
            test_total += num_samples
    
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
        
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "test/accuracy": test_acc,
                f"test/num_{model.task_level}s": test_total,
            })
    else:
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
    mode: str = "linear",
    epochs: int = 100,
    lr: float = 0.001,
    batch_size: int = 32,
    patience: int = 20,
    device: str = "cpu",
    seed: int = 42,
    use_wandb: bool = False,
    wandb_project: str = "downstream_eval",
    graphuniverse_override: dict | None = None,
    classifier_dropout: float = 0.5,
    input_dropout: float = None,
    downstream_task: str | None = None,
    readout_type: str = "mean",
    pretraining_config: dict | None = None,
) -> dict:
    """Run full downstream evaluation pipeline."""
    run_dir = Path(run_dir)
    
    config = load_wandb_config(run_dir)
    task_level = detect_task_level(config)
    
    checkpoint_path = get_checkpoint_path_from_summary(run_dir)
    if checkpoint_path is None:
        raise ValueError("No checkpoint path found in wandb-summary.json")
    
    loader_target = config["dataset"]["loader"]["_target_"]
    dataset_params = config["dataset"]["parameters"]
    
    if downstream_task is not None:
        if downstream_task == "basic_property_reconstruction":
            task_type = "property_reconstruction"
            loss_type = "joint_mse"
            num_classes = 3
            actual_task_level = "graph"
        elif downstream_task == "community_related_property_reconstruction":
            task_type = "community_property_reconstruction"
            loss_type = "joint_mixed"
            num_classes = 10
            actual_task_level = "mixed"
        else:
            raise ValueError(f"Unknown downstream_task: {downstream_task}")
        
        gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
        task_name = gen_params.get("task", "community_detection")
        universe_params = gen_params.get("universe_parameters", {})
        num_classes = universe_params.get("K", 10)
    
    # Create datasets
    gen_params = config["dataset"]["loader"]["parameters"].get("generation_parameters", {})
    family_params = gen_params.get("family_parameters", {})
    universe_params = gen_params.get("universe_parameters", {})
    pretraining_universe_seed = universe_params["seed"]
    pretraining_family_seed = family_params["seed"]

    family_evaluation_seed = pretraining_family_seed + 1
    family_training_seed = pretraining_family_seed + 2
    
    eval_dataset, eval_data_dir, eval_dataset_info = create_dataset_from_config(
        config,
        n_graphs=n_evaluation_graphs,
        universe_seed=pretraining_universe_seed,
        family_seed=family_evaluation_seed,
        dataset_purpose="eval",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )
    
    transforms_config = config.get("transforms")
    eval_preprocessor = apply_transforms(eval_dataset, eval_data_dir, transforms_config)
    eval_data_list = eval_preprocessor.data_list
    
    train_dataset, train_data_dir, train_dataset_info = create_dataset_from_config(
        config,
        n_graphs=n_train,
        universe_seed=pretraining_universe_seed,
        family_seed=family_training_seed,
        dataset_purpose="train",
        graphuniverse_override=graphuniverse_override,
        downstream_task=downstream_task,
    )
    
    train_preprocessor = apply_transforms(train_dataset, train_data_dir, transforms_config)
    train_data = train_preprocessor.data_list
    
    # Split eval data into val/test
    import random
    random.seed(pretraining_universe_seed)
    eval_indices = list(range(len(eval_data_list)))
    random.shuffle(eval_indices)
    n_val = len(eval_indices) // 2
    val_indices = eval_indices[:n_val]
    test_indices = eval_indices[n_val:]
    val_data = [eval_data_list[i] for i in val_indices]
    test_data = [eval_data_list[i] for i in test_indices]
    
    # Pre-compute properties if needed
    if task_type in ["property_reconstruction", "community_property_reconstruction"]:
        from graph_properties import add_properties_to_dataset
        
        include_complex = (task_type == "community_property_reconstruction")
        train_data = add_properties_to_dataset(train_data, K=num_classes, include_complex=include_complex, verbose=False)
        val_data = add_properties_to_dataset(val_data, K=num_classes, include_complex=include_complex, verbose=False)
        test_data = add_properties_to_dataset(test_data, K=num_classes, include_complex=include_complex, verbose=False)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    # Load or create encoder
    if mode == "scratch" or mode == "scratch_frozen":
        feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
    else:
        feature_encoder, backbone, hidden_dim = load_pretrained_encoder(config, checkpoint_path, device=device)
    
    # Wrap encoder based on task level
    if actual_task_level in ["graph", "mixed"]:
        encoder = GraphLevelEncoder(feature_encoder, backbone, readout_type=readout_type)
    else:
        encoder = PretrainedEncoder(feature_encoder, backbone)
    
    # Verify encoder
    verify_encoder_outputs(encoder, train_loader, device=device)
    
    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb_config = {
            "mode": mode,
            "task_type": task_type,
            "task_level": task_level,
            "loss_type": loss_type,
            "n_evaluation_graphs": n_evaluation_graphs,
            "n_train": n_train,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "seed": seed,
            "num_classes": num_classes,
            "downstream_task": downstream_task,
            "readout_type": readout_type,
            "classifier_dropout": classifier_dropout,
            "input_dropout": input_dropout,
            "hidden_dim": hidden_dim,
        }
        
        if pretraining_config is not None:
            def flatten_dict(d, parent_key='', sep='/'):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}{sep}{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key, sep=sep).items())
                    else:
                        items.append((new_key, v))
                return dict(items)
            
            for key in ["dataset", "model", "optimizer", "trainer"]:
                if key in pretraining_config:
                    cfg = pretraining_config[key]
                    if isinstance(cfg, dict) and "value" in cfg:
                        cfg = cfg["value"]
                    flattened = flatten_dict(cfg)
                    for k, v in flattened.items():
                        wandb_config[f"pretrain/{key}/{k}"] = v
        
        wandb.init(project=wandb_project, config=wandb_config)
    
    # Create downstream model
    freeze_encoder = mode in ["linear", "mlp", "scratch_frozen"]
    use_mlp_heads = mode in ["mlp", "finetune-mlp", "scratch"]
    
    if task_type == "property_reconstruction":
        property_regressor = MultiPropertyRegressor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=classifier_dropout,
            input_dropout=input_dropout,
            use_mlp_heads=use_mlp_heads,
        )
        downstream_model = PropertyReconstructionModel(encoder, property_regressor, freeze_encoder)
    
    elif task_type == "community_property_reconstruction":
        property_regressor = CommunityPropertyRegressor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            dropout=classifier_dropout,
            input_dropout=input_dropout,
            use_mlp_heads=use_mlp_heads,
            K=num_classes,
        )
        downstream_model = CommunityPropertyReconstructionModel(encoder, property_regressor, freeze_encoder)
    
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Train
    normalization_scales = extract_normalization_scales_from_config(config)
    
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
        K=num_classes,
        normalization_scales=normalization_scales,
    )
    
    # Evaluate
    results = evaluate_property_reconstruction(
        model=downstream_model,
        test_loader=test_loader,
        device=device,
        use_wandb=use_wandb,
        K=num_classes,
        normalization_scales=normalization_scales,
    )
    
    # Cleanup wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    # Add metadata
    results["mode"] = mode
    results["task_type"] = task_type
    results["n_train"] = n_train
    results["n_val"] = len(val_data)
    results["n_test"] = len(test_data)
    results["num_classes"] = num_classes
    results["history"] = history
    results["encoder_frozen"] = freeze_encoder
    
    return results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Downstream evaluation pipeline.")
    parser.add_argument("--run_dir", type=str, required=True, help="Wandb run directory")
    parser.add_argument("--downstream_task", type=str, default="community_related_property_reconstruction",
                       choices=["basic_property_reconstruction", "community_related_property_reconstruction"])
    parser.add_argument("--n_evaluation_graphs", type=int, default=200)
    parser.add_argument("--n_train", type=int, default=50)
    parser.add_argument("--mode", type=str, default="finetune-linear",
                       choices=["linear", "mlp", "finetune-linear", "finetune-mlp", "scratch", "scratch_frozen"])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="downstream_eval")
    parser.add_argument("--graphuniverse_override", type=str, default=None)
    parser.add_argument("--classifier_dropout", type=float, default=0.3)
    parser.add_argument("--input_dropout", type=float, default=None)
    parser.add_argument("--readout_type", type=str, default="mean", choices=["mean", "max", "sum"])
    
    args = parser.parse_args()
    
    # Parse override
    graphuniverse_override = None
    if args.graphuniverse_override:
        graphuniverse_override = json.loads(args.graphuniverse_override)
    elif GRAPHUNIVERSE_OVERRIDE_DEFAULT:
        graphuniverse_override = GRAPHUNIVERSE_OVERRIDE_DEFAULT
    
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
    )
    
    # Print results
    task_type = results.get('task_type', 'classification')
    print("\n" + "=" * 60)
    print(f"RESULTS: {task_type.upper()}")
    print("=" * 60)
    print(f"Mode: {results['mode']}")
    
    if task_type == "property_reconstruction":
        print(f"Weighted MAE: {results['test_mae_weighted']:.4f}")
    elif task_type == "community_property_reconstruction":
        print(f"Weighted MAE: {results['test_mae_weighted']:.4f}")
    
    print(f"Train: {results['n_train']}, Val: {results['n_val']}, Test: {results['n_test']}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()