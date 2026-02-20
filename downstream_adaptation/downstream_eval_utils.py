"""Shared utilities for downstream evaluation (inductive and transductive)."""

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
import yaml
from omegaconf import OmegaConf
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

from topobench.data.preprocessor import PreProcessor
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
        elif "GraphMAE" in wrapper_target:
            return "graphmae"
        elif "GraphCL" in wrapper_target:
            return "graphcl"
        elif "LinkPred" in wrapper_target:
            return "linkpred"
    
    loss_config = config.get("loss", {})
    dataset_loss = loss_config.get("dataset_loss", {})
    loss_target = dataset_loss.get("_target_", "")
    
    if "DGI" in loss_target:
        return "dgi"
    elif "GraphMAEv2" in loss_target:
        return "graphmaev2"
    elif "GRACE" in loss_target or "grace" in loss_target.lower():
        return "grace"
    elif "GraphMAE" in loss_target:
        return "graphmae"
    elif "LinkPred" in loss_target:
        return "linkpred"
    
    return "supervised"


def detect_task_level(config: dict) -> str:
    """Detect task level from config."""
    dataset_params = config.get("dataset", {}).get("parameters", {})
    task_level = dataset_params.get("task_level", "node")
    return task_level


def detect_learning_setting(config: dict) -> str:
    """Detect learning setting (inductive/transductive) from config."""
    split_params = config.get("dataset", {}).get("split_params", {})
    learning_setting = split_params.get("learning_setting", "inductive")
    return learning_setting


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


def verify_encoder_outputs(encoder: nn.Module, data_loader, device: str = "cpu"):
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


# =============================================================================
# Encoder Wrappers
# =============================================================================

class PretrainedEncoder(nn.Module):
    """Wrapper combining feature encoder + backbone for node-level features."""
    
    def __init__(self, feature_encoder: nn.Module, backbone: nn.Module):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
    
    def forward(self, batch):
        """Get encoded node-level features."""
        batch = prepare_batch_for_topobench(batch)
        
        # IMPORTANT: Reset x_0 to original features before encoding
        # The feature encoder modifies x_0 in-place, so we need to reset it
        # each time to avoid applying the transformation twice
        if hasattr(batch, 'x') and not hasattr(batch, 'x_0'):
            batch.x_0 = batch.x
        elif hasattr(batch, 'x'):
            # Reset x_0 to original x for each forward pass
            batch.x_0 = batch.x
        
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

