"""
Prompt-based adaptation techniques for downstream evaluation.

This module provides functions to create and train models using:
- GPF (Graph Prompt Feature): Single global prompt vector
- GPF-Plus: Multiple basis vectors with node-wise attention

These techniques freeze the pre-trained encoder and only tune:
1. The prompt parameters
2. The downstream classifier
"""

import torch
import torch.nn as nn
from pathlib import Path

# Import from existing downstream_eval
import sys
_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from downstream_eval import (
    load_pretrained_encoder,
    create_random_encoder,
    PretrainedEncoder,
    GraphLevelEncoder,
    DownstreamModel,
)

from prompt_modules import GPF, GPFPlus, PromptedEncoder


def create_prompted_model(
    config: dict,
    checkpoint_path: str | Path,
    num_classes: int,
    prompt_type: str = "gpf",
    p_num: int = 5,
    task_level: str = "node",
    device: str = "cpu",
    classifier_type: str = "mlp",
    mode: str = "finetune",  # "finetune" or "scratch"
    readout_type: str = "sum",  # For graph-level tasks
    task_type: str = "classification",  # "classification" or "regression"
) -> tuple[nn.Module, dict]:
    """
    Create a prompted model for downstream evaluation.
    
    Parameters
    ----------
    config : dict
        Configuration from wandb run.
    checkpoint_path : str or Path
        Path to pre-trained checkpoint (ignored if mode="scratch").
    num_classes : int
        Number of output classes.
    prompt_type : str
        Type of prompt: "gpf" or "gpf-plus".
    p_num : int
        Number of basis vectors for GPF-Plus (default: 5).
    task_level : str
        "node" or "graph" level task.
    device : str
        Device to use.
    classifier_type : str
        Type of classifier: "linear" or "mlp".
    mode : str
        "finetune" (use pre-trained weights) or "scratch" (random init).
    readout_type : str
        Pooling type for graph-level tasks: "mean", "max", "sum" (default: "sum").
    task_type : str
        Task type: "classification" or "regression" (default: "classification").
    
    Returns
    -------
    tuple
        (model, info_dict) where model is the complete downstream model
        and info_dict contains metadata about the model configuration.
    """
    # Load encoder components (pre-trained or random)
    if mode == "scratch":
        print(f"Creating randomly initialized encoder for {prompt_type} baseline...")
        feature_encoder, backbone, hidden_dim = create_random_encoder(config, device=device)
    else:
        print(f"Loading pre-trained encoder for {prompt_type}...")
        feature_encoder, backbone, hidden_dim = load_pretrained_encoder(
            config, checkpoint_path, device=device
        )
    
    # Create prompt module (operates on features, applied before backbone)
    if prompt_type == "gpf":
        prompt = GPF(in_channels=hidden_dim)
        print(f"  Created GPF prompt (1 learnable vector, dim={hidden_dim})")
    elif prompt_type == "gpf-plus":
        prompt = GPFPlus(in_channels=hidden_dim, p_num=p_num)
        print(f"  Created GPF-Plus prompt ({p_num} basis vectors, dim={hidden_dim})")
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    
    prompt = prompt.to(device)
    
    # Create prompted encoder (integrates prompt BEFORE backbone)
    # This matches GraphPrompt paper: feature_encoder -> prompt -> backbone
    if task_level == "graph":
        # For graph-level: we need to add pooling after backbone
        from torch_geometric.nn import global_mean_pool
        
        class GraphLevelPromptedEncoder(nn.Module):
            def __init__(self, feature_encoder, backbone, prompt, readout_type="mean"):
                super().__init__()
                self.prompted_encoder = PromptedEncoder(feature_encoder, backbone, prompt)
                
                if readout_type == "mean":
                    self.pool = global_mean_pool
                elif readout_type == "max":
                    from torch_geometric.nn import global_max_pool
                    self.pool = global_max_pool
                elif readout_type == "sum":
                    from torch_geometric.nn import global_add_pool
                    self.pool = global_add_pool
                else:
                    raise ValueError(f"Unknown readout type: {readout_type}")
            
            def forward(self, batch):
                from downstream_eval import prepare_batch_for_topobench
                batch = prepare_batch_for_topobench(batch)
                
                node_features = self.prompted_encoder(batch)
                batch_indices = batch.batch_0 if hasattr(batch, 'batch_0') else batch.batch
                graph_features = self.pool(node_features, batch_indices)
                return graph_features
            
            def train(self, mode=True):
                super().train(mode)
                self.prompted_encoder.train(mode)
                return self
        
        prompted_encoder = GraphLevelPromptedEncoder(feature_encoder, backbone, prompt, readout_type=readout_type)
        print(f"  Using GraphLevelPromptedEncoder with {readout_type} pooling")
    else:
        # Node-level: just use the prompted encoder directly
        prompted_encoder = PromptedEncoder(feature_encoder, backbone, prompt)
        print(f"  Using PromptedEncoder (node-level)")
    
    # Create classifier or regressor
    from downstream_eval import (
        LinearClassifier, MLPClassifier, 
        LinearRegressor, MLPRegressor
    )
    
    if task_type == "regression":
        if classifier_type == "linear":
            classifier = LinearRegressor(hidden_dim, num_classes)
        else:  # mlp
            classifier = MLPRegressor(hidden_dim, num_classes)
    else:  # classification
        if classifier_type == "linear":
            classifier = LinearClassifier(hidden_dim, num_classes)
        else:  # mlp
            classifier = MLPClassifier(hidden_dim, num_classes)
    
    # Create complete downstream model
    # Note: freeze_encoder=True because PromptedEncoder already handles freezing
    model = DownstreamModel(
        encoder=prompted_encoder,
        classifier=classifier,
        freeze_encoder=False,  # Already frozen inside PromptedEncoder
        task_level=task_level,
    )
    
    # Calculate trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    prompt_params = sum(p.numel() for p in prompt.parameters())
    classifier_params = sum(p.numel() for p in classifier.parameters())
    # Encoder params = feature_encoder + backbone
    encoder_params = sum(p.numel() for p in feature_encoder.parameters()) + sum(p.numel() for p in backbone.parameters())
    
    info = {
        "prompt_type": prompt_type,
        "classifier_type": classifier_type,
        "p_num": p_num if prompt_type == "gpf-plus" else 1,
        "hidden_dim": hidden_dim,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "encoder_params": encoder_params,
        "prompt_params": prompt_params,
        "classifier_params": classifier_params,
        "encoder_frozen": True,
        "mode": mode,
    }
    
    print(f"\nModel parameter summary:")
    print(f"  Total params: {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"    - Encoder: {encoder_params:,} (frozen)")
    print(f"    - Prompt ({prompt_type}): {prompt_params:,} (trainable)")
    print(f"    - Classifier ({classifier_type}): {classifier_params:,} (trainable)")
    
    # Show the complexity difference
    if classifier_type == "mlp":
        linear_params = hidden_dim * num_classes
        complexity_ratio = classifier_params / linear_params
        print(f"  Note: MLP classifier is {complexity_ratio:.1f}x more complex than linear ({classifier_params:,} vs {linear_params:,} params)")
    
    return model, info


def get_prompt_hyperparams():
    """
    Get hyperparameter grid for prompt tuning methods.
    
    Returns
    -------
    dict
        Dictionary with hyperparameter options for each prompt type.
    """
    return {
        "gpf": {
            "p_num": [1],  # Not used, but kept for consistency
            "lr": [0.001, 0.0001],  # Learning rates to try
            "classifier_type": ["linear", "mlp"],  # linear or mlp
        },
        "gpf-plus": {
            "p_num": [3, 5, 10],  # Number of basis vectors
            "lr": [0.001, 0.0001],  # Learning rates to try
            "classifier_type": ["linear", "mlp"],  # linear or mlp
        },
    }

