"""
Graph Prompt modules for parameter-efficient fine-tuning.

Implements:
- GPF (Graph Prompt Feature): Single learnable feature vector added to all nodes
- GPF-Plus: Multiple learnable basis vectors with node-wise attention weights

References:
- Sun et al. "Graph Prompt Learning: A Graph-Level Prompting Method for Graph Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.inits import glorot


class GPF(nn.Module):
    """
    Graph Prompt Feature (GPF).
    
    Adds a single learnable global feature vector to all node embeddings.
    The pre-trained encoder remains frozen, only the prompt and classifier are trained.
    
    Parameters
    ----------
    in_channels : int
        Dimension of node features (must match encoder output dimension).
    """
    
    def __init__(self, in_channels: int):
        super(GPF, self).__init__()
        self.global_emb = nn.Parameter(torch.Tensor(1, in_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.global_emb)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add global prompt to node features.
        
        Parameters
        ----------
        x : Tensor
            Node features of shape (num_nodes, in_channels).
        
        Returns
        -------
        Tensor
            Prompted node features of shape (num_nodes, in_channels).
        """
        return x + self.global_emb


class GPFPlus(nn.Module):
    """
    Graph Prompt Feature-Plus (GPF-Plus).
    
    Uses multiple learnable basis vectors with node-wise attention weights
    to compute adaptive prompts for each node.
    
    Parameters
    ----------
    in_channels : int
        Dimension of node features (must match encoder output dimension).
    p_num : int
        Number of learnable basis vectors (default: 5).
    """
    
    def __init__(self, in_channels: int, p_num: int = 5):
        super(GPFPlus, self).__init__()
        self.p_num = p_num
        
        # Learnable basis vectors
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        
        # Linear projection to compute attention weights
        self.a = nn.Linear(in_channels, p_num)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.p_list)
        self.a.reset_parameters()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add node-wise adaptive prompt to node features.
        
        Parameters
        ----------
        x : Tensor
            Node features of shape (num_nodes, in_channels).
        
        Returns
        -------
        Tensor
            Prompted node features of shape (num_nodes, in_channels).
        """
        # Compute attention scores for each node
        score = self.a(x)  # (num_nodes, p_num)
        
        # Softmax to get attention weights
        weight = F.softmax(score, dim=1)  # (num_nodes, p_num)
        
        # Weighted sum of basis vectors
        p = weight.mm(self.p_list)  # (num_nodes, in_channels)
        
        return x + p


class PromptedEncoder(nn.Module):
    """
    Wrapper that integrates prompt with a frozen encoder.
    
    CRITICAL: The prompt is applied at the INPUT to the encoder (before the first GNN layer),
    NOT after the encoder completes. This matches the original GraphPrompt implementation
    where prompts are added to node features before message passing begins.
    
    Parameters
    ----------
    feature_encoder : nn.Module
        Feature encoder (will be frozen).
    backbone : nn.Module
        GNN backbone (will be frozen).
    prompt : nn.Module
        Prompt module (GPF or GPFPlus) - applied BEFORE backbone.
    """
    
    def __init__(self, feature_encoder: nn.Module, backbone: nn.Module, prompt: nn.Module):
        super(PromptedEncoder, self).__init__()
        self.feature_encoder = feature_encoder
        self.backbone = backbone
        self.prompt = prompt
        
        # Freeze encoder components
        for param in self.feature_encoder.parameters():
            param.requires_grad = False
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.feature_encoder.eval()
        self.backbone.eval()
    
    def forward(self, batch):
        """
        Forward pass: feature_encoder -> prompt -> backbone -> output.
        
        This order ensures prompts are added to the input features before
        message passing, which is the correct way according to GraphPrompt paper.
        
        Parameters
        ----------
        batch : Data
            PyG Data/Batch object.
        
        Returns
        -------
        Tensor
            Node/graph features after prompted encoding.
        """
        from downstream_eval import prepare_batch_for_topobench
        
        # Prepare batch
        batch = prepare_batch_for_topobench(batch)
        
        # Feature encoding (frozen)
        self.feature_encoder.eval()
        with torch.no_grad():
            batch_encoded = self.feature_encoder(batch)
        
        # Extract the encoded features
        if hasattr(batch_encoded, 'x_0'):
            x = batch_encoded.x_0
        elif isinstance(batch_encoded, dict):
            x = batch_encoded.get('x_0', batch_encoded.get('x'))
        else:
            x = batch_encoded
        
        # ⭐ APPLY PROMPT HERE - before backbone GNN ⭐
        x_prompted = self.prompt(x)
        
        # Get other attributes
        edge_index = batch.edge_index
        batch_indices = batch.batch_0 if hasattr(batch, 'batch_0') else batch.batch
        edge_weight = getattr(batch, 'edge_weight', None)
        
        # Backbone encoding with PROMPTED features (frozen)
        self.backbone.eval()
        with torch.no_grad():
            node_features = self.backbone(
                x_prompted,  # Use prompted features!
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )
        
        return node_features
    
    def train(self, mode=True):
        """Override train to keep encoder frozen."""
        super().train(mode)
        # Always keep encoder in eval mode
        self.feature_encoder.eval()
        self.backbone.eval()
        return self

