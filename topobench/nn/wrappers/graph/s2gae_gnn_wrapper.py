"""Wrapper for S2GAE (Structure-Aware Graph Autoencoder) pre-training with GNN models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling, add_self_loops

from topobench.nn.wrappers.base import AbstractWrapper


class S2GAEGNNWrapper(AbstractWrapper):
    r"""Wrapper for S2GAE pre-training with GNN models.

    S2GAE performs structure-aware self-supervised learning by:
    1. Masking edges randomly during training
    2. Collecting representations from all GNN layers
    3. Using cross-layer interactions for edge reconstruction
    
    The key idea is that different GNN layers capture different structural
    patterns, and combining them improves link prediction.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (must support returning intermediate layers).
    mask_ratio : float, optional
        The ratio of edges to mask during training (default: 0.5).
    mask_type : str, optional
        Type of masking: "undirected" or "directed" (default: "undirected").
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_ratio: float = 0.5,
        mask_type: str = "undirected",
        mask_during_eval: bool = True,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.mask_ratio = mask_ratio
        self.mask_type = mask_type
        self.mask_during_eval = mask_during_eval
        
    def mask_edges_undirected(self, edge_index, num_nodes, device):
        """Mask edges in an undirected manner.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (remaining_edges, masked_edges)
        """
        num_edges = edge_index.size(1)
        perm = torch.randperm(num_edges, device=device)
        num_mask = int(self.mask_ratio * num_edges)
        
        # Edges to keep for training encoder
        keep_indices = perm[num_mask:]
        remaining_edges = edge_index[:, keep_indices]
        
        # Edges that are masked (to be reconstructed)
        mask_indices = perm[:num_mask]
        masked_edges = edge_index[:, mask_indices]
        
        return remaining_edges, masked_edges
    
    def mask_edges_directed(self, edge_index, num_nodes, device):
        """Mask edges in a directed manner (includes reverse edges).
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (remaining_edges, masked_edges)
        """
        # For directed graphs, we also consider reverse edges
        # Create reverse edges
        reverse_edges = torch.stack([edge_index[1], edge_index[0]], dim=0)
        all_edges = torch.cat([edge_index, reverse_edges], dim=1)
        
        # Now mask from combined edge set
        num_edges = all_edges.size(1)
        perm = torch.randperm(num_edges, device=device)
        num_mask = int(self.mask_ratio * num_edges)
        
        keep_indices = perm[num_mask:]
        remaining_edges = all_edges[:, keep_indices]
        
        mask_indices = perm[:num_mask]
        masked_edges = all_edges[:, mask_indices]
        
        return remaining_edges, masked_edges
    
    def forward(self, batch):
        r"""Forward pass for S2GAE encoding with edge masking.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing multi-layer representations and masked edges.
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Apply edge masking during training (and optionally during eval)
        if self.training or self.mask_during_eval:
            if self.mask_type == "undirected":
                remaining_edges, masked_edges = self.mask_edges_undirected(
                    edge_index, num_nodes, device
                )
            else:
                remaining_edges, masked_edges = self.mask_edges_directed(
                    edge_index, num_nodes, device
                )
            
            # Add self-loops to remaining edges
            use_edge_index, _ = add_self_loops(remaining_edges, num_nodes=num_nodes)
        else:
            use_edge_index = edge_index
            masked_edges = None
        
        # Encode with GNN and collect ALL layer representations
        # The backbone needs to return intermediate representations
        # We'll call it and expect a list or we'll manually iterate
        layer_reps = []
        x = x_0
        
        # Check if backbone has a method to return all layers
        if hasattr(self.backbone, 'convs'):
            # Manual forward through layers (for GCN, SAGE, etc.)
            for i, conv in enumerate(self.backbone.convs):
                try:
                    # Try with edge_weight first (for GCN, SAGE, etc.)
                    if edge_weight is not None:
                        x = conv(x, use_edge_index, edge_weight=edge_weight)
                    else:
                        x = conv(x, use_edge_index)
                except TypeError:
                    # If edge_weight not supported (GIN, etc.), call without it
                    x = conv(x, use_edge_index)
                
                if i < len(self.backbone.convs) - 1:
                    x = F.relu(x)
                    if hasattr(self.backbone, 'dropout'):
                        x = F.dropout(x, p=self.backbone.dropout, training=self.training)
                layer_reps.append(x)
        else:
            # Fallback: just use final output
            try:
                x = self.backbone(x, use_edge_index, batch=batch_indices, edge_weight=edge_weight)
            except TypeError:
                x = self.backbone(x, use_edge_index, batch=batch_indices)
            layer_reps = [x]
        
        # Prepare outputs for readout
        model_out = {
            "layer_reps": layer_reps,  # List of representations from each layer
            "x_0": layer_reps[-1],  # Final layer representation
            "masked_edges": masked_edges,  # Edges that were masked
            "full_edge_index": edge_index,  # Original full edge index
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
        }
        
        return model_out



