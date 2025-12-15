"""Wrapper for Deep Graph Infomax (DGI) pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

from topobench.nn.wrappers.base import AbstractWrapper


class DGIGNNWrapper(AbstractWrapper):
    r"""Wrapper for Deep Graph Infomax (DGI) pre-training with GNN models.

    This wrapper implements the encoding and corruption logic for DGI
    self-supervised pre-training on graphs. DGI maximizes mutual information
    between node representations and a graph-level summary.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    corruption_type : str, optional
        Type of corruption to apply. Options: "shuffle", "random" (default: "shuffle").
    readout_type : str, optional
        Type of readout for graph-level summary. Options: "mean", "max", "sum" (default: "mean").
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        corruption_type: str = "shuffle",
        readout_type: str = "mean",
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.corruption_type = corruption_type
        self.readout_type = readout_type
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
    
    def corrupt_features(self, x, batch_indices, device):
        """Corrupt node features by shuffling across the batch.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        batch_indices : torch.Tensor
            Batch assignment for each node.
        device : torch.device
            Device to use.
            
        Returns
        -------
        torch.Tensor
            Corrupted node features.
        """
        if self.corruption_type == "shuffle":
            # Shuffle features within the batch (across all nodes)
            perm = torch.randperm(x.size(0), device=device)
            return x[perm]
        elif self.corruption_type == "random":
            # Replace with random noise
            return torch.randn_like(x)
        else:
            raise ValueError(f"Unknown corruption type: {self.corruption_type}")
    
    def pool_graph(self, x, batch_indices):
        """Pool node embeddings to get graph-level summary.
        
        Parameters
        ----------
        x : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Graph-level summary of shape (num_graphs, hidden_dim).
        """
        if self.readout_type == "mean":
            return global_mean_pool(x, batch_indices)
        elif self.readout_type == "max":
            from torch_geometric.nn import global_max_pool
            return global_max_pool(x, batch_indices)
        elif self.readout_type == "sum":
            from torch_geometric.nn import global_add_pool
            return global_add_pool(x, batch_indices)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
    
    def cycle_index(self, num_graphs, shift=1):
        """Create cyclic shifted indices for negative sampling.
        
        Parameters
        ----------
        num_graphs : int
            Number of graphs in the batch.
        shift : int, optional
            Shift amount (default: 1).
            
        Returns
        -------
        torch.Tensor
            Shifted indices.
        """
        arr = torch.arange(num_graphs) + shift
        arr[-shift:] = torch.arange(shift)
        return arr
    
    def forward(self, batch):
        r"""Forward pass for DGI encoding with corruption.

        Generates positive samples (original graph) and negative samples
        (corrupted graph) for contrastive learning.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Encoded node features (positive)
            - x_corrupted: Encoded corrupted node features (negative)
            - summary: Graph-level summary embeddings
            - positive_expanded_summary: Summary expanded to match positive nodes
            - negative_expanded_summary: Summary from different graphs for negative pairs
            - batch_0: Batch assignment
            - labels: Original labels (for compatibility)
        """
        # Input features
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        device = x_0.device
        
        # Encode positive (original) features
        pos_enc = self.backbone(
            x_0,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Corrupt features for negative samples
        x_corrupted = self.corrupt_features(x_0, batch_indices, device)
        
        # Encode corrupted features (negative)
        neg_enc = self.backbone(
            x_corrupted,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Get graph-level summary (apply sigmoid as in original DGI)
        summary = torch.sigmoid(self.pool_graph(pos_enc, batch_indices))
        
        # Expand summary for positive samples (each node gets its graph's summary)
        positive_expanded_summary = summary[batch_indices]
        
        # Get number of unique graphs
        num_graphs = summary.size(0)
        
        # Create negative pairs by shifting summaries (each node gets a different graph's summary)
        shifted_indices = self.cycle_index(num_graphs, shift=1).to(device)
        shifted_summary = summary[shifted_indices]
        negative_expanded_summary = shifted_summary[batch_indices]
        
        # Prepare outputs for readout
        model_out = {
            "x_0": pos_enc,  # Positive node encodings
            "x_corrupted": neg_enc,  # Negative (corrupted) node encodings
            "summary": summary,  # Graph-level summary
            "positive_expanded_summary": positive_expanded_summary,
            "negative_expanded_summary": negative_expanded_summary,
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
        }
        
        return model_out

