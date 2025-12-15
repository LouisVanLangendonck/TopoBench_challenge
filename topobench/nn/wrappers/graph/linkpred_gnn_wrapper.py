"""Wrapper for Link Prediction pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling, to_undirected

from topobench.nn.wrappers.base import AbstractWrapper


class LinkPredGNNWrapper(AbstractWrapper):
    r"""Wrapper for Link Prediction pre-training with GNN models.

    This wrapper implements edge masking and negative sampling for
    self-supervised link prediction pre-training on graphs.
    The task is to predict whether edges exist or not.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    mask_ratio : float, optional
        Ratio of edges to mask for prediction (default: 0.15).
    neg_sampling_ratio : float, optional
        Ratio of negative samples per positive edge (default: 1.0).
    add_negative_train_samples : bool, optional
        Whether to add negative samples during training (default: True).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_ratio: float = 0.15,
        neg_sampling_ratio: float = 1.0,
        add_negative_train_samples: bool = True,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.mask_ratio = mask_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.add_negative_train_samples = add_negative_train_samples
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
    
    def sample_edges(self, edge_index, num_nodes, batch_indices, device):
        """Sample positive and negative edges for link prediction.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        num_nodes : int
            Total number of nodes in the batch.
        batch_indices : torch.Tensor
            Batch assignment for each node.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (message_edge_index, pos_edge_index, neg_edge_index)
            - message_edge_index: Edges used for message passing (remaining after masking)
            - pos_edge_index: Positive edges to predict (masked edges)
            - neg_edge_index: Negative edges (non-existent)
        """
        num_edges = edge_index.size(1)
        
        # Number of edges to mask
        num_mask = max(int(num_edges * self.mask_ratio), 1)
        
        # Random permutation to select edges to mask
        perm = torch.randperm(num_edges, device=device)
        
        # Split into message passing edges and supervision edges
        mask_idx = perm[:num_mask]
        msg_idx = perm[num_mask:]
        
        # Message passing edges (for encoding)
        message_edge_index = edge_index[:, msg_idx]
        
        # Positive supervision edges (to predict)
        pos_edge_index = edge_index[:, mask_idx]
        
        # Sample negative edges
        if self.add_negative_train_samples:
            num_neg = int(num_mask * self.neg_sampling_ratio)
            neg_edge_index = negative_sampling(
                edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg,
            )
        else:
            neg_edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        return message_edge_index, pos_edge_index, neg_edge_index
    
    def forward(self, batch):
        r"""Forward pass for link prediction encoding with edge masking.

        Masks some edges for prediction and samples negative edges.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Encoded node features
            - pos_edge_index: Positive edges to predict
            - neg_edge_index: Negative edges
            - batch_0: Batch assignment
            - labels: Original labels (for compatibility)
        """
        # Input features
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        device = x_0.device
        num_nodes = x_0.size(0)
        
        # Sample edges for training
        message_edge_index, pos_edge_index, neg_edge_index = self.sample_edges(
            edge_index, num_nodes, batch_indices, device
        )
        
        # Encode using message passing edges only
        node_enc = self.backbone(
            x_0,
            message_edge_index,
            batch=batch_indices,
            edge_weight=None,  # Edge weights don't match masked edges
        )
        
        # Prepare outputs for readout
        model_out = {
            "x_0": node_enc,  # Node encodings
            "pos_edge_index": pos_edge_index,  # Positive edges to predict
            "neg_edge_index": neg_edge_index,  # Negative edges
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
        }
        
        return model_out


class LinkPredGNNWrapperInductive(AbstractWrapper):
    r"""Wrapper for Link Prediction pre-training with inductive edge splits.

    This variant uses all edges for message passing but creates train/val/test
    splits of edges for evaluation. Better suited for inductive settings.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    neg_sampling_ratio : float, optional
        Ratio of negative samples per positive edge (default: 1.0).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neg_sampling_ratio: float = 1.0,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.neg_sampling_ratio = neg_sampling_ratio
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
    
    def forward(self, batch):
        r"""Forward pass for inductive link prediction.

        Uses all edges for encoding and samples supervision edges.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Encoded node features
            - pos_edge_index: All edges as positive samples
            - neg_edge_index: Sampled negative edges
            - batch_0: Batch assignment
            - labels: Original labels (for compatibility)
        """
        # Input features
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        device = x_0.device
        num_nodes = x_0.size(0)
        num_edges = edge_index.size(1)
        
        # Encode using all edges
        node_enc = self.backbone(
            x_0,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Sample a subset of positive edges for supervision
        num_pos = max(int(num_edges * 0.5), 1)  # Use 50% of edges for supervision
        perm = torch.randperm(num_edges, device=device)[:num_pos]
        pos_edge_index = edge_index[:, perm]
        
        # Sample negative edges
        num_neg = int(num_pos * self.neg_sampling_ratio)
        neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=num_neg,
        )
        
        # Prepare outputs for readout
        model_out = {
            "x_0": node_enc,  # Node encodings
            "pos_edge_index": pos_edge_index,  # Positive edges to predict
            "neg_edge_index": neg_edge_index,  # Negative edges
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
        }
        
        return model_out

