"""Wrapper for Graph Contrastive Learning (GraphCL) pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import dropout_edge, subgraph

from topobench.nn.wrappers.base import AbstractWrapper


class GraphCLGNNWrapper(AbstractWrapper):
    r"""Wrapper for Graph Contrastive Learning (GraphCL) pre-training with GNN models.

    This wrapper implements the augmentation and encoding logic for GraphCL
    self-supervised pre-training on graphs. GraphCL maximizes agreement
    between two augmented views of the same graph.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    aug1 : str, optional
        First augmentation type (default: "drop_edge").
        Options: "none", "drop_node", "drop_edge", "mask_attr", "subgraph"
    aug2 : str, optional
        Second augmentation type (default: "mask_attr").
        Options: "none", "drop_node", "drop_edge", "mask_attr", "subgraph"
    aug_ratio1 : float, optional
        Ratio for first augmentation (default: 0.2).
    aug_ratio2 : float, optional
        Ratio for second augmentation (default: 0.2).
    readout_type : str, optional
        Type of graph-level pooling (default: "mean").
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        aug1: str = "drop_edge",
        aug2: str = "mask_attr",
        aug_ratio1: float = 0.2,
        aug_ratio2: float = 0.2,
        readout_type: str = "mean",
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.aug1 = aug1
        self.aug2 = aug2
        self.aug_ratio1 = aug_ratio1
        self.aug_ratio2 = aug_ratio2
        self.readout_type = readout_type
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
    
    def augment(self, x, edge_index, batch_indices, aug_type, aug_ratio, device):
        """Apply augmentation to the graph.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        batch_indices : torch.Tensor
            Batch assignment for each node.
        aug_type : str
            Type of augmentation to apply.
        aug_ratio : float
            Ratio of augmentation.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (augmented_x, augmented_edge_index)
        """
        if aug_type == "none":
            return x, edge_index
        
        elif aug_type == "drop_node":
            # Randomly drop nodes
            num_nodes = x.size(0)
            keep_mask = torch.rand(num_nodes, device=device) > aug_ratio
            # Ensure at least one node per graph is kept
            keep_mask = self._ensure_connected(keep_mask, batch_indices)
            
            # Create new node features
            aug_x = x[keep_mask]
            
            # Create mapping from old to new node indices
            node_idx_mapping = torch.zeros(num_nodes, dtype=torch.long, device=device) - 1
            node_idx_mapping[keep_mask] = torch.arange(keep_mask.sum(), device=device)
            
            # Filter edges and remap indices
            edge_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
            aug_edge_index = node_idx_mapping[edge_index[:, edge_mask]]
            
            # Update batch indices
            new_batch_indices = batch_indices[keep_mask]
            
            return aug_x, aug_edge_index, new_batch_indices
        
        elif aug_type == "drop_edge":
            # Randomly drop edges using torch_geometric's dropout_edge
            aug_edge_index, _ = dropout_edge(edge_index, p=aug_ratio, training=True)
            return x, aug_edge_index, batch_indices
        
        elif aug_type == "mask_attr":
            # Randomly mask node attributes
            num_nodes = x.size(0)
            mask = torch.rand(num_nodes, device=device) < aug_ratio
            aug_x = x.clone()
            aug_x[mask] = 0.0  # Zero out masked nodes
            return aug_x, edge_index, batch_indices
        
        elif aug_type == "subgraph":
            # Sample a random connected subgraph
            num_nodes = x.size(0)
            keep_ratio = 1.0 - aug_ratio
            
            # Sample nodes to keep
            num_keep = max(int(num_nodes * keep_ratio), 1)
            perm = torch.randperm(num_nodes, device=device)[:num_keep]
            
            # Ensure we keep at least one node per graph
            keep_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            keep_mask[perm] = True
            keep_mask = self._ensure_connected(keep_mask, batch_indices)
            
            # Get subgraph
            aug_edge_index, _, edge_mask = subgraph(
                keep_mask, edge_index, relabel_nodes=True, return_edge_mask=True
            )
            aug_x = x[keep_mask]
            new_batch_indices = batch_indices[keep_mask]
            
            return aug_x, aug_edge_index, new_batch_indices
        
        else:
            raise ValueError(f"Unknown augmentation type: {aug_type}")
    
    def _ensure_connected(self, keep_mask, batch_indices):
        """Ensure at least one node per graph is kept.
        
        Parameters
        ----------
        keep_mask : torch.Tensor
            Boolean mask of nodes to keep.
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Updated keep mask with at least one node per graph.
        """
        # Get unique batch indices
        unique_batches = torch.unique(batch_indices)
        
        for b in unique_batches:
            batch_mask = batch_indices == b
            if not keep_mask[batch_mask].any():
                # If no node is kept for this graph, keep a random one
                indices = batch_mask.nonzero(as_tuple=True)[0]
                random_idx = indices[torch.randint(len(indices), (1,))]
                keep_mask[random_idx] = True
        
        return keep_mask
    
    def pool_graph(self, x, batch_indices):
        """Pool node embeddings to get graph-level representation.
        
        Parameters
        ----------
        x : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Graph-level representation of shape (num_graphs, hidden_dim).
        """
        if self.readout_type == "mean":
            return global_mean_pool(x, batch_indices)
        elif self.readout_type == "max":
            return global_max_pool(x, batch_indices)
        elif self.readout_type == "sum":
            return global_add_pool(x, batch_indices)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
    
    def forward(self, batch):
        r"""Forward pass for GraphCL encoding with augmentations.

        Creates two augmented views of each graph and encodes them.
        Augmentation is applied in BOTH training and validation.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Original encoded node features
            - z1: Graph-level embedding for first augmented view
            - z2: Graph-level embedding for second augmented view
            - batch_0: Original batch assignment
            - labels: Original labels (for compatibility)
        """
        # Input features
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        device = x_0.device
        
        # Create first augmented view
        aug_x1, aug_edge_index1, aug_batch1 = self.augment(
            x_0, edge_index, batch_indices, self.aug1, self.aug_ratio1, device
        )
        
        # Create second augmented view
        aug_x2, aug_edge_index2, aug_batch2 = self.augment(
            x_0, edge_index, batch_indices, self.aug2, self.aug_ratio2, device
        )
        
        # Encode first augmented view
        enc1 = self.backbone(
            aug_x1,
            aug_edge_index1,
            batch=aug_batch1,
            edge_weight=edge_weight if self.aug1 not in ["drop_edge", "drop_node", "subgraph"] else None,
        )
        
        # Encode second augmented view
        enc2 = self.backbone(
            aug_x2,
            aug_edge_index2,
            batch=aug_batch2,
            edge_weight=edge_weight if self.aug2 not in ["drop_edge", "drop_node", "subgraph"] else None,
        )
        
        # Pool to get graph-level representations
        z1 = self.pool_graph(enc1, aug_batch1)
        z2 = self.pool_graph(enc2, aug_batch2)
        
        # Also encode original for x_0 output
        enc_original = self.backbone(
            x_0,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Prepare outputs for readout
        model_out = {
            "x_0": enc_original,  # Original node encodings
            "z1": z1,  # Graph embedding from view 1
            "z2": z2,  # Graph embedding from view 2
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
        }
        
        return model_out

