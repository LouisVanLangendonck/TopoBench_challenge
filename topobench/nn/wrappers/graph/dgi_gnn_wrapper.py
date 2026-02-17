"""Wrapper for DGI (Deep Graph Infomax) pre-training with GNN models."""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class DGIGNNWrapper(AbstractWrapper):
    r"""Wrapper for Deep Graph Infomax (DGI) pre-training with GNN models.

    DGI maximizes mutual information between patch representations (node embeddings)
    and graph-level summary vectors. This is achieved by:
    1. Encoding the original graph to get node embeddings (positive samples)
    2. Creating a corrupted version of the graph (negative samples)
    3. Encoding the corrupted graph
    4. Using a discriminator to distinguish between positive and negative samples
    
    The corruption function shuffles node features across nodes while keeping
    the graph structure intact.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (GNN).
    corruption_type : str, optional
        Type of corruption to apply: "feature_shuffle" (default).
        - "feature_shuffle": Randomly permute node features across nodes
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
        
    Notes
    -----
    This wrapper works for both transductive (single graph) and inductive (multiple graphs)
    settings by processing each graph in the batch separately.
    """

    def __init__(
        self,
        backbone: nn.Module,
        corruption_type: str = "feature_shuffle",
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.corruption_type = corruption_type
        
        if corruption_type not in ["feature_shuffle"]:
            raise ValueError(
                f"corruption_type must be 'feature_shuffle', got {corruption_type}"
            )
    
    def corrupt_graph(self, x, batch_indices):
        """Corrupt node features by shuffling them across nodes within each graph.
        
        This creates negative samples for the discriminator while preserving
        the graph structure.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Corrupted node features with same shape as input.
        """
        if self.corruption_type == "feature_shuffle":
            # Shuffle features within each graph separately
            batch_ids = batch_indices.unique()
            corrupted_x = x.clone()
            
            for batch_id in batch_ids:
                # Get nodes for this graph
                graph_mask = (batch_indices == batch_id)
                graph_indices = torch.where(graph_mask)[0]
                
                # Shuffle node indices within this graph
                perm = torch.randperm(len(graph_indices), device=x.device)
                shuffled_indices = graph_indices[perm]
                
                # Replace features with shuffled features
                corrupted_x[graph_indices] = x[shuffled_indices]
            
            return corrupted_x
        else:
            raise ValueError(f"Unknown corruption_type: {self.corruption_type}")
    
    def forward(self, batch):
        r"""Forward pass for DGI encoding.

        This wrapper ALWAYS applies corruption during pre-training to generate
        positive and negative samples for the discriminator.
        
        For downstream tasks, use the standard GNNWrapper instead by loading
        only the backbone encoder without this wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Node embeddings from positive (original) graph
            - x_0_corrupted: Node embeddings from negative (corrupted) graph
            - batch_0: Batch indices
            - edge_index: Edge indices (for readout)
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        # Encode original graph (positive samples)
        h_positive = self.backbone(
            x_0,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Create corrupted graph (negative samples)
        x_corrupted = self.corrupt_graph(x_0, batch_indices)
        
        # Encode corrupted graph (negative samples)
        h_negative = self.backbone(
            x_corrupted,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Return both positive and negative embeddings for discriminator
        model_out = {
            "x_0": h_positive,  # Positive node embeddings
            "x_0_corrupted": h_negative,  # Negative node embeddings
            "batch_0": batch_indices,
            "edge_index": edge_index,
            "labels": batch.y if hasattr(batch, 'y') else None,
        }
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"corruption_type={self.corruption_type})"
        )


