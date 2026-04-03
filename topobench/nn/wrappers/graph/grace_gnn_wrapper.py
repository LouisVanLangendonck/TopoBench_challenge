"""Wrapper for GRACE (Graph Contrastive Representation Learning) pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
from torch_scatter import scatter

from topobench.nn.wrappers.base import AbstractWrapper


def compute_pagerank(edge_index, num_nodes, damp=0.85, k=10):
    """Compute PageRank centrality scores.
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    num_nodes : int
        Number of nodes.
    damp : float, optional
        Damping factor (default: 0.85).
    k : int, optional
        Number of iterations (default: 10).
    
    Returns
    -------
    torch.Tensor
        PageRank scores of shape [num_nodes].
    """
    device = edge_index.device
    deg_out = degree(edge_index[0], num_nodes=num_nodes)
    x = torch.ones(num_nodes, device=device, dtype=torch.float32)
    
    for _ in range(k):
        edge_msg = x[edge_index[0]] / (deg_out[edge_index[0]] + 1e-8)
        agg_msg = scatter(edge_msg, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        x = (1 - damp) * x + damp * agg_msg
    
    return x


def compute_eigenvector_centrality_approx(edge_index, num_nodes, k=10):
    """Compute approximate eigenvector centrality using power iteration.
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges].
    num_nodes : int
        Number of nodes.
    k : int, optional
        Number of power iterations (default: 10).
    
    Returns
    -------
    torch.Tensor
        Eigenvector centrality scores of shape [num_nodes].
    """
    device = edge_index.device
    x = torch.ones(num_nodes, device=device, dtype=torch.float32)
    x = x / x.norm()
    
    for _ in range(k):
        # Aggregate from neighbors
        x_new = scatter(x[edge_index[0]], edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        # Normalize
        x = x_new / (x_new.norm() + 1e-8)
    
    # Make positive
    x = torch.abs(x) + 1e-8
    
    return x


class GRACEGNNWrapper(AbstractWrapper):
    r"""Wrapper for GRACE pre-training with GNN models.

    GRACE (Graph Contrastive Representation Learning) uses:
    1. Adaptive topology augmentation (edge dropping based on centrality)
    2. Adaptive feature augmentation (feature masking based on importance)
    3. Contrastive loss between two augmented views

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    drop_edge_rate_1 : float, optional
        Overall edge drop rate for view 1 (default: 0.2).
    drop_edge_rate_2 : float, optional
        Overall edge drop rate for view 2 (default: 0.4).
    drop_feature_rate_1 : float, optional
        Overall feature mask rate for view 1 (default: 0.1).
    drop_feature_rate_2 : float, optional
        Overall feature mask rate for view 2 (default: 0.0).
    centrality_measure : str, optional
        Node centrality measure: "degree", "eigenvector", "pagerank" (default: "degree").
    drop_scheme : str, optional
        Dropping scheme: "uniform" or "adaptive" (default: "adaptive").
    threshold : float, optional
        Maximum drop probability threshold (default: 0.7).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        drop_edge_rate_1: float = 0.2,
        drop_edge_rate_2: float = 0.4,
        drop_feature_rate_1: float = 0.1,
        drop_feature_rate_2: float = 0.0,
        centrality_measure: str = "degree",
        drop_scheme: str = "adaptive",
        threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.drop_edge_rate_1 = drop_edge_rate_1
        self.drop_edge_rate_2 = drop_edge_rate_2
        self.drop_feature_rate_1 = drop_feature_rate_1
        self.drop_feature_rate_2 = drop_feature_rate_2
        self.centrality_measure = centrality_measure
        self.drop_scheme = drop_scheme
        self.threshold = threshold
        
        # Cache for centrality scores (computed once per graph)
        self._edge_weights_cache = None
        self._feature_weights_cache = None
        self._cache_initialized = False
    
    def _compute_edge_drop_weights(self, edge_index, num_nodes):
        """Compute edge drop weights based on centrality.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        num_nodes : int
            Number of nodes.
        
        Returns
        -------
        torch.Tensor
            Edge drop weights of shape [num_edges].
        """
        device = edge_index.device
        
        # Compute node centrality
        if self.centrality_measure == "degree":
            # Use in-degree for directed graphs, total degree for undirected
            edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
            node_centrality = degree(edge_index_undirected[1], num_nodes=num_nodes).float()
        elif self.centrality_measure == "eigenvector":
            edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
            node_centrality = compute_eigenvector_centrality_approx(edge_index_undirected, num_nodes)
        elif self.centrality_measure == "pagerank":
            node_centrality = compute_pagerank(edge_index, num_nodes)
        else:
            raise ValueError(f"Unknown centrality measure: {self.centrality_measure}")
        
        # Compute edge centrality as average of endpoints
        edge_centrality = (node_centrality[edge_index[0]] + node_centrality[edge_index[1]]) / 2.0
        
        # Log transform
        s_edge = torch.log(edge_centrality + 1e-8)
        
        # Normalize to probabilities (higher centrality = lower drop probability)
        s_max = s_edge.max()
        s_mean = s_edge.mean()
        
        # Inverse relationship: important edges (high centrality) get low drop prob
        weights = (s_max - s_edge) / (s_max - s_mean + 1e-8)
        
        return weights
    
    def _compute_feature_drop_weights(self, x, edge_index, num_nodes):
        """Compute feature drop weights based on node centrality.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, num_features].
        edge_index : torch.Tensor
            Edge indices.
        num_nodes : int
            Number of nodes.
        
        Returns
        -------
        torch.Tensor
            Feature drop weights of shape [num_features].
        """
        device = x.device
        
        # Compute node centrality
        if self.centrality_measure == "degree":
            edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
            node_centrality = degree(edge_index_undirected[1], num_nodes=num_nodes).float()
        elif self.centrality_measure == "eigenvector":
            edge_index_undirected = to_undirected(edge_index, num_nodes=num_nodes)
            node_centrality = compute_eigenvector_centrality_approx(edge_index_undirected, num_nodes)
        elif self.centrality_measure == "pagerank":
            node_centrality = compute_pagerank(edge_index, num_nodes)
        else:
            raise ValueError(f"Unknown centrality measure: {self.centrality_measure}")
        
        # Weight features by their occurrence in important nodes
        # For dense features, use absolute values
        x_abs = torch.abs(x)
        feature_weights = (x_abs.t() @ node_centrality).view(-1)
        
        # Log transform
        s_feat = torch.log(feature_weights + 1e-8)
        
        # Normalize (higher importance = lower drop probability)
        s_max = s_feat.max()
        s_mean = s_feat.mean()
        
        weights = (s_max - s_feat) / (s_max - s_mean + 1e-8)
        
        return weights
    
    def _initialize_cache(self, batch):
        """Initialize centrality-based weights cache."""
        if self._cache_initialized:
            return
        
        x_0 = batch.x_0
        edge_index = batch.edge_index
        num_nodes = x_0.size(0)
        
        if self.drop_scheme == "adaptive":
            # Compute edge and feature weights once
            self._edge_weights_cache = self._compute_edge_drop_weights(edge_index, num_nodes)
            self._feature_weights_cache = self._compute_feature_drop_weights(x_0, edge_index, num_nodes)
        
        self._cache_initialized = True
    
    def _drop_edges(self, edge_index, drop_rate):
        """Drop edges according to weights or uniformly.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape [2, num_edges].
        drop_rate : float
            Overall drop rate.
        
        Returns
        -------
        torch.Tensor
            Filtered edge indices.
        """
        num_edges = edge_index.size(1)
        device = edge_index.device
        
        if self.drop_scheme == "uniform":
            # Uniform random dropping
            mask = torch.rand(num_edges, device=device) > drop_rate
        else:
            # Adaptive dropping based on edge weights
            edge_weights = self._edge_weights_cache
            
            # Handle size mismatch (can happen with batched graphs)
            if edge_weights.size(0) != num_edges:
                # Fall back to uniform dropping if cache size doesn't match
                mask = torch.rand(num_edges, device=device) > drop_rate
            else:
                # Scale weights by drop_rate
                drop_probs = edge_weights / (edge_weights.mean() + 1e-8) * drop_rate
                drop_probs = torch.clamp(drop_probs, max=self.threshold)
                
                # Sample
                mask = torch.rand(num_edges, device=device) > drop_probs
        
        return edge_index[:, mask]
    
    def _drop_features(self, x, drop_rate):
        """Drop features according to weights or uniformly.
        
        Parameters
        ----------
        x : torch.Tensor
            Node features of shape [num_nodes, num_features].
        drop_rate : float
            Overall drop rate.
        
        Returns
        -------
        torch.Tensor
            Masked node features.
        """
        num_features = x.size(1)
        device = x.device
        
        if self.drop_scheme == "uniform":
            # Uniform random masking
            mask = torch.rand(num_features, device=device) > drop_rate
        else:
            # Adaptive masking based on feature weights
            feature_weights = self._feature_weights_cache
            
            # Handle size mismatch (should be same, but check anyway)
            if feature_weights.size(0) != num_features:
                # Fall back to uniform dropping
                mask = torch.rand(num_features, device=device) > drop_rate
            else:
                # Scale weights by drop_rate
                drop_probs = feature_weights / (feature_weights.mean() + 1e-8) * drop_rate
                drop_probs = torch.clamp(drop_probs, max=self.threshold)
                
                # Sample
                mask = torch.rand(num_features, device=device) > drop_probs
        
        # Apply mask
        x_masked = x.clone()
        x_masked[:, ~mask] = 0.0
        
        return x_masked
    
    def forward(self, batch):
        r"""Forward pass for GRACE encoding with two augmented views.

        IMPORTANT: This wrapper ALWAYS applies augmentation during pre-training
        (both train and eval modes) to properly compute contrastive loss metrics.
        
        For downstream tasks, use the standard GNNWrapper instead by loading
        only the backbone encoder without this wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing encoded representations from two augmented views.
        """
        # Initialize cache if needed (for adaptive augmentation)
        self._initialize_cache(batch)
        
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        # ALWAYS generate two augmented views for GRACE pre-training
        # (both during training and evaluation of the pre-training task)
        
        # View 1
        x_1 = self._drop_features(x_0, self.drop_feature_rate_1)
        edge_index_1 = self._drop_edges(edge_index, self.drop_edge_rate_1)
        
        # View 2
        x_2 = self._drop_features(x_0, self.drop_feature_rate_2)
        edge_index_2 = self._drop_edges(edge_index, self.drop_edge_rate_2)
        
        # Encode both views
        z_1 = self.backbone(
            x_1,
            edge_index_1,
            batch=batch_indices,
            edge_weight=edge_weight if edge_index_1.size(1) == edge_index.size(1) else None,
        )
        
        z_2 = self.backbone(
            x_2,
            edge_index_2,
            batch=batch_indices,
            edge_weight=edge_weight if edge_index_2.size(1) == edge_index.size(1) else None,
        )
        
        # Return both views for contrastive loss
        model_out = {
            "x_0": z_1,  # View 1 embeddings
            "z_1": z_1,  # View 1 embeddings (explicit)
            "z_2": z_2,  # View 2 embeddings
            "labels": batch.y if hasattr(batch, "y") else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,
        }
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"drop_edge_rate_1={self.drop_edge_rate_1}, "
            f"drop_edge_rate_2={self.drop_edge_rate_2}, "
            f"drop_feature_rate_1={self.drop_feature_rate_1}, "
            f"drop_feature_rate_2={self.drop_feature_rate_2}, "
            f"centrality_measure={self.centrality_measure}, "
            f"drop_scheme={self.drop_scheme})"
        )

