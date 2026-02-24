"""Wrapper for Link Prediction pre-training with GNN models."""

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from topobench.nn.wrappers.base import AbstractWrapper


class LinkPredGNNWrapper(AbstractWrapper):
    r"""Wrapper for Link Prediction pre-training with GNN models.

    This wrapper implements link prediction as a self-supervised pretraining task:
    1. Remove a percentage of edges from each graph (positive samples)
    2. Run message passing on remaining graph
    3. Generate node embeddings via GNN
    4. Sample negative edges (non-existent edges)
    5. Readout will score edges and classify positive vs negative

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    edge_sample_ratio : float, optional
        Ratio of edges to remove for prediction (default: 0.5).
        E.g., 0.5 means 50% of edges are removed and used as positive samples.
    neg_sample_ratio : float, optional
        Ratio of negative samples per positive sample (default: 1.0).
        E.g., 1.0 means equal number of negative and positive samples.
    sampling_method : str, optional
        Negative sampling method: "sparse" or "dense" (default: "sparse").
        Use "sparse" for large graphs, "dense" for small graphs.
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_sample_ratio: float = 0.5,
        neg_sample_ratio: float = 1.0,
        sampling_method: str = "sparse",
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.edge_sample_ratio = edge_sample_ratio
        self.neg_sample_ratio = neg_sample_ratio
        self.sampling_method = sampling_method
        
        # Validate parameters
        if not 0.0 < edge_sample_ratio < 1.0:
            raise ValueError(f"edge_sample_ratio must be in (0, 1), got {edge_sample_ratio}")
        if neg_sample_ratio <= 0:
            raise ValueError(f"neg_sample_ratio must be positive, got {neg_sample_ratio}")
        if sampling_method not in ["sparse", "dense"]:
            raise ValueError(f"sampling_method must be 'sparse' or 'dense', got {sampling_method}")
    
    def sample_edges(self, edge_index, batch_indices, num_nodes, device):
        """Sample positive edges (to remove) and keep remaining edges for message passing.
        
        Fast global sampling - since edges only exist within graphs (never between graphs),
        we can sample globally without per-graph processing.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Full edge index of shape (2, num_edges).
        batch_indices : torch.Tensor
            Batch assignment for each node (unused, kept for API consistency).
        num_nodes : int
            Total number of nodes across all graphs (unused, kept for API consistency).
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (remaining_edge_index, pos_edge_index)
            - remaining_edge_index: Edges to keep for message passing
            - pos_edge_index: Edges removed (positive samples for prediction)
        """
        num_edges = edge_index.size(1)
        
        if num_edges == 0:
            return (
                torch.empty((2, 0), dtype=edge_index.dtype, device=device),
                torch.empty((2, 0), dtype=edge_index.dtype, device=device)
            )
        
        num_pos_samples = max(1, int(self.edge_sample_ratio * num_edges))
        
        perm = torch.randperm(num_edges, device=device)
        pos_indices = perm[:num_pos_samples]
        remain_indices = perm[num_pos_samples:]
        
        remaining_edge_index = edge_index[:, remain_indices]
        pos_edge_index = edge_index[:, pos_indices]
        
        return remaining_edge_index, pos_edge_index
    
    def sample_negative_edges(self, remaining_edge_index, num_pos_edges, batch_indices, num_nodes, device):
        """Sample negative edges (non-existent edges) within each graph.
        
        Parameters
        ----------
        remaining_edge_index : torch.Tensor
            The edges still in the graph after positive sampling.
        num_pos_edges : int
            Number of positive edges (to match with negatives).
        batch_indices : torch.Tensor
            Batch assignment for each node.
        num_nodes : int
            Total number of nodes.
        device : torch.device
            Device to use.
            
        Returns
        -------
        torch.Tensor
            Negative edge index of shape (2, num_neg_samples).
        """
        num_neg_samples = int(num_pos_edges * self.neg_sample_ratio)
        
        if num_neg_samples == 0:
            return torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)
        
        # Sample negative edges per graph
        batch_ids = batch_indices.unique()
        neg_edges_list = []
        
        # Calculate samples per graph proportionally
        samples_per_graph = {}
        total_nodes = 0
        for batch_id in batch_ids:
            graph_node_mask = (batch_indices == batch_id)
            num_graph_nodes = graph_node_mask.sum().item()
            samples_per_graph[batch_id.item()] = num_graph_nodes
            total_nodes += num_graph_nodes
        
        for batch_id in batch_ids:
            graph_node_mask = (batch_indices == batch_id)
            graph_nodes = torch.where(graph_node_mask)[0]
            num_graph_nodes = len(graph_nodes)
            
            if num_graph_nodes < 2:
                continue
            
            # Get edges for this graph
            src_in_graph = torch.isin(remaining_edge_index[0], graph_nodes)
            dst_in_graph = torch.isin(remaining_edge_index[1], graph_nodes)
            graph_edge_mask = src_in_graph & dst_in_graph
            graph_edges = remaining_edge_index[:, graph_edge_mask]
            
            # Calculate max possible negative edges FIRST
            # For directed graph: n*(n-1) - num_existing_edges
            max_possible_negs = num_graph_nodes * (num_graph_nodes - 1) - graph_edges.size(1)
            
            # Calculate number of negative samples for this graph
            graph_neg_samples = max(1, int(num_neg_samples * samples_per_graph[batch_id.item()] / total_nodes))
            
            # Cap at what's actually possible BEFORE any sampling
            graph_neg_samples = min(graph_neg_samples, max(1, max_possible_negs))
            
            # Create node index mapping BEFORE converting edges
            node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_mapping[graph_nodes] = torch.arange(num_graph_nodes, device=device)
            
            # Map edges to local indices - verify they're valid
            local_edges = node_mapping[graph_edges]
            
            # Verify no invalid indices before calling negative_sampling
            if (local_edges < 0).any() or (local_edges >= num_graph_nodes).any():
                # Fallback to random sampling
                src_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                dst_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                global_neg_edges = torch.stack([graph_nodes[src_idx], graph_nodes[dst_idx]], dim=0)
                neg_edges_list.append(global_neg_edges)
                continue
            
            try:
                
                local_neg_edges = negative_sampling(
                    edge_index=local_edges,
                    num_nodes=num_graph_nodes,
                    num_neg_samples=graph_neg_samples,
                    method=self.sampling_method,
                )
                
                # Validate indices are within bounds
                if (local_neg_edges >= num_graph_nodes).any() or (local_neg_edges < 0).any():
                    raise ValueError(f"Invalid local indices from negative_sampling")
                
                # Map back to global indices - index each row separately for safety
                # local_neg_edges is (2, num_samples), so index source and dest separately
                global_neg_edges = torch.stack([
                    graph_nodes[local_neg_edges[0]],  # Map source nodes
                    graph_nodes[local_neg_edges[1]]   # Map destination nodes
                ], dim=0)
                
                # Final validation of global indices before adding to list
                if (global_neg_edges >= num_nodes).any() or (global_neg_edges < 0).any():
                    raise ValueError(f"Global negative edges out of bounds!")
                
                neg_edges_list.append(global_neg_edges)
                
            except Exception as e:
                # Fallback: safe random sampling
                src_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                dst_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                # Ensure src != dst
                mask = src_idx != dst_idx
                if mask.sum() > 0:
                    global_neg_edges = torch.stack([graph_nodes[src_idx[mask]], graph_nodes[dst_idx[mask]]], dim=0)
                    neg_edges_list.append(global_neg_edges)
        
        if len(neg_edges_list) > 0:
            neg_edge_index = torch.cat(neg_edges_list, dim=1)
        else:
            neg_edge_index = torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)
        
        return neg_edge_index
    
    def forward(self, batch):
        r"""Forward pass for Link Prediction encoding.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Node embeddings from encoder (for downstream use)
            - pos_edge_index: Positive edges (removed edges)
            - neg_edge_index: Negative edges (sampled non-existent edges)
            - edge_index: Original edge index
            - batch_0: Batch indices
        """
        # Input features
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Sample positive edges (to remove) and keep remaining for message passing
        remaining_edge_index, pos_edge_index = self.sample_edges(
            edge_index, batch_indices, num_nodes, device
        )
        
        # Run GNN on remaining graph (without positive edges)
        node_embeddings = self.backbone(
            x_0,
            remaining_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight if edge_weight is not None else None,
        )
        
        # Sample negative edges
        neg_edge_index = self.sample_negative_edges(
            remaining_edge_index,
            pos_edge_index.size(1),
            batch_indices,
            num_nodes,
            device
        )
        
        # Prepare outputs for readout
        model_out = {
            "x_0": node_embeddings,  # Node embeddings (for downstream use)
            "pos_edge_index": pos_edge_index,  # Positive edges to predict
            "neg_edge_index": neg_edge_index,  # Negative edges to predict
            "edge_index": edge_index,  # Original full edge index
            "remaining_edge_index": remaining_edge_index,  # Edges used for message passing
            "batch_0": batch_indices,
            "labels": batch.y if hasattr(batch, 'y') else None,
        }
        
        return model_out

