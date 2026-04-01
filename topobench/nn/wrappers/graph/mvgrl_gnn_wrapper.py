"""MVGRL (Multi-View Graph Representation Learning) unified wrapper.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
https://arxiv.org/abs/2006.05582

This unified implementation follows the paper's description exactly:
- Same encoder architecture for both node and graph tasks
- Same JK-Net style pooling
- Same JSD (dot product) discriminator
- Same MLP projection heads

The ONLY difference between inductive (graph) and transductive (node) settings:
- Inductive: negatives from other graphs in the batch
- Transductive: negatives from shuffled features within each graph
  + Subsampling: random windows of nodes treated as independent graphs

From the paper:
"To generate negative samples in transductive tasks, we randomly shuffle the features"
"This procedure allows our approach to be applied to inductive tasks...and also to
transductive tasks by considering sub-samples as independent graphs."

Hyperparameter recommendations from the paper:
- Node classification: num_layers=1 (following DGI)
- Graph classification: num_layers=2,4,8,12 (choose via validation)
"""

import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from topobench.nn.wrappers.base import AbstractWrapper
from topobench.nn.wrappers.graph.mvgrl_utils import (
    MLP,
    MVGRLGCNLayer,
    add_self_loops,
    symmetric_normalize,
)


class MVGRLEncoder(nn.Module):
    """GCN encoder for MVGRL with JK-Net style graph pooling.

    From the paper (Equation 4):
    "We use a readout function similar to jumping knowledge network (JK-Net)
    where we concatenate the summation of the node representations in each
    GCN layer and then feed them to a single layer feed-forward network"

    Returns both node representations (from final layer) and graph representations
    (concatenation of sum-pooled representations from ALL layers).

    Parameters
    ----------
    in_ft : int
        Input feature dimension.
    out_ft : int
        Output (hidden) feature dimension.
    num_layers : int
        Number of GCN layers.
    bias : bool
        Whether to use bias in GCN layers. Paper uses bias=False for graph-level.
    """

    def __init__(self, in_ft, out_ft, num_layers, bias=False):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        # First layer: in_ft -> out_ft
        self.layers.append(MVGRLGCNLayer(in_ft, out_ft, bias=bias))

        # Remaining layers: out_ft -> out_ft
        for _ in range(num_layers - 1):
            self.layers.append(MVGRLGCNLayer(out_ft, out_ft, bias=bias))

    def forward(self, feat, edge_index, batch_indices, edge_weight):
        """Forward pass returning both node and graph representations.

        Parameters
        ----------
        feat : torch.Tensor
            Node features (num_nodes, in_ft).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        batch_indices : torch.Tensor
            Graph assignment for each node.
        edge_weight : torch.Tensor
            Pre-normalized edge weights.

        Returns
        -------
        h : torch.Tensor
            Node representations from final layer (num_nodes, out_ft).
        hg : torch.Tensor
            Graph representations from JK-Net pooling (num_graphs, num_layers * out_ft).
        """
        num_nodes = feat.size(0)
        num_graphs = batch_indices.max().item() + 1

        # First layer
        h = self.layers[0](feat, edge_index, edge_weight, num_nodes)
        # Sum pooling for first layer (JK-Net style)
        hg = scatter_add(h, batch_indices, dim=0, dim_size=num_graphs)

        # Remaining layers with JK-Net concatenation
        for idx in range(self.num_layers - 1):
            h = self.layers[idx + 1](h, edge_index, edge_weight, num_nodes)
            # Concatenate sum-pooled representation from this layer
            hg = torch.cat(
                (
                    hg,
                    scatter_add(h, batch_indices, dim=0, dim_size=num_graphs),
                ),
                dim=-1,
            )

        return h, hg


class MVGRLWrapper(AbstractWrapper):
    r"""Unified wrapper for MVGRL pre-training (both node and graph tasks).

    MVGRL learns representations by contrasting node and graph encodings
    from two structural views:
    1. Adjacency matrix (local structure) with symmetric normalization
    2. Diffusion matrix (global structure, e.g., PPR) - precomputed via transform

    The model maximizes mutual information using Jensen-Shannon Divergence (JSD)
    estimator (dot product discriminator) between:
    - Node representations from view 1 and graph representation from view 2
    - Node representations from view 2 and graph representation from view 1

    This is a UNIFIED implementation following the paper exactly.
    The only difference between settings is how negatives are generated:
    - "cross_graph": Negatives from other graphs in batch (inductive/graph tasks)
    - "feature_shuffle": Negatives from shuffled features (transductive/node tasks)
      + With subsampling: random windows of nodes treated as independent graphs

    From the paper:
    "To generate negative samples in transductive tasks, we randomly shuffle the features"
    "considering sub-samples as independent graphs"

    For transductive (node) tasks, the wrapper implements window-based subsampling
    matching the original code: random contiguous windows of nodes are extracted
    and treated as independent graphs in a batch.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (NOT USED - kept for TopoBench compatibility).
    num_layers : int, optional
        Number of GCN layers (default: 1 for node tasks, use 2-12 for graph tasks).
    negative_sampling : str, optional
        How to generate negatives: "cross_graph" (inductive) or "feature_shuffle" (transductive).
    sample_size : int, optional
        Number of nodes per subgraph window (only for feature_shuffle). Default: 2000.
    subsample_batch_size : int, optional
        Number of subgraph windows per batch (only for feature_shuffle). Default: 4.
    diff_edge_index_attr : str, optional
        Attribute name for precomputed diffusion edge indices.
    diff_edge_weight_attr : str, optional
        Attribute name for precomputed diffusion edge weights.
    **kwargs : dict
        Additional arguments. Must include 'out_channels' for hidden dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_layers: int = 1,
        negative_sampling: str = "cross_graph",
        sample_size: int = 2000,
        subsample_batch_size: int = 4,
        diff_edge_index_attr: str = "edge_index_diff",
        diff_edge_weight_attr: str = "edge_weight_diff",
        **kwargs,
    ):
        kwargs["residual_connections"] = False
        super().__init__(backbone, **kwargs)

        if negative_sampling not in ["cross_graph", "feature_shuffle"]:
            raise ValueError(
                f"negative_sampling must be 'cross_graph' or 'feature_shuffle', got {negative_sampling}"
            )

        self.diff_edge_index_attr = diff_edge_index_attr
        self.diff_edge_weight_attr = diff_edge_weight_attr
        self.num_layers = num_layers
        self.negative_sampling = negative_sampling
        self.sample_size = sample_size
        self.subsample_batch_size = subsample_batch_size

        # Get feature dimensions
        self.hidden_dim = kwargs.get("out_channels")
        if self.hidden_dim is None:
            raise ValueError("Please provide 'out_channels' in kwargs.")

        in_channels = kwargs.get("in_channels")
        if in_channels is None:
            if hasattr(backbone, "in_channels"):
                in_channels = backbone.in_channels
            else:
                in_channels = self.hidden_dim
        self.in_channels = in_channels

        # Create MVGRL encoders (same architecture for both settings)
        # Paper uses bias=False for graph-level tasks
        self.encoder1 = MVGRLEncoder(
            self.in_channels, self.hidden_dim, num_layers, bias=False
        )
        self.encoder2 = MVGRLEncoder(
            self.in_channels, self.hidden_dim, num_layers, bias=False
        )

        # MLP projection heads (from paper Section 3.2)
        # "The learned representations are then fed into a shared projection head
        #  fψ(.) which is an MLP with two hidden layers"
        self.mlp_node = MLP(self.hidden_dim, self.hidden_dim)
        self.mlp_graph = MLP(num_layers * self.hidden_dim, self.hidden_dim)

    def shuffle_features(self, x, batch_indices):
        """Shuffle node features within each graph for transductive negative sampling.

        From the paper: "To generate negative samples in transductive tasks,
        we randomly shuffle the features"

        Parameters
        ----------
        x : torch.Tensor
            Node features (num_nodes, num_features).
        batch_indices : torch.Tensor
            Graph assignment for each node.

        Returns
        -------
        torch.Tensor
            Shuffled node features.
        """
        batch_ids = batch_indices.unique()
        shuffled_x = x.clone()

        for batch_id in batch_ids:
            graph_mask = batch_indices == batch_id
            graph_indices = torch.where(graph_mask)[0]
            perm = torch.randperm(len(graph_indices), device=x.device)
            shuffled_indices = graph_indices[perm]
            shuffled_x[graph_indices] = x[shuffled_indices]

        return shuffled_x

    def subsample_graph(
        self,
        x,
        edge_index,
        edge_weight,
        edge_index_diff,
        diff_weights,
        num_nodes,
    ):
        """Subsample the graph into multiple windows for transductive training.

        From the original MVGRL code (node/train.py):
        ```
        sample_size = 2000
        batch_size = 4
        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])
        ```

        This extracts contiguous windows of nodes and treats them as independent graphs.

        Parameters
        ----------
        x : torch.Tensor
            Node features (num_nodes, num_features).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        edge_weight : torch.Tensor or None
            Edge weights.
        edge_index_diff : torch.Tensor
            Diffusion edge indices (2, num_diff_edges).
        diff_weights : torch.Tensor
            Diffusion edge weights.
        num_nodes : int
            Total number of nodes.

        Returns
        -------
        tuple
            (subsampled_x, subsampled_edge_index, subsampled_edge_weight,
             subsampled_edge_index_diff, subsampled_diff_weights, batch_indices)
        """
        device = x.device
        sample_size = min(self.sample_size, num_nodes)
        batch_size = self.subsample_batch_size

        # If graph is smaller than sample_size, just use the whole graph
        if num_nodes <= sample_size:
            batch_indices = torch.zeros(
                num_nodes, dtype=torch.long, device=device
            )
            return (
                x,
                edge_index,
                edge_weight,
                edge_index_diff,
                diff_weights,
                batch_indices,
            )

        # Random starting indices for each window
        max_start = num_nodes - sample_size
        start_indices = np.random.randint(0, max_start + 1, batch_size)

        all_features = []
        all_adj_edges = []
        all_adj_weights = []
        all_diff_edges = []
        all_diff_weights = []
        all_batch_indices = []

        node_offset = 0

        for batch_idx, start in enumerate(start_indices):
            end = start + sample_size

            # Node mask for this window
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            node_mask[start:end] = True

            # Extract features for this window
            window_features = x[start:end]
            all_features.append(window_features)

            # Create batch indices for this window
            window_batch = torch.full(
                (sample_size,), batch_idx, dtype=torch.long, device=device
            )
            all_batch_indices.append(window_batch)

            # Filter adjacency edges: both endpoints must be in window
            row, col = edge_index
            adj_mask = (
                (row >= start) & (row < end) & (col >= start) & (col < end)
            )
            window_adj_edges = (
                edge_index[:, adj_mask] - start + node_offset
            )  # Reindex
            all_adj_edges.append(window_adj_edges)

            if edge_weight is not None:
                all_adj_weights.append(edge_weight[adj_mask])

            # Filter diffusion edges
            row_diff, col_diff = edge_index_diff
            diff_mask = (
                (row_diff >= start)
                & (row_diff < end)
                & (col_diff >= start)
                & (col_diff < end)
            )
            window_diff_edges = (
                edge_index_diff[:, diff_mask] - start + node_offset
            )  # Reindex
            all_diff_edges.append(window_diff_edges)

            if diff_weights is not None:
                all_diff_weights.append(diff_weights[diff_mask])

            node_offset += sample_size

        # Concatenate all windows
        subsampled_x = torch.cat(all_features, dim=0)
        subsampled_edge_index = torch.cat(all_adj_edges, dim=1)
        subsampled_edge_index_diff = torch.cat(all_diff_edges, dim=1)
        batch_indices = torch.cat(all_batch_indices, dim=0)

        subsampled_edge_weight = None
        if edge_weight is not None and all_adj_weights:
            subsampled_edge_weight = torch.cat(all_adj_weights, dim=0)

        subsampled_diff_weights = None
        if diff_weights is not None and all_diff_weights:
            subsampled_diff_weights = torch.cat(all_diff_weights, dim=0)

        return (
            subsampled_x,
            subsampled_edge_index,
            subsampled_edge_weight,
            subsampled_edge_index_diff,
            subsampled_diff_weights,
            batch_indices,
        )

    def forward(self, batch):
        r"""Forward pass for MVGRL with contrastive learning.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object with precomputed diffusion edges.

        Returns
        -------
        dict
            Dictionary containing representations and contrastive loss.
        """
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        num_nodes_original = x_0.size(0)
        device = x_0.device

        # Get precomputed diffusion edges
        if (
            not hasattr(batch, self.diff_edge_index_attr)
            or getattr(batch, self.diff_edge_index_attr) is None
        ):
            raise ValueError(
                "MVGRL requires precomputed diffusion edges. "
                "Add PPRDiffusion or HeatDiffusion transform to your pipeline."
            )

        edge_index_diff = getattr(batch, self.diff_edge_index_attr)
        diff_weights = getattr(batch, self.diff_edge_weight_attr, None)

        if edge_index_diff.device != device:
            edge_index_diff = edge_index_diff.to(device)
        if diff_weights is not None and diff_weights.device != device:
            diff_weights = diff_weights.to(device)

        # === For transductive (feature_shuffle), apply subsampling during training ===
        if self.negative_sampling == "feature_shuffle" and self.training:
            # Subsample the graph into multiple windows
            (
                x_0,
                edge_index,
                edge_weight,
                edge_index_diff,
                diff_weights,
                batch_indices,
            ) = self.subsample_graph(
                x_0,
                edge_index,
                edge_weight,
                edge_index_diff,
                diff_weights,
                num_nodes_original,
            )

        num_nodes = x_0.size(0)
        num_graphs = batch_indices.max().item() + 1

        # Prepare adjacency with normalization
        adj_edge_index, adj_edge_weight = add_self_loops(
            edge_index, edge_weight, num_nodes
        )
        adj_edge_weight = symmetric_normalize(
            adj_edge_index, adj_edge_weight, num_nodes
        )

        # === Encode real features through both views ===
        lv1, gv1 = self.encoder1(
            x_0, adj_edge_index, batch_indices, adj_edge_weight
        )
        lv2, gv2 = self.encoder2(
            x_0, edge_index_diff, batch_indices, diff_weights
        )

        # Apply MLP projections
        lv1_proj = self.mlp_node(lv1)
        lv2_proj = self.mlp_node(lv2)
        gv1_proj = self.mlp_graph(gv1)
        gv2_proj = self.mlp_graph(gv2)

        # Final embeddings (paper: sum of both views)
        h_combined = lv1 + lv2  # Node representations
        g_combined = gv1_proj + gv2_proj  # Graph representations

        model_out = {
            "x_0": h_combined,  # For node-level downstream tasks
            "x_graph": g_combined,  # For graph-level downstream tasks
            "lv1": lv1,
            "lv2": lv2,
            "lv1_proj": lv1_proj,  # For loss computation in MVGRLLoss
            "lv2_proj": lv2_proj,  # For loss computation in MVGRLLoss
            "gv1_proj": gv1_proj,
            "gv2_proj": gv2_proj,
            "negative_sampling": self.negative_sampling,  # For loss module to know which mode
            "labels": batch.y if hasattr(batch, "y") else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_graphs": num_graphs,
            "num_nodes_original": num_nodes_original,  # Track original size for inference
        }

        # For transductive mode, compute shuffled embeddings for loss module
        if self.negative_sampling == "feature_shuffle":
            x_shuffled = self.shuffle_features(x_0, batch_indices)

            # Encode shuffled features
            lv1_shuf, _ = self.encoder1(
                x_shuffled, adj_edge_index, batch_indices, adj_edge_weight
            )
            lv2_shuf, _ = self.encoder2(
                x_shuffled, edge_index_diff, batch_indices, diff_weights
            )

            # Project shuffled embeddings - passed to loss module
            model_out["lv1_shuf_proj"] = self.mlp_node(lv1_shuf)
            model_out["lv2_shuf_proj"] = self.mlp_node(lv2_shuf)

        return model_out

    def embed(self, batch):
        """Get embeddings without loss computation (for inference).

        Note: For transductive mode, this runs on the FULL graph (no subsampling)
        since we want embeddings for all nodes.
        """
        # Ensure we're in eval mode (no subsampling)
        was_training = self.training
        self.eval()

        with torch.no_grad():
            model_out = self.forward(batch)

        if was_training:
            self.train()

        if self.negative_sampling == "feature_shuffle":
            return model_out["x_0"].detach()  # Node embeddings for node tasks
        else:
            return model_out[
                "x_graph"
            ].detach()  # Graph embeddings for graph tasks


# Aliases for backward compatibility
MVGRLGNNWrapper = MVGRLWrapper
MVGRLInductiveWrapper = MVGRLWrapper
MVGRLTransductiveWrapper = MVGRLWrapper
