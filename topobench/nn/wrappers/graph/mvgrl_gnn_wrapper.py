"""MVGRL (Multi-View Graph Representation Learning) wrapper.

Based on: https://github.com/kavehhassani/mvgrl
         https://github.com/dmlc/dgl/tree/master/examples/pytorch/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
https://arxiv.org/abs/2006.05582

This implementation matches the original paper's Equation 4 for graph pooling:
- JK-Net style: concatenates sum-pooled representations from ALL GCN layers
- Graph representations have dimension `num_layers * hidden_dim` before MLP projection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from topobench.nn.wrappers.base import AbstractWrapper


def add_self_loops(edge_index, edge_weight, num_nodes):
    """Add self-loops to edge_index and edge_weight.
    
    Matches original: adj + I
    """
    device = edge_index.device
    
    # Create self-loop edges: (0,0), (1,1), ..., (n-1, n-1)
    loop_index = torch.arange(0, num_nodes, dtype=edge_index.dtype, device=device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    
    # Concatenate with original edges
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    
    # Add weight 1.0 for self-loops
    if edge_weight is not None:
        loop_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    else:
        edge_weight = torch.ones(edge_index.size(1), device=device)
    
    return edge_index, edge_weight


def symmetric_normalize(edge_index, edge_weight, num_nodes):
    """Compute symmetric normalization: D^{-1/2} A D^{-1/2}.
    
    Matches original normalize_adj function.
    """
    row, col = edge_index
    
    # Compute degree
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    
    # Normalize: D^{-1/2} A D^{-1/2}
    normalized_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    
    return normalized_weight


class GCNLayer(nn.Module):
    """Single GCN layer matching original MVGRL implementation exactly.
    
    IMPORTANT: This layer does NOT perform D^{-1/2} normalization internally.
    The adjacency/diffusion weights must be pre-normalized before being passed.
    
    Original MVGRL uses: GraphConv(in_dim, out_dim, bias=False, norm=norm, activation=nn.PReLU())
    So NO bias is used in the GCN layers.
    
    Uses xavier_uniform initialization for weights (matches DGL GraphConv default).
    """
    
    def __init__(self, in_ft, out_ft):
        super().__init__()
        # Original uses bias=False in GraphConv
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        
        # Initialize weights with xavier_uniform (matches DGL GraphConv)
        nn.init.xavier_uniform_(self.fc.weight.data)
    
    def forward(self, feat, edge_index, edge_weight, num_nodes=None):
        """Forward pass with sparse adjacency (no internal normalization).
        
        Parameters
        ----------
        feat : torch.Tensor
            Node features (num_nodes, in_ft).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        edge_weight : torch.Tensor
            Pre-normalized edge weights (num_edges,). REQUIRED.
            For adjacency: should be D^{-1/2}(A+I)D^{-1/2} values.
            For diffusion: should be raw PPR/heat diffusion values.
        num_nodes : int, optional
            Number of nodes.
        """
        # Linear transformation: X' = XW
        feat = self.fc(feat)
        
        # Message passing: just aggregate with given weights (no normalization)
        # This matches original: out = adj @ feat
        if num_nodes is None:
            num_nodes = feat.size(0)
        
        row, col = edge_index
        
        # Aggregate: sum over neighbors weighted by edge_weight
        # Equivalent to sparse matrix multiply: adj @ feat
        out = scatter_add(feat[row] * edge_weight.unsqueeze(-1), col, dim=0, dim_size=num_nodes)
        
        # No bias (original uses bias=False)
        return self.act(out)


class MVGRLEncoder(nn.Module):
    """GCN encoder for MVGRL with JK-Net style graph pooling.
    
    Returns both node representations (from final layer) and graph representations
    (concatenation of sum-pooled representations from ALL layers).
    
    This matches the original MVGRL implementation exactly.
    """
    
    def __init__(self, in_ft, out_ft, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        # First layer: in_ft -> out_ft
        self.layers.append(GCNLayer(in_ft, out_ft))
        
        # Remaining layers: out_ft -> out_ft
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(out_ft, out_ft))
    
    def forward(self, feat, edge_index, batch_indices, edge_weight=None):
        """Forward pass returning both node and graph representations.
        
        Parameters
        ----------
        feat : torch.Tensor
            Node features (num_nodes, in_ft).
        edge_index : torch.Tensor
            Edge indices (2, num_edges).
        batch_indices : torch.Tensor
            Graph assignment for each node.
        edge_weight : torch.Tensor, optional
            Edge weights.
            
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
        # Sum pooling for first layer
        hg = scatter_add(h, batch_indices, dim=0, dim_size=num_graphs)
        
        # Remaining layers with JK-Net concatenation
        for idx in range(self.num_layers - 1):
            h = self.layers[idx + 1](h, edge_index, edge_weight, num_nodes)
            # Concatenate sum-pooled representation from this layer
            hg = torch.cat((hg, scatter_add(h, batch_indices, dim=0, dim_size=num_graphs)), dim=-1)
        
        return h, hg


class MLP(nn.Module):
    """MLP projection head (matches original MVGRL).
    
    Architecture: 3 Linear layers with PReLU activations + residual shortcut.
    Uses PyTorch default initialization (kaiming_uniform_).
    """
    
    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class MVGRLGNNWrapper(AbstractWrapper):
    r"""Wrapper for MVGRL pre-training with GNN models.

    MVGRL learns representations by contrasting node and graph encodings
    from two structural views:
    1. Adjacency matrix (local structure)
    2. Diffusion matrix (global structure, e.g., PPR) - precomputed via transform

    The model maximizes mutual information using Jensen-Shannon Divergence (JSD)
    estimator between:
    - Node representations from view 1 and graph representation from view 2
    - Node representations from view 2 and graph representation from view 1

    This implementation uses custom GCN encoders (not the backbone) to enable
    JK-Net style graph pooling as described in the paper's Equation 4:
    - Node representations: from final GCN layer
    - Graph representations: concatenation of sum-pooled reps from ALL layers

    Note: Diffusion edges must be precomputed using PPRDiffusion or HeatDiffusion
    transform in the data preprocessing pipeline.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (NOT USED - kept for TopoBench compatibility).
        MVGRL uses its own custom encoders for JK-Net pooling.
    num_layers : int, optional
        Number of GCN layers (default: 4).
    diff_edge_index_attr : str, optional
        Attribute name for precomputed diffusion edge indices (default: "edge_index_diff").
    diff_edge_weight_attr : str, optional
        Attribute name for precomputed diffusion edge weights (default: "edge_weight_diff").
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
        Must include 'out_channels' for hidden dimension.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_layers: int = 4,
        diff_edge_index_attr: str = "edge_index_diff",
        diff_edge_weight_attr: str = "edge_weight_diff",
        **kwargs
    ):
        # Disable residual connections since MVGRL uses custom encoders
        kwargs['residual_connections'] = False
        super().__init__(backbone, **kwargs)

        self.diff_edge_index_attr = diff_edge_index_attr
        self.diff_edge_weight_attr = diff_edge_weight_attr
        self.num_layers = num_layers

        # Get feature dimensions from kwargs
        self.hidden_dim = kwargs.get('out_channels', None)
        if self.hidden_dim is None:
            raise ValueError("Cannot determine hidden dimension. Please provide 'out_channels' in kwargs.")
        
        # Get input dimension - this should come from feature encoder
        in_channels = kwargs.get('in_channels', None)
        if in_channels is None:
            # Try to infer from backbone if available
            if hasattr(backbone, 'in_channels'):
                in_channels = backbone.in_channels
            else:
                in_channels = self.hidden_dim  # Fallback: assume already encoded
        self.in_channels = in_channels

        # Create custom MVGRL encoders (NOT using the backbone)
        # View 1: Adjacency encoder
        self.encoder1 = MVGRLEncoder(self.in_channels, self.hidden_dim, num_layers)
        # View 2: Diffusion encoder
        self.encoder2 = MVGRLEncoder(self.in_channels, self.hidden_dim, num_layers)
        
        # MLP projection heads (matches original MVGRL exactly)
        # For node-level representations: input is hidden_dim (from final layer)
        self.mlp_node = MLP(self.hidden_dim, self.hidden_dim)
        # For graph-level representations: input is num_layers * hidden_dim (JK-Net concat)
        self.mlp_graph = MLP(num_layers * self.hidden_dim, self.hidden_dim)

    def get_positive_expectation(self, p_samples, average=True):
        """Computes the positive part of JS Divergence.
        
        Matches original MVGRL implementation exactly.
        Formula: log(2) - softplus(-p_samples)
        
        Parameters
        ----------
        p_samples : torch.Tensor
            Positive samples (similarity scores for positive pairs).
        average : bool
            Whether to return mean or full tensor.
        """
        log_2 = math.log(2.0)
        Ep = log_2 - F.softplus(-p_samples)
        
        if average:
            return Ep.mean()
        return Ep

    def get_negative_expectation(self, q_samples, average=True):
        """Computes the negative part of JS Divergence.
        
        Matches original MVGRL implementation exactly.
        Formula: softplus(-q_samples) + q_samples - log(2)
        
        Parameters
        ----------
        q_samples : torch.Tensor
            Negative samples (similarity scores for negative pairs).
        average : bool
            Whether to return mean or full tensor.
        """
        log_2 = math.log(2.0)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        
        if average:
            return Eq.mean()
        return Eq

    def local_global_loss(self, l_enc, g_enc, batch_indices, num_graphs):
        """Compute local-global contrastive loss using JSD estimator.
        
        Matches original MVGRL implementation exactly.
        
        Parameters
        ----------
        l_enc : torch.Tensor
            Local (node) encodings after MLP projection (num_nodes, hidden_dim).
        g_enc : torch.Tensor
            Global (graph) encodings after MLP projection (num_graphs, hidden_dim).
        batch_indices : torch.Tensor
            Graph assignment for each node.
        num_graphs : int
            Number of graphs in batch.
            
        Returns
        -------
        torch.Tensor
            Scalar contrastive loss value.
        """
        num_nodes = l_enc.shape[0]
        device = l_enc.device
        
        # Create positive mask: pos_mask[i,j] = 1 if node i belongs to graph j
        # Create negative mask: neg_mask[i,j] = 1 if node i does NOT belong to graph j
        pos_mask = torch.zeros((num_nodes, num_graphs), device=device)
        neg_mask = torch.ones((num_nodes, num_graphs), device=device)
        
        for nodeidx, graphidx in enumerate(batch_indices):
            pos_mask[nodeidx][graphidx] = 1.0
            neg_mask[nodeidx][graphidx] = 0.0
        
        # Compute similarity scores: (num_nodes, num_graphs)
        res = torch.mm(l_enc, g_enc.t())
        
        # Compute positive expectation (only for positive pairs)
        E_pos = self.get_positive_expectation(res * pos_mask, average=False).sum()
        E_pos = E_pos / num_nodes
        
        # Compute negative expectation (only for negative pairs)
        E_neg = self.get_negative_expectation(res * neg_mask, average=False).sum()
        E_neg = E_neg / (num_nodes * (num_graphs - 1))
        
        return E_neg - E_pos

    def forward(self, batch):
        r"""Forward pass for MVGRL encoding with contrastive learning.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.
            Must have precomputed diffusion edges from PPRDiffusion or HeatDiffusion transform:
            - edge_index_diff: Precomputed diffusion edge indices
            - edge_weight_diff: Precomputed diffusion edge weights

        Returns
        -------
        dict
            Dictionary containing encoded representations and loss components.
        """
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        num_graphs = batch_indices.max().item() + 1
        device = x_0.device

        # Get precomputed diffusion edges
        if not hasattr(batch, self.diff_edge_index_attr) or getattr(batch, self.diff_edge_index_attr) is None:
            raise ValueError(
                f"MVGRL requires precomputed diffusion edges. "
                f"Attribute '{self.diff_edge_index_attr}' not found in batch. "
                f"Add PPRDiffusion or HeatDiffusion transform to your data preprocessing pipeline."
            )
        
        edge_index_diff = getattr(batch, self.diff_edge_index_attr)
        diff_weights = getattr(batch, self.diff_edge_weight_attr, None)
        
        # Ensure on correct device
        if edge_index_diff.device != device:
            edge_index_diff = edge_index_diff.to(device)
        if diff_weights is not None and diff_weights.device != device:
            diff_weights = diff_weights.to(device)

        # === View 1: Adjacency encoding ===
        # Original: adj = normalize_adj(adj + I) then gcn(feat, adj)
        # We add self-loops and compute D^{-1/2}(A+I)D^{-1/2} normalization
        adj_edge_index, adj_edge_weight = add_self_loops(edge_index, edge_weight, num_nodes)
        adj_edge_weight = symmetric_normalize(adj_edge_index, adj_edge_weight, num_nodes)
        # Returns: lv1 (nodes from final layer), gv1 (JK-Net concat graph rep)
        lv1, gv1 = self.encoder1(x_0, adj_edge_index, batch_indices, adj_edge_weight)
        
        # === View 2: Diffusion encoding ===
        # Original: gcn(feat, diff) where diff is raw PPR matrix (no additional normalization)
        # The PPR matrix already incorporates normalization in its formula: α(I - (1-α)Ã)^{-1}
        # Returns: lv2 (nodes from final layer), gv2 (JK-Net concat graph rep)
        lv2, gv2 = self.encoder2(x_0, edge_index_diff, batch_indices, diff_weights)

        # Apply MLP projection heads (matches original MVGRL exactly)
        # Node MLPs: hidden_dim -> hidden_dim
        lv1_proj = self.mlp_node(lv1)
        lv2_proj = self.mlp_node(lv2)
        # Graph MLPs: num_layers * hidden_dim -> hidden_dim
        gv1_proj = self.mlp_graph(gv1)
        gv2_proj = self.mlp_graph(gv2)

        # === Compute contrastive loss ===
        # Loss 1: node from view1 with graph from view2
        loss1 = self.local_global_loss(lv1_proj, gv2_proj, batch_indices, num_graphs)
        # Loss 2: node from view2 with graph from view1
        loss2 = self.local_global_loss(lv2_proj, gv1_proj, batch_indices, num_graphs)
        
        contrastive_loss = loss1 + loss2

        # Final embeddings: sum of projected graph representations (matches original)
        g_combined = gv1_proj + gv2_proj

        model_out = {
            "x_0": lv1,  # Node representations (for downstream node tasks)
            "x_graph": g_combined,  # Combined graph representations (for downstream graph tasks)
            "lv1": lv1,
            "lv2": lv2,
            "gv1": gv1,  # Before MLP (JK-Net concat)
            "gv2": gv2,  # Before MLP (JK-Net concat)
            "gv1_proj": gv1_proj,  # After MLP
            "gv2_proj": gv2_proj,  # After MLP
            "contrastive_loss": contrastive_loss,
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "num_graphs": num_graphs,
        }

        return model_out

    def embed(self, batch):
        """Get embeddings without loss computation (for inference).
        
        Returns graph-level embeddings (sum of both projected views).
        Matches original: (gv1 + gv2).detach() where gv1/gv2 are after MLP projection.
        """
        model_out = self.forward(batch)
        return model_out["x_graph"].detach()
