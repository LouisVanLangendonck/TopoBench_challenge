"""Shared utilities for MVGRL implementations.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
https://arxiv.org/abs/2006.05582

This module contains the core building blocks used by the unified MVGRLWrapper.
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add


def add_self_loops(edge_index, edge_weight, num_nodes):
    """Add self-loops to edge_index and edge_weight.

    Matches original MVGRL: adj + I

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices (2, num_edges).
    edge_weight : torch.Tensor or None
        Edge weights (num_edges,).
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        (edge_index, edge_weight) with self-loops added.
    """
    device = edge_index.device

    loop_index = torch.arange(
        0, num_nodes, dtype=edge_index.dtype, device=device
    )
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index, loop_index], dim=1)

    if edge_weight is not None:
        loop_weight = torch.ones(
            num_nodes, dtype=edge_weight.dtype, device=device
        )
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
    else:
        edge_weight = torch.ones(edge_index.size(1), device=device)

    return edge_index, edge_weight


def symmetric_normalize(edge_index, edge_weight, num_nodes):
    """Compute symmetric normalization: D^{-1/2} A D^{-1/2}.

    Matches original MVGRL normalize_adj function.
    """
    row, col = edge_index

    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    normalized_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return normalized_weight


class MVGRLGCNLayer(nn.Module):
    """GCN layer for MVGRL.

    This layer does NOT perform D^{-1/2} normalization internally.
    The adjacency/diffusion weights must be pre-normalized before being passed.

    Uses xavier_uniform initialization for weights.

    Parameters
    ----------
    in_ft : int
        Input feature dimension.
    out_ft : int
        Output feature dimension.
    bias : bool
        Whether to use bias (default: False, matching paper).
    """

    def __init__(self, in_ft, out_ft, bias=False):
        super().__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, feat, edge_index, edge_weight, num_nodes=None):
        """Forward pass with sparse adjacency (no internal normalization)."""
        feat = self.fc(feat)

        if num_nodes is None:
            num_nodes = feat.size(0)

        row, col = edge_index
        out = scatter_add(
            feat[row] * edge_weight.unsqueeze(-1),
            col,
            dim=0,
            dim_size=num_nodes,
        )

        if self.bias is not None:
            out = out + self.bias

        return self.act(out)


class MLP(nn.Module):
    """MLP projection head matching original MVGRL.

    From the paper (Section 3.2):
    "The learned representations are then fed into a shared projection head
    fψ(.) which is an MLP with two hidden layers and PReLU non-linearity"

    Architecture: 3 Linear layers with PReLU activations + residual shortcut.
    """

    def __init__(self, in_ft, out_ft):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)
