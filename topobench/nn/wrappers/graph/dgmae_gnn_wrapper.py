"""DGMAE (Discrepancy-aware Graph Masked Auto-Encoder) wrapper.

Based on: https://github.com/zhengziyu77/DGMAE
Paper: "Discrepancy-Aware Graph Mask Auto-Encoder" (KDD 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops
from torch_geometric.nn.dense.linear import Linear

from topobench.nn.wrappers.base import AbstractWrapper


def sce_loss(x, y, alpha=3):
    """Scaled Cosine Error loss for reconstruction."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class DGMAEGNNWrapper(AbstractWrapper):
    r"""Wrapper for DGMAE pre-training with GNN models.

    DGMAE (Discrepancy-aware Graph Masked Auto-Encoder) extends GraphMAE with:
    1. Heterophily-aware high-pass filtering
    2. MLP for predicting high-frequency component
    3. Discrepancy loss between reconstruction and high-freq prediction

    Attention-Weighted Edge Selection (for GAT encoders):
    ------------------------------------------------------
    When `encoder_type="gat"`, the wrapper extracts attention weights from the
    GAT encoder's MASKED pass (not clean!) and uses them for adaptive edge selection.
    This matches the original DGMAE implementation where encode_attn comes from
    encoder(use_g, use_x) with masked features.
    
    - weights_hp = 1 - sigmoid(attention)
    - Low attention (dissimilar nodes) → high weight → likely KEPT
    - High attention (similar nodes) → low weight → likely DROPPED
    
    This keeps heterophilic edges and drops homophilic ones for the high-pass
    filter, matching the original DGMAE paper and code.
    
    For non-GAT encoders (GCN, GIN, etc.), the full graph Laplacian is used
    for high-pass filtering (standard behavior).

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    mask_rate : float, optional
        The rate of nodes to mask (default: 0.5).
    replace_rate : float, optional
        The rate of masked nodes to replace with random features (default: 0.0).
    drop_edge_rate : float, optional
        The rate of edges to drop during training (default: 0.0).
    hop : int, optional
        Number of hops for high-pass filter (default: 2).
    in_channels : int, optional
        Input feature dimension for MLP output (default: None, inferred from out_channels).
    encoder_type : str, optional
        Type of encoder: "gat", "gcn", "gin", etc. (default: "gcn").
        If "gat", attention weights are used for edge selection.
    attn_edge_p : float, optional
        Base probability for attention-weighted edge selection (default: 0.3).
    attn_edge_threshold : float, optional
        Maximum probability threshold for edge selection (default: 0.7).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_rate: float = 0.5,
        replace_rate: float = 0.0,
        drop_edge_rate: float = 0.0,
        hop: int = 2,
        in_channels: int = None,
        encoder_type: str = "gcn",
        attn_edge_p: float = 0.3,
        attn_edge_threshold: float = 0.7,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)

        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.hop = hop
        self.encoder_type = encoder_type.lower()
        self.attn_edge_p = attn_edge_p
        self.attn_edge_threshold = attn_edge_threshold
        
        # Whether to use attention-weighted edge selection
        self.use_attention = self.encoder_type in ("gat", "dotgat")

        # Calculate mask token rate
        self.mask_token_rate = 1 - self.replace_rate

        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)

        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")

        # Input dimension for MLP output (original feature space)
        self.in_channels = in_channels if in_channels is not None else self.feature_dim

        # Create learnable mask token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.feature_dim))

        # MLP for high-frequency prediction (DGMAE-specific)
        # Maps: hidden_dim -> in_dim (to match original feature space)
        # Using PyG's Linear with glorot initialization (matches original)
        self.hetero_mlp = nn.Sequential(
            Linear(self.feature_dim, self.feature_dim, bias=False, weight_initializer='glorot'),
            nn.PReLU(),
            nn.Dropout(0.2),
            Linear(self.feature_dim, self.in_channels, bias=False, weight_initializer='glorot'),
        )

    def encoding_mask_noise(self, x, num_nodes, device):
        """Apply masking and noise to node features.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.

        Returns
        -------
        tuple
            (masked_x, mask_nodes, keep_nodes)
        """
        perm = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(self.mask_rate * num_nodes)

        # Split into masked and kept nodes
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # Clone features
        out_x = x.clone()

        if self.replace_rate > 0:
            # Some masked nodes get replaced with random features
            num_noise_nodes = int(self.replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=device)

            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]

            noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]

            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x[mask_nodes] = 0.0
            token_nodes = mask_nodes

        # Add learnable mask token
        out_x[token_nodes] += self.enc_mask_token

        return out_x, mask_nodes, keep_nodes

    def drop_edges(self, edge_index, num_nodes, device, return_edges=False):
        """Drop edges randomly and add self-loops.
        
        Matches original DGMAE implementation:
        1. Bernoulli sampling to decide which edges to keep
        2. Add self-loops to the resulting graph
        3. Optionally return dropped edges
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        num_nodes : int
            Number of nodes (needed for add_self_loops).
        device : torch.device
            Device to use.
        return_edges : bool, optional
            If True, also return the dropped edges (default: False).
            
        Returns
        -------
        torch.Tensor or tuple
            New edge_index with self-loops, or (new_edge_index, dropped_edges) if return_edges=True.
        """
        if self.drop_edge_rate <= 0:
            if return_edges:
                return edge_index, None
            return edge_index

        num_edges = edge_index.size(1)
        
        # Bernoulli mask: 1 = keep, 0 = drop
        # P(keep) = 1 - drop_rate
        mask_rates = torch.ones(num_edges, device=device) * self.drop_edge_rate
        keep_mask = torch.bernoulli(1 - mask_rates).bool()
        
        # Keep edges
        new_edge_index = edge_index[:, keep_mask]
        
        # Add self-loops (as in original DGMAE)
        new_edge_index, _ = add_self_loops(new_edge_index, num_nodes=num_nodes)
        
        if return_edges:
            # Return dropped edges as well
            drop_mask = ~keep_mask
            dropped_edges = edge_index[:, drop_mask]
            return new_edge_index, dropped_edges
        
        return new_edge_index

    def drop_edge_weighted(self, edge_index, edge_weights, p, threshold, device):
        """Attention-weighted edge selection for heterophily.
        
        Matches original DGMAE `drop_edge_weighted` from utils.py:
        - Normalizes weights by mean and scales by p
        - Clips to threshold
        - Bernoulli samples based on weights
        - High weight = more likely to be SELECTED (kept for high-pass filter)
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        edge_weights : torch.Tensor
            Edge weights (heterophily weights: 1 - sigmoid(attention)).
        p : float
            Base selection probability.
        threshold : float
            Maximum selection probability.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (selected_edges, dropped_edges, selected_weights)
        """
        # Normalize weights and scale by p
        edge_weights = edge_weights / (edge_weights.mean() + 1e-8) * p
        
        # Clip to threshold
        edge_weights = torch.where(
            edge_weights < threshold,
            edge_weights,
            torch.ones_like(edge_weights) * threshold
        )
        
        # Bernoulli sample: higher weight = more likely selected
        sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
        
        selected_edges = edge_index[:, sel_mask]
        dropped_edges = edge_index[:, ~sel_mask]
        selected_weights = edge_weights[sel_mask]
        
        return selected_edges, dropped_edges, selected_weights

    def compute_high_pass_features(self, edge_index, x, hop, num_nodes, device):
        """Compute high-pass filtered features using Laplacian.
        
        Implements heterophily_highfilter_sp from original DGMAE:
        L = I - A_norm (Laplacian)
        hx = L^hop * x (high-frequency component)
        
        Note: The original DGMAE uses attention-weighted edge dropping before
        this step when using GAT encoder. For non-GAT encoders, it uses the
        full graph (which is what this implementation does).
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        hop : int
            Number of hops for the filter.
        num_nodes : int
            Number of nodes.
        device : torch.device
            Device to use.
            
        Returns
        -------
        torch.Tensor
            High-pass filtered features.
        """
        row, col = edge_index
        
        # Compute degree
        deg = scatter_add(torch.ones(row.size(0), device=device), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Normalized edge weights: D^-0.5 * A * D^-0.5
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Apply Laplacian filter iteratively: L = I - A_norm
        hx = x.clone()
        for _ in range(hop):
            # A_norm * x via scatter
            ax = scatter_add(
                norm_weights.unsqueeze(-1) * hx[row],
                col, dim=0, dim_size=num_nodes
            )
            # L * x = x - A_norm * x
            hx = hx - ax
        
        return hx

    def forward(self, batch):
        r"""Forward pass for DGMAE encoding with masking and heterophily components.

        IMPORTANT: Following original DGMAE, we do TWO encoder passes:
        1. CLEAN pass: encoder(x) → for MLP prediction
        2. MASKED pass: encoder(masked_x) → for reconstruction + attention extraction

        For GAT encoders (matching original DGMAE):
        - Attention weights are extracted from the MASKED pass (not clean!)
        - This matches original: encode_attn comes from encoder(use_g, use_x)
          where use_x has masked features
        - Attention-weighted edge selection is used for high-pass filter
        
        For non-GAT encoders:
        - Full graph Laplacian is used for high-pass filter

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing encoded representations, masking info,
            and heterophily-related outputs.
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        device = x_0.device

        # Store ORIGINAL RAW features for reconstruction and high-pass filter
        if hasattr(batch, 'x_raw'):
            x_raw_original = batch.x_raw.clone()
        else:
            # Fallback
            x_raw_original = x_0.clone()

        # Apply masking to the ENCODED features (x_0)
        masked_x, mask_nodes, keep_nodes = self.encoding_mask_noise(
            x_0, num_nodes, device
        )

        # Drop edges (training only) - adds self-loops as in original DGMAE
        if self.training and self.drop_edge_rate > 0:
            use_edge_index = self.drop_edges(edge_index, num_nodes, device)
        else:
            use_edge_index = edge_index

        # === DGMAE: Two encoder passes (following original implementation) ===
        
        # Pass 1: CLEAN encoding (for MLP prediction only, no attention needed)
        enc_rep_clean = self.backbone(
            x_0,
            edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Pass 2: MASKED encoding (for reconstruction)
        # For GAT: extract attention weights from THIS pass (matches original DGMAE!)
        # Original code: encode_attn comes from encoder(use_g, use_x) where use_x is masked
        encode_attn = None
        
        if self.use_attention:
            # Try to get attention weights from GAT backbone on MASKED features
            try:
                enc_rep_masked, attn_output = self.backbone(
                    masked_x,
                    use_edge_index,
                    batch=batch_indices,
                    edge_weight=edge_weight,
                    return_attention_weights=True,
                )
                # attn_output is typically (edge_index, attention_weights)
                if isinstance(attn_output, tuple) and len(attn_output) == 2:
                    _, encode_attn = attn_output
                elif isinstance(attn_output, torch.Tensor):
                    encode_attn = attn_output
            except (TypeError, RuntimeError):
                # Backbone doesn't support attention extraction
                # Fall back to standard encoding
                enc_rep_masked = self.backbone(
                    masked_x,
                    use_edge_index,
                    batch=batch_indices,
                    edge_weight=edge_weight,
                )
                encode_attn = None
        else:
            # Non-GAT encoder: standard forward pass
            enc_rep_masked = self.backbone(
                masked_x,
                use_edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )

        # DGMAE-specific: MLP prediction uses CLEAN encoding
        high_pred = self.hetero_mlp(enc_rep_clean)

        # DGMAE-specific: Compute high-pass filtered features
        # For GAT: use attention-weighted edge selection
        # For non-GAT: use full graph Laplacian
        
        if encode_attn is not None and self.training:
            # GAT with attention: adaptive edge selection
            # Original: weights_hp = 1 - sigmoid(mean(attention))
            if encode_attn.dim() > 1:
                # Multi-head attention: average over heads
                attn_mean = encode_attn.mean(dim=-1)
            else:
                attn_mean = encode_attn
            
            # Convert to heterophily weights: low attention = high weight
            weights_lp = torch.sigmoid(attn_mean)
            weights_hp = 1 - weights_lp
            
            # Select edges based on heterophily weights
            selected_edges, _, _ = self.drop_edge_weighted(
                edge_index, weights_hp, 
                self.attn_edge_p, self.attn_edge_threshold, device
            )
            
            # Compute high-pass filter on SELECTED edges
            high_pass_features = self.compute_high_pass_features(
                selected_edges, x_raw_original, self.hop, num_nodes, device
            )
        else:
            # Non-GAT or eval mode: use full graph Laplacian
            high_pass_features = self.compute_high_pass_features(
                edge_index, x_raw_original, self.hop, num_nodes, device
            )

        # Prepare outputs for readout
        model_out = {
            "x_0": enc_rep_masked,  # MASKED encoding for reconstruction
            "x_raw_original": x_raw_original,  # ORIGINAL RAW features
            "mask_nodes": mask_nodes,  # Which nodes were masked
            "keep_nodes": keep_nodes,  # Which nodes were kept
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,  # Full edge_index for decoder
            "edge_weight": edge_weight,
            # DGMAE-specific outputs
            "high_pred": high_pred,  # MLP prediction (from CLEAN encoding)
            "high_pass_features": high_pass_features,  # High-pass filtered features
            "hop": self.hop,
            "encoder_type": self.encoder_type,  # For debugging/logging
        }

        return model_out
