"""Wrapper for DGMAE pre-training with GNN models.

DGMAE (Discrepancy-Aware Graph Mask Auto-Encoder) learns representations by:
1. Original feature reconstruction (like GraphMAE)
2. Feature discrepancy reconstruction (novel approach)

The key innovation is preserving the discrepancy between nodes and their neighbors
in the low-dimensional embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from topobench.nn.wrappers.base import AbstractWrapper


class DGMAEGNNWrapper(AbstractWrapper):
    r"""Wrapper for DGMAE pre-training with GNN models.

    DGMAE learns discriminative representations by reconstructing both:
    1. Original features (contextual information)
    2. Feature discrepancies (unique node information)

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
    lam : float, optional
        Weight for discrepancy loss (default: 0.5).
    p_c : float, optional
        Edge sampling probability for discrepancy (default: 0.5).
    p_tau : float, optional
        Cut-off probability to avoid over-sampling (default: 0.9).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_rate: float = 0.5,
        replace_rate: float = 0.0,
        drop_edge_rate: float = 0.0,
        lam: float = 0.5,
        p_c: float = 0.5,
        p_tau: float = 0.9,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.lam = lam
        self.p_c = p_c
        self.p_tau = p_tau
        
        # Calculate mask token rate
        self.mask_token_rate = 1 - self.replace_rate
        
        # Get feature dimension from kwargs
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
        
        # Create learnable mask token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.feature_dim))
        nn.init.xavier_normal_(self.enc_mask_token)
        
        # Projector for unmasked graph (used in discrepancy computation)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim * 2),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim * 2, self.feature_dim)
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
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, mask_nodes, keep_nodes
    
    def drop_edges(self, edge_index, num_edges, device):
        """Drop edges randomly."""
        if self.drop_edge_rate <= 0:
            return edge_index
        
        mask = torch.rand(num_edges, device=device) > self.drop_edge_rate
        new_edge_index = edge_index[:, mask]
        
        return new_edge_index
    
    def compute_attention_weights(self, x, edge_index):
        """Compute attention weights for edge sampling.
        
        Uses a simple attention mechanism to compute similarity between connected nodes.
        
        Parameters
        ----------
        x : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
            
        Returns
        -------
        torch.Tensor
            Attention weights of shape (num_edges,).
        """
        # Get source and target node embeddings
        src_nodes = edge_index[0]
        dst_nodes = edge_index[1]
        
        # Concatenate source and target embeddings
        edge_features = torch.cat([x[src_nodes], x[dst_nodes]], dim=-1)
        
        # Simple linear attention (could be replaced with GAT-style attention)
        # For simplicity, we'll use cosine similarity
        src_emb = F.normalize(x[src_nodes], p=2, dim=-1)
        dst_emb = F.normalize(x[dst_nodes], p=2, dim=-1)
        
        # Attention weights (higher = more similar)
        attention = (src_emb * dst_emb).sum(dim=-1)
        
        # Normalize to [0, 1]
        attention = (attention + 1) / 2
        
        return attention
    
    def adaptive_discrepancy_selection(self, edge_index, attention_weights):
        """Adaptively select edges with high discrepancy.
        
        Edges with low attention (dissimilar nodes) have higher probability of selection.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        attention_weights : torch.Tensor
            Attention weights of shape (num_edges,).
            
        Returns
        -------
        torch.Tensor
            Binary mask indicating selected edges (1 = selected).
        """
        # Compute sampling probability (Eq. 7)
        # Low attention = high discrepancy = higher probability
        discrepancy_prob = (1 - attention_weights) * self.p_c
        discrepancy_prob = torch.clamp(discrepancy_prob, max=self.p_tau)
        
        # Sample edges via Bernoulli (Eq. 8)
        mask = torch.bernoulli(discrepancy_prob).to(torch.bool)
        
        return mask
    
    def compute_raw_feature_discrepancy(self, x_raw, edge_index, discrepancy_mask):
        """Compute raw feature discrepancy for selected edges (Eq. 9).
        
        Parameters
        ----------
        x_raw : torch.Tensor
            Original raw node features of shape (num_nodes, num_features).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        discrepancy_mask : torch.Tensor
            Binary mask indicating which edges to compute discrepancy for.
            
        Returns
        -------
        torch.Tensor
            Feature discrepancy for each node of shape (num_nodes, num_features).
        """
        num_nodes = x_raw.size(0)
        device = x_raw.device
        
        # Get selected edges
        selected_edges = edge_index[:, discrepancy_mask]
        
        if selected_edges.size(1) == 0:
            # No edges selected, return zero discrepancy
            return torch.zeros_like(x_raw)
        
        # Compute degree normalization
        src = selected_edges[0]
        dst = selected_edges[1]
        
        # Count degree for normalization (symmetric normalization)
        degree_src = torch.bincount(src, minlength=num_nodes).float()
        degree_dst = torch.bincount(dst, minlength=num_nodes).float()
        
        # Avoid division by zero
        degree_src = torch.clamp(degree_src, min=1.0)
        degree_dst = torch.clamp(degree_dst, min=1.0)
        
        # Compute normalized discrepancy (Eq. 9)
        # x_D_i = sum_{j in N(i)} m_ij / sqrt(d_i * d_j) * (x_i - x_j)
        
        # Normalize factor
        norm_factor = 1.0 / (torch.sqrt(degree_src[src]) * torch.sqrt(degree_dst[dst]))
        
        # Compute difference
        diff = x_raw[src] - x_raw[dst]
        
        # Weight by normalization
        weighted_diff = diff * norm_factor.unsqueeze(-1)
        
        # Aggregate to nodes
        x_discrepancy = torch.zeros_like(x_raw)
        x_discrepancy.index_add_(0, src, weighted_diff)
        
        return x_discrepancy
    
    def forward(self, batch):
        r"""Forward pass for DGMAE encoding with masking and discrepancy computation.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing encoded representations and discrepancy components.
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Store ORIGINAL RAW features for reconstruction
        if hasattr(batch, 'x_raw'):
            x_raw_original = batch.x_raw.clone()
        else:
            # Fallback
            x_raw_original = x_0.clone()
        
        # ===== Branch 1: Original Feature Reconstruction (masked graph) =====
        
        # Apply masking to the ENCODED features
        masked_x, mask_nodes, keep_nodes = self.encoding_mask_noise(
            x_0, num_nodes, device
        )
        
        # Drop edges for masked encoding
        if self.training and self.drop_edge_rate > 0:
            use_edge_index = self.drop_edges(edge_index, edge_index.size(1), device)
        else:
            use_edge_index = edge_index
        
        # Encode with masked input
        enc_rep_masked = self.backbone(
            masked_x,
            use_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # ===== Branch 2: Feature Discrepancy Reconstruction (unmasked graph) =====
        
        # Encode clean input (no masking) for discrepancy computation
        enc_rep_clean = self.backbone(
            x_0,  # Clean encoded input
            edge_index,  # Full edges
            batch=batch_indices,
            edge_weight=edge_weight,
        )
        
        # Project clean embeddings for discrepancy computation
        z_clean = self.projector(enc_rep_clean)
        
        # Compute attention weights from clean embeddings (for adaptive selection)
        attention_weights = self.compute_attention_weights(enc_rep_clean, edge_index)
        
        # Adaptive discrepancy selection
        discrepancy_mask = self.adaptive_discrepancy_selection(edge_index, attention_weights)
        
        # Compute raw feature discrepancy (target for reconstruction)
        x_raw_discrepancy = self.compute_raw_feature_discrepancy(
            x_raw_original, edge_index, discrepancy_mask
        )
        
        # Compute embedding discrepancy (Eq. 11)
        # z_D_i = z_i - z_hat_i (difference between clean and masked embeddings)
        z_discrepancy = z_clean - enc_rep_masked
        
        # Prepare outputs for readout
        model_out = {
            # For original feature reconstruction
            "x_0": enc_rep_masked,  # Masked graph embeddings (for decoder input)
            "x_raw_original": x_raw_original,  # Original RAW features (reconstruction target)
            "mask_nodes": mask_nodes,  # Which nodes were masked
            "keep_nodes": keep_nodes,  # Which nodes were kept (for discrepancy reconstruction)
            
            # For discrepancy reconstruction
            "z_discrepancy": z_discrepancy,  # Embedding discrepancy (prediction)
            "x_raw_discrepancy": x_raw_discrepancy,  # Raw feature discrepancy (target)
            "discrepancy_mask": discrepancy_mask,  # Which edges were selected
            "attention_weights": attention_weights,  # For analysis
            
            # Metadata
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,
        }
        
        return model_out


