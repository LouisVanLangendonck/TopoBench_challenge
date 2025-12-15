"""Wrapper for GraphMAE pre-training with GNN models."""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class GraphMAEGNNWrapper(AbstractWrapper):
    r"""Wrapper for GraphMAE pre-training with GNN models.

    This wrapper implements the masking and encoding logic for GraphMAE
    self-supervised pre-training on graphs. The decoding/reconstruction
    is handled by the GraphMAEReadOut component.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model.
    mask_rate : float, optional
        The rate of nodes to mask (default: 0.5).
    replace_rate : float, optional
        The rate of masked nodes to replace with random features (default: 0.1).
    drop_edge_rate : float, optional
        The rate of edges to drop during training (default: 0.0).
    concat_hidden : bool, optional
        Whether to concatenate hidden layers (default: False).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        mask_rate: float = 0.5,
        replace_rate: float = 0.1,
        drop_edge_rate: float = 0.0,
        concat_hidden: bool = False,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.concat_hidden = concat_hidden
        
        # Calculate mask token rate (remaining after replace rate)
        self.mask_token_rate = 1 - self.replace_rate
        
        # Get feature dimension from kwargs (input dimension to the GNN encoder)
        # This is the dimension of batch.x_0 after feature encoding
        self.feature_dim = kwargs.get('out_channels', None)
        
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
        
        # Create learnable mask token (same dimension as input features)
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.feature_dim))
    
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
            (masked_x, mask_nodes, keep_nodes) where:
            - masked_x: Features with masking and noise applied
            - mask_nodes: Indices of masked nodes
            - keep_nodes: Indices of non-masked nodes
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
            
            # Split masked nodes into token nodes and noise nodes
            token_nodes = mask_nodes[perm_mask[:int(self.mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-num_noise_nodes:]]
            
            # Select random features to replace noise nodes
            noise_to_be_chosen = torch.randperm(num_nodes, device=device)[:num_noise_nodes]
            
            # Zero out token nodes
            out_x[token_nodes] = 0.0
            # Replace noise nodes with random features
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            # All masked nodes are token nodes
            out_x[mask_nodes] = 0.0
            token_nodes = mask_nodes
        
        # Add learnable mask token
        out_x[token_nodes] = out_x[token_nodes] + self.enc_mask_token
        
        return out_x, mask_nodes, keep_nodes
    
    def drop_edges(self, edge_index, num_edges, device):
        """Drop edges randomly.
        
        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
        num_edges : int
            Number of edges.
        device : torch.device
            Device to use.
            
        Returns
        -------
        torch.Tensor
            New edge indices with some edges dropped.
        """
        if self.drop_edge_rate <= 0:
            return edge_index
        
        # Create mask for edges to keep
        mask = torch.rand(num_edges, device=device) > self.drop_edge_rate
        
        # Filter edges
        new_edge_index = edge_index[:, mask]
        
        return new_edge_index
    
    def forward(self, batch):
        r"""Forward pass for GraphMAE encoding with masking.

        Masking is applied in BOTH training and validation for proper reconstruction evaluation.
        Training: Also applies edge dropping.
        Val/Test: No edge dropping, but masking is still applied.
        
        Each forward pass uses different random masks (controlled by global RNG seed for reproducibility).

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Encoded node features after GNN
            - x_original: Original input features (to reconstruct)
            - mask_nodes: Indices of randomly masked nodes
            - labels: Original labels (for compatibility)
            - batch_0: Batch assignment
        """
        # Input features (after all preprocessing/feature encoding)
        # This is what we will mask and reconstruct
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Store original input features for reconstruction
        # GraphMAE reconstructs these exact features at masked positions
        x_original = x_0.clone()
        
        # ALWAYS apply masking (both train and eval) for proper reconstruction evaluation
        # The randomness ensures different patterns each time (controlled by global RNG seed)
        masked_x, mask_nodes, keep_nodes = self.encoding_mask_noise(
            x_0, num_nodes, device
        )
        
        # Only drop edges during training
        if self.training and self.drop_edge_rate > 0:
            num_edges = edge_index.size(1)
            use_edge_index = self.drop_edges(edge_index, num_edges, device)
        else:
            use_edge_index = edge_index
        
        # Encode with backbone
        # Check if backbone returns hidden states (for concat_hidden)
        if self.concat_hidden and hasattr(self.backbone, 'return_hidden'):
            enc_rep = self.backbone(
                masked_x,
                use_edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
                return_hidden=True
            )
            # Handle case where backbone returns (output, hidden_states)
            if isinstance(enc_rep, tuple):
                enc_rep, all_hidden = enc_rep
                # Concatenate all hidden layers
                enc_rep = torch.cat(all_hidden, dim=1)
        else:
            enc_rep = self.backbone(
                masked_x,
                use_edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )
        
        # Prepare outputs for readout
        model_out = {
            "x_0": enc_rep,  # Encoded representations
            "x_original": x_original,  # Original features (for reconstruction)
            "mask_nodes": mask_nodes,  # Which nodes were masked
            "labels": batch.y if hasattr(batch, 'y') else None,  # For compatibility
            "batch_0": batch_indices,
        }
        
        return model_out

