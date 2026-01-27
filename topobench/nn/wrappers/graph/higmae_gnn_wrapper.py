"""Wrapper for Hi-GMAE (Hierarchical Graph Masked Autoencoder) pre-training with GNN models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
import numpy as np

from topobench.nn.wrappers.base import AbstractWrapper


def create_hierarchical_structure(edge_index, num_nodes, coarsening_ratio=0.5, device='cpu'):
    """
    Create hierarchical projection using FAST random coarsening.
    
    NOTE: Switched back to random coarsening for speed.
    Structure-aware coarsening was too slow (CPU bottleneck).
    
    Parameters
    ----------
    edge_index : torch.Tensor
        Edge indices of shape [2, num_edges]
    num_nodes : int
        Number of nodes in the graph
    coarsening_ratio : float
        Target ratio for coarsening (0.5 means aim for half the nodes)
    device : str
        Device to use
        
    Returns
    -------
    proj_matrix : torch.Tensor
        Projection matrix from fine to coarse level [num_coarse_nodes, num_nodes]
    """
    # FAST: Simple random clustering (vectorized)
    num_coarse_nodes = max(int(num_nodes * coarsening_ratio), 1)
    
    # Random assignment of nodes to clusters (fully on GPU)
    cluster_assignments = torch.randint(0, num_coarse_nodes, (num_nodes,), device=device)
    
    # Create projection matrix (vectorized)
    proj_matrix = torch.zeros(num_coarse_nodes, num_nodes, device=device)
    proj_matrix[cluster_assignments, torch.arange(num_nodes, device=device)] = 1.0
    
    # Normalize: each row (coarse node) is normalized by sqrt of cluster size
    row_sums = proj_matrix.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0  # Avoid division by zero
    proj_matrix = proj_matrix / row_sums.sqrt()
    
    return proj_matrix


class HiGMAEGNNWrapper(AbstractWrapper):
    r"""Wrapper for Hi-GMAE (Hierarchical Graph Masked Autoencoder) pre-training.

    Hi-GMAE creates a hierarchy of coarsened graphs and performs masked autoencoding
    at multiple levels with skip connections during decoding.

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (should be a list of encoders for each level).
    num_levels : int, optional
        Number of hierarchical levels (default: 2).
    mask_rate : float, optional
        The rate of nodes to mask at each level (default: 0.25).
    coarsening_ratio : float, optional
        Ratio for graph coarsening at each level (default: 0.5).
    recover_rate : float, optional
        Rate at which masked nodes are recovered during training (default: 0.0).
    gamma : float, optional
        Decay factor for recover_rate over epochs (default: 0.5).
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_levels: int = 2,
        mask_rate: float = 0.25,
        coarsening_ratio: float = 0.5,
        recover_rate: float = 0.0,
        gamma: float = 0.5,
        **kwargs
    ):
        super().__init__(backbone, **kwargs)
        
        self.num_levels = num_levels
        self.mask_rate = mask_rate
        self.coarsening_ratio = coarsening_ratio
        self.recover_rate = recover_rate
        self.gamma = gamma
        
        # Get feature dimension
        self.feature_dim = kwargs.get('out_channels', None)
        if self.feature_dim is None:
            raise ValueError("Cannot determine feature dimension. Please provide 'out_channels' in kwargs.")
        
        # Create learnable mask token
        self.enc_mask_token = nn.Parameter(torch.zeros(1, self.feature_dim))
        nn.init.xavier_normal_(self.enc_mask_token)
        
        # Coarse-level processing layers (simple MLP with LayerNorm for stability)
        # These add learnable transformations at coarse levels beyond just projection
        self.coarse_layers = nn.ModuleList()
        for _ in range(num_levels - 1):
            coarse_layer = nn.Sequential(
                nn.LayerNorm(self.feature_dim),
                nn.Linear(self.feature_dim, self.feature_dim),
                nn.GELU()
            )
            self.coarse_layers.append(coarse_layer)
        
        # Track current epoch for recover_rate decay
        self.current_epoch = 0
        
        # Store projection matrices (computed per batch)
        self.proj_matrices = None
    
    def adjust_recover_rate(self, epoch, max_epoch, epoch_rate=0.8):
        """Adjust recover rate based on training progress."""
        if epoch < max_epoch * epoch_rate:
            return self.recover_rate
        else:
            progress = (epoch - max_epoch * epoch_rate) / (max_epoch * (1 - epoch_rate))
            return self.recover_rate * (self.gamma ** progress)
    
    def encoding_mask_noise(self, num_nodes, device):
        """Apply masking to nodes.
        
        Parameters
        ----------
        num_nodes : int
            Number of nodes to mask
        device : torch.device
            Device to use
            
        Returns
        -------
        tuple
            (mask_nodes, token_nodes) - indices of masked and token nodes
        """
        perm = torch.randperm(num_nodes, device=device)
        num_mask_nodes = int(self.mask_rate * num_nodes)
        
        mask_nodes = perm[:num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]
        
        # For simplicity, all masked nodes are token nodes (no noise replacement)
        token_nodes = mask_nodes
        
        return mask_nodes, token_nodes
    
    def get_mask_list(self, mask_nodes, proj_matrices, device):
        """
        Propagate mask indices through hierarchy levels.
        
        Parameters
        ----------
        mask_nodes : torch.Tensor
            Masked node indices at finest level
        proj_matrices : list of torch.Tensor
            Projection matrices for each level
        device : torch.device
            Device to use
            
        Returns
        -------
        list of torch.Tensor
            Mask indicators (0/1) for each level
        """
        num_nodes_base = proj_matrices[0].size(1) if len(proj_matrices) > 0 else mask_nodes.size(0)
        
        # Create mask indicator for base level
        mask_indicator = torch.ones(num_nodes_base, device=device)
        mask_indicator[mask_nodes] = 0
        
        mask_list = [mask_indicator]
        
        # Propagate through hierarchy
        # A coarse node's mask is the AVERAGE of its children's masks
        # This creates softer masking at coarse levels
        for proj in proj_matrices:
            coarse_mask = torch.matmul(proj, mask_indicator)
            # coarse_mask now contains averaged values (due to normalized proj)
            # Keep as continuous values for better gradient flow
            mask_list.append(coarse_mask)
        
        return mask_list
    
    def recover_mask(self, mask_list, current_recover_rate):
        """Recover some masked nodes based on recover_rate."""
        if current_recover_rate <= 0:
            return mask_list
        
        new_mask_list = []
        for mask_indicator in mask_list:
            masked_indices = torch.where(mask_indicator == 0)[0]
            if len(masked_indices) > 0:
                num_recover = int(current_recover_rate * len(masked_indices))
                if num_recover > 0:
                    recover_indices = masked_indices[torch.randperm(len(masked_indices))[:num_recover]]
                    mask_indicator = mask_indicator.clone()
                    mask_indicator[recover_indices] = 1
            new_mask_list.append(mask_indicator)
        
        return new_mask_list
    
    def forward(self, batch):
        r"""Forward pass for Hi-GMAE encoding with hierarchical masking.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing encoded representations at multiple levels.
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0
        
        num_nodes = x_0.size(0)
        device = x_0.device
        
        # Store ORIGINAL features for reconstruction
        if hasattr(batch, 'x_raw'):
            x_raw_original = batch.x_raw.clone()
        else:
            x_raw_original = x_0.clone()
        
        # Create hierarchical structure (simple version for single graph in batch)
        # TODO: Handle batched graphs properly
        proj_matrices = []
        current_nodes = num_nodes
        
        # DEBUG: Log coarsening
        if torch.rand(1).item() < 0.01:  # Log 1% of batches
            print(f"\n[DEBUG] Starting coarsening: num_nodes={num_nodes}, num_levels={self.num_levels}")
        
        for level in range(self.num_levels - 1):
            proj = create_hierarchical_structure(
                edge_index, current_nodes, self.coarsening_ratio, device
            )
            proj_matrices.append(proj)
            prev_nodes = current_nodes
            current_nodes = proj.size(0)
            
            # DEBUG: Log coarsening ratio
            if torch.rand(1).item() < 0.01:
                print(f"[DEBUG] Level {level}: {prev_nodes} -> {current_nodes} nodes (ratio: {current_nodes/prev_nodes:.3f})")
        
        # Apply masking at finest level
        mask_nodes, token_nodes = self.encoding_mask_noise(num_nodes, device)
        
        # Get current recover rate
        current_recover_rate = self.adjust_recover_rate(
            self.current_epoch, 100, epoch_rate=0.8
        )
        
        # Propagate masks through hierarchy
        mask_list = self.get_mask_list(mask_nodes, proj_matrices, device)
        mask_list = self.recover_mask(mask_list, current_recover_rate)
        
        # Apply masking to features
        masked_x = x_0.clone()
        masked_x[token_nodes] = 0.0
        masked_x[token_nodes] = masked_x[token_nodes] + self.enc_mask_token
        
        # Multi-level encoding
        level_features = []
        current_features = masked_x
        
        # Encode through hierarchy
        for level_idx in range(self.num_levels):
            if level_idx == 0:
                # Base level encoding
                try:
                    current_features = self.backbone(
                        current_features,
                        edge_index,
                        batch=batch_indices,
                        edge_weight=edge_weight,
                    )
                except TypeError:
                    # Fallback if backbone doesn't accept edge_weight
                    current_features = self.backbone(
                        current_features,
                        edge_index,
                        batch=batch_indices,
                    )
                
                # DEBUG: Check for NaN/Inf
                if torch.rand(1).item() < 0.01:
                    has_nan = torch.isnan(current_features).any()
                    has_inf = torch.isinf(current_features).any()
                    mean_val = current_features.mean().item()
                    std_val = current_features.std().item()
                    print(f"[DEBUG] Level 0 features: shape={current_features.shape}, mean={mean_val:.4f}, std={std_val:.4f}, nan={has_nan}, inf={has_inf}")
            else:
                # Coarsen and process at coarse level
                proj = proj_matrices[level_idx - 1]
                current_features = torch.matmul(proj, current_features)
                # Apply learnable transformation at coarse level
                current_features = self.coarse_layers[level_idx - 1](current_features)
                
                # DEBUG
                if torch.rand(1).item() < 0.01:
                    mean_val = current_features.mean().item()
                    std_val = current_features.std().item()
                    print(f"[DEBUG] Level {level_idx} features: shape={current_features.shape}, mean={mean_val:.4f}, std={std_val:.4f}")
            
            level_features.append(current_features)
        
        # Prepare outputs for readout/decoder
        model_out = {
            "x_0": level_features[-1],  # Coarsest level features
            "level_features": level_features,  # All level features
            "proj_matrices": proj_matrices,  # Projection matrices
            "mask_list": mask_list,  # Masks at each level
            "mask_nodes": mask_nodes,  # Masked nodes at finest level
            "x_raw_original": x_raw_original,  # Original features for reconstruction
            "labels": batch.y if hasattr(batch, 'y') else None,
            "batch_0": batch_indices,
            "edge_index": edge_index,
        }
        
        return model_out
    
    def on_epoch_end(self, epoch):
        """Called at the end of each epoch."""
        self.current_epoch = epoch

