"""Hi-GMAE Readout for hierarchical reconstruction-based pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class HiGMAEReadOut(AbstractZeroCellReadOut):
    r"""Hi-GMAE readout layer for hierarchical feature reconstruction.

    This readout implements the decoder for Hi-GMAE pre-training,
    using hierarchical decoding from coarse to fine levels with skip connections.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (should match input feature dimension).
    num_levels : int, optional
        Number of hierarchical levels (default: 2).
    decoder_type : str, optional
        Type of decoder: "linear", "mlp" (default: "mlp").
    decoder_hidden_dim : int, optional
        Hidden dimension for MLP decoder (default: 256).
    task_level : str, optional
        Task level (default: "node").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        num_levels: int = 2,
        decoder_type: str = "mlp",
        decoder_hidden_dim: int = 256,
        task_level: str = "node",
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,
            **kwargs
        )
        
        self.num_levels = num_levels
        self.decoder_type = decoder_type
        
        # Build decoders for each level
        # Decoder at coarsest level (level num_levels-1) outputs hidden_dim
        # Decoder at finest level (level 0) outputs out_channels (raw features)
        self.decoders = nn.ModuleList()
        for level in range(num_levels):
            # Output hidden_dim for all levels except the finest (level 0)
            out_dim = out_channels if level == 0 else hidden_dim
            decoder = self._build_decoder(
                decoder_type, hidden_dim, out_dim, decoder_hidden_dim
            )
            self.decoders.append(decoder)
        
        # Simple learnable weights for skip connections
        self.skip_weights = nn.ParameterList()
        for level in range(num_levels - 1):
            # Single learnable weight per level (simpler than full gates)
            weight = nn.Parameter(torch.tensor(0.5))
            self.skip_weights.append(weight)
    
    def _build_decoder(
        self, 
        decoder_type: str, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int
    ) -> nn.Module:
        """Build a decoder module with improved architecture."""
        if decoder_type == "linear":
            return nn.Linear(in_dim, out_dim)
        elif decoder_type == "mlp":
            # Simple MLP with LayerNorm for stability
            return nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'linear', 'mlp'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for Hi-GMAE hierarchical reconstruction.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - level_features: List of features at each hierarchy level
            - proj_matrices: Projection matrices for coarsening
            - mask_list: Masks at each level
            - mask_nodes: Indices of masked nodes at finest level
            - x_raw_original: Original RAW node features
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing reconstruction results.
        """
        level_features = model_out["level_features"]
        proj_matrices = model_out["proj_matrices"]
        mask_list = model_out["mask_list"]
        mask_nodes = model_out["mask_nodes"]
        x_raw_original = model_out["x_raw_original"]
        
        device = level_features[0].device
        
        # Start decoding from coarsest level
        current_features = level_features[-1]
        
        # Decode through hierarchy (from coarse to fine)
        for level_idx in range(self.num_levels - 1, -1, -1):
            # Apply decoder
            decoded_features = self.decoders[level_idx](current_features)
            
            # Project to finer level (if not at finest level)
            if level_idx > 0:
                proj = proj_matrices[level_idx - 1]
                current_features = torch.matmul(proj.T, decoded_features)
                
                # Weighted skip connection from encoder at this level
                mask_indicator = mask_list[level_idx - 1].view(-1, 1)
                skip_features = level_features[level_idx - 1] * mask_indicator
                
                # Simple weighted combination (learnable weight per level)
                alpha = torch.sigmoid(self.skip_weights[level_idx - 1])
                current_features = alpha * skip_features + (1 - alpha) * current_features
            else:
                current_features = decoded_features
        
        # Final reconstruction at finest level
        x_reconstructed = current_features[mask_nodes]
        x_original = x_raw_original[mask_nodes]
        
        # CRITICAL: Clip reconstructed features to prevent explosion
        x_reconstructed = torch.clamp(x_reconstructed, min=-10.0, max=10.0)
        
        # DEBUG: Check reconstruction quality
        if torch.rand(1).item() < 0.01:
            recon_mean = x_reconstructed.mean().item()
            recon_std = x_reconstructed.std().item()
            orig_mean = x_original.mean().item()
            orig_std = x_original.std().item()
            cos_sim = F.cosine_similarity(x_reconstructed, x_original, dim=-1).mean().item()
            mse = F.mse_loss(x_reconstructed, x_original).item()
            print(f"[DEBUG] Reconstruction: recon_mean={recon_mean:.4f}, orig_mean={orig_mean:.4f}, cos_sim={cos_sim:.4f}, mse={mse:.4f}")
            print(f"[DEBUG] Num masked nodes: {len(mask_nodes)}/{level_features[0].size(0)}")
        
        # Update model output
        model_out["x_reconstructed"] = x_reconstructed  # Reconstructed features
        model_out["x_original"] = x_original  # Original features at masked positions
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_levels={self.num_levels}, "
            f"decoder_type={self.decoder_type})"
        )

