"""DGMAE Readout for dual-branch reconstruction.

Handles both original feature reconstruction and feature discrepancy reconstruction.
"""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class DGMAEReadOut(AbstractZeroCellReadOut):
    r"""DGMAE readout layer for dual-branch reconstruction.

    This readout implements the decoder for DGMAE pre-training with:
    1. Original feature reconstruction (from masked embeddings)
    2. Feature discrepancy reconstruction (from embedding discrepancies)

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (should match input feature dimension).
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
        
        self.decoder_type = decoder_type
        
        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)
        
        # Build decoder for original feature reconstruction
        self.decoder = self._build_decoder(
            decoder_type, hidden_dim, out_channels, decoder_hidden_dim
        )
        
        # Build decoder for discrepancy reconstruction
        # Takes embedding discrepancy as input and predicts raw feature discrepancy
        self.discrepancy_decoder = self._build_decoder(
            decoder_type, hidden_dim, out_channels, decoder_hidden_dim
        )
    
    def _build_decoder(
        self, 
        decoder_type: str, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int
    ) -> nn.Module:
        """Build the decoder module."""
        if decoder_type == "linear":
            return nn.Linear(in_dim, out_dim)
        elif decoder_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim * 2),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim * 2, out_dim)
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'linear', 'mlp'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for DGMAE dual-branch reconstruction.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded node features from GNN (masked graph)
            - x_raw_original: Original RAW node features
            - mask_nodes: Indices of masked nodes
            - keep_nodes: Indices of kept nodes
            - z_discrepancy: Embedding discrepancy
            - x_raw_discrepancy: Raw feature discrepancy (target)
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing reconstruction results for both branches.
        """
        enc_rep = model_out["x_0"]
        x_raw_original = model_out.get("x_raw_original", model_out.get("x_original"))
        mask_nodes = model_out["mask_nodes"]
        keep_nodes = model_out["keep_nodes"]
        
        z_discrepancy = model_out["z_discrepancy"]
        x_raw_discrepancy = model_out["x_raw_discrepancy"]
        
        num_nodes = enc_rep.size(0)
        device = enc_rep.device
        
        # ===== Branch 1: Original Feature Reconstruction =====
        
        # Project from encoder to decoder space
        origin_rep = self.encoder_to_decoder(enc_rep)
        
        # Decode to reconstruct RAW features
        # Note: In DGMAE, masked nodes should be set to 0 before decoding
        # This forces the decoder to rely on context (neighbors)
        origin_rep_for_decode = origin_rep.clone()
        origin_rep_for_decode[mask_nodes] = 0.0
        
        recon_full = self.decoder(origin_rep_for_decode)
        
        # Extract reconstructed features at masked positions
        x_reconstructed = recon_full[mask_nodes]
        x_original = x_raw_original[mask_nodes]
        
        # ===== Branch 2: Feature Discrepancy Reconstruction =====
        
        # Decode embedding discrepancy to predict raw feature discrepancy
        # Only for UNMASKED nodes (keep_nodes) as per DGMAE paper
        x_discrepancy_pred = self.discrepancy_decoder(z_discrepancy[keep_nodes])
        x_discrepancy_target = x_raw_discrepancy[keep_nodes]
        
        # Update model output with reconstruction results
        model_out["x_reconstructed"] = x_reconstructed  # Reconstructed RAW features (masked nodes)
        model_out["x_original"] = x_original  # Original RAW features (masked nodes)
        
        model_out["x_discrepancy_pred"] = x_discrepancy_pred  # Predicted discrepancy (unmasked nodes)
        model_out["x_discrepancy_target"] = x_discrepancy_target  # Target discrepancy (unmasked nodes)
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type})"
        )


