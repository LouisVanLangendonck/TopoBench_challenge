"""GraphMAE Readout for reconstruction-based pre-training."""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class GraphMAEReadOut(AbstractZeroCellReadOut):
    r"""GraphMAE readout layer for feature reconstruction.

    This readout implements the decoder for GraphMAE pre-training,
    reconstructing the original node features from the masked encodings.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (should match input feature dimension).
    decoder_type : str, optional
        Type of decoder: "linear", "mlp", or "gnn" (default: "linear").
    decoder_hidden_dim : int, optional
        Hidden dimension for MLP decoder (default: 256).
    remask : bool, optional
        Whether to re-mask the masked nodes before decoding (default: True).
    task_level : str, optional
        Task level (default: "node").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        decoder_type: str = "linear",
        decoder_hidden_dim: int = 256,
        remask: bool = True,
        task_level: str = "node",
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,  # We handle projection ourselves
            **kwargs
        )
        
        self.decoder_type = decoder_type
        self.remask = remask
        
        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Build decoder
        self.decoder = self._build_decoder(
            decoder_type, hidden_dim, out_channels, decoder_hidden_dim
        )
    
    def _build_decoder(
        self, 
        decoder_type: str, 
        in_dim: int, 
        out_dim: int, 
        hidden_dim: int
    ) -> nn.Module:
        """Build the decoder module.
        
        Parameters
        ----------
        decoder_type : str
            Type of decoder ("linear", "mlp", or "gnn").
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension (feature dimension).
        hidden_dim : int
            Hidden dimension for MLP decoder.
            
        Returns
        -------
        nn.Module
            The decoder module.
        """
        if decoder_type == "linear":
            return nn.Linear(in_dim, out_dim)
        elif decoder_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, out_dim)
            )
        elif decoder_type == "gnn":
            # For GNN decoder, we would need the graph structure
            # For now, fallback to MLP
            # TODO: Implement proper GNN decoder if needed
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'linear', 'mlp', 'gnn'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for GraphMAE reconstruction.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded node features from GNN
            - x_raw_original: Original RAW node features (before encoding)
            - mask_nodes: Indices of masked nodes
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_reconstructed: Reconstructed RAW features (at masked positions)
            - x_original: Original RAW features (at masked positions)
            - logits: Reconstructed features (for compatibility)
            Plus all original model_out keys
        """
        # Get encoded representations from GNN
        enc_rep = model_out["x_0"]
        
        # Get ORIGINAL RAW features (what we want to reconstruct)
        x_raw_original = model_out.get("x_raw_original", model_out.get("x_original"))
        mask_nodes = model_out["mask_nodes"]
        
        # Project from encoder to decoder space
        rep = self.encoder_to_decoder(enc_rep)
        
        # Re-mask: zero out the masked nodes in the representation
        # This forces the decoder to rely on the neighborhood information
        if self.remask and self.decoder_type != "linear":
            rep = rep.clone()  # Don't modify in-place
            rep[mask_nodes] = 0
        
        # Decode to reconstruct RAW features
        if self.decoder_type in ("mlp", "linear"):
            x_reconstructed_full = self.decoder(rep)
        else:
            # GNN decoder would need edge information
            # For now, use MLP-style decoding
            x_reconstructed_full = self.decoder(rep)
        
        # Update model output with reconstruction results
        # We only care about reconstructing the MASKED positions
        model_out["x_reconstructed"] = x_reconstructed_full[mask_nodes]  # Reconstructed RAW features at masked positions
        model_out["x_original"] = x_raw_original[mask_nodes]  # Original RAW features at masked positions
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type}, "
            f"remask={self.remask}, "
            f"task_level={self.task_level})"
        )