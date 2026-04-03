"""DGMAE Readout for reconstruction-based pre-training.

Based on: https://github.com/zhengziyu77/DGMAE
Paper: "Discrepancy-Aware Graph Mask Auto-Encoder" (KDD 2025)
"""

import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class DGMAEReadOut(AbstractZeroCellReadOut):
    r"""DGMAE readout layer for feature reconstruction.

    This readout implements the decoder for DGMAE pre-training.
    Unlike GraphMAEv2, it does NOT use re-masking (following original DGMAE).
    It outputs full reconstruction for heterophily loss computation.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (should match input feature dimension).
    decoder_type : str, optional
        Type of decoder: "linear", "mlp", "gat", "gcn" (default: "mlp").
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
        
        # Encoder to decoder projection (uses PyTorch default init as in original DGMAE)
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
        elif decoder_type == "gat":
            from torch_geometric.nn import GATConv
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        elif decoder_type == "gcn":
            from torch_geometric.nn import GCNConv
            return GCNConv(in_dim, out_dim)
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'linear', 'mlp', 'gat', 'gcn'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for DGMAE reconstruction.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded node features from GNN
            - x_raw_original: Original RAW node features
            - mask_nodes: Indices of masked nodes
            - edge_index: Edge indices (required for GNN decoders)
            - edge_weight: Edge weights (optional)
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing reconstruction results.
        """
        enc_rep = model_out["x_0"]
        x_raw_original = model_out.get("x_raw_original", model_out.get("x_original"))
        mask_nodes = model_out["mask_nodes"]
        edge_index = model_out.get("edge_index")
        edge_weight = model_out.get("edge_weight")
        
        # Project from encoder to decoder space
        rep = self.encoder_to_decoder(enc_rep)
        
        # Re-mask at mask positions before decoding (as in original GraphMAE)
        # This forces the decoder to reconstruct without seeing masked representations
        is_gnn_decoder = self.decoder_type in ["gat", "gcn"]
        if is_gnn_decoder:
            rep = rep.clone()
            rep[mask_nodes] = 0
        
        # Decode to reconstruct RAW features
        if is_gnn_decoder:
            if edge_index is None:
                raise ValueError("GNN decoder requires edge_index in model_out")
            if edge_weight is not None:
                recon_full = self.decoder(rep, edge_index, edge_weight=edge_weight)
            else:
                recon_full = self.decoder(rep, edge_index)
        else:
            recon_full = self.decoder(rep)
        
        # Update model output
        # For reconstruction loss: only masked nodes
        model_out["x_reconstructed"] = recon_full[mask_nodes]
        model_out["x_original"] = x_raw_original[mask_nodes]
        
        # For heterophily loss: full reconstruction
        model_out["x_reconstructed_full"] = recon_full
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type})"
        )
