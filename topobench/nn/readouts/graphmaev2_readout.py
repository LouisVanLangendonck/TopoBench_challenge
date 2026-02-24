"""GraphMAEv2 Readout for reconstruction-based pre-training with re-masking.
"""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class GraphMAEv2ReadOut(AbstractZeroCellReadOut):
    r"""GraphMAEv2 readout layer for feature reconstruction with re-masking.

    This readout implements the decoder for GraphMAEv2 pre-training,
    using multiple random re-masks during decoding to improve robustness.

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
    num_remasking : int, optional
        Number of re-masking iterations (default: 3).
    remask_rate : float, optional
        Rate of nodes to re-mask during decoding (default: 0.5).
    remask_method : str, optional
        Re-masking method: "random" or "fixed" (default: "random").
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
        num_remasking: int = 3,
        remask_rate: float = 0.5,
        remask_method: str = "random",
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
        self.num_remasking = num_remasking
        self.remask_rate = remask_rate
        self.remask_method = remask_method
        
        # Decoder mask token (for re-masking)
        self.dec_mask_token = nn.Parameter(torch.zeros(1, hidden_dim))
        nn.init.xavier_normal_(self.dec_mask_token)
        
        # Encoder to decoder projection
        self.encoder_to_decoder = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.xavier_normal_(self.encoder_to_decoder.weight, gain=1.414)
        
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
        
        Note: GNN decoders (gat, gcn) support re-masking via message passing.
              MLP/linear decoders should NOT use re-masking (set num_remasking=1).
        """
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
            # Single-layer GAT as in GraphMAEv2 paper
            from torch_geometric.nn import GATConv
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        elif decoder_type == "gcn":
            # Alternative: Single-layer GCN
            from torch_geometric.nn import GCNConv
            return GCNConv(in_dim, out_dim)
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'linear', 'mlp', 'gat', 'gcn'"
            )
    
    def random_remask(self, rep, num_nodes, device):
        """Apply random re-masking to decoder input.
        
        Parameters
        ----------
        rep : torch.Tensor
            Representation to re-mask.
        num_nodes : int
            Number of nodes.
        device : torch.device
            Device to use.
            
        Returns
        -------
        tuple
            (remasked_rep, remask_nodes, rekeep_nodes)
        """
        perm = torch.randperm(num_nodes, device=device)
        num_remask = int(self.remask_rate * num_nodes)
        
        remask_nodes = perm[:num_remask]
        rekeep_nodes = perm[num_remask:]
        
        rep_remasked = rep.clone()
        rep_remasked[remask_nodes] = 0
        rep_remasked[remask_nodes] = rep_remasked[remask_nodes] + self.dec_mask_token
        
        return rep_remasked, remask_nodes, rekeep_nodes
    
    def fixed_remask(self, rep, mask_nodes):
        """Apply fixed re-masking at original mask positions.
        
        Parameters
        ----------
        rep : torch.Tensor
            Representation to re-mask.
        mask_nodes : torch.Tensor
            Indices of originally masked nodes.
            
        Returns
        -------
        torch.Tensor
            Re-masked representation.
        """
        rep_remasked = rep.clone()
        rep_remasked[mask_nodes] = 0
        rep_remasked[mask_nodes] = rep_remasked[mask_nodes] + self.dec_mask_token
        return rep_remasked
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for GraphMAEv2 reconstruction with re-masking.

        The wrapper provides all necessary information via model_out, including
        edge_index for GNN decoders that need message passing during re-masking.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded node features from GNN
            - x_raw_original: Original RAW node features
            - mask_nodes: Indices of masked nodes
            - edge_index: Edge indices (required for GNN decoders)
            - edge_weight: Edge weights (optional, for weighted graphs)
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing reconstruction results and losses.
        """
        enc_rep = model_out["x_0"]
        x_raw_original = model_out.get("x_raw_original", model_out.get("x_original"))
        mask_nodes = model_out["mask_nodes"]
        edge_index = model_out.get("edge_index")
        edge_weight = model_out.get("edge_weight")
        
        num_nodes = enc_rep.size(0)
        device = enc_rep.device
        
        # Project from encoder to decoder space
        origin_rep = self.encoder_to_decoder(enc_rep)
        
        # Check if decoder is GNN-based (supports message passing)
        is_gnn_decoder = self.decoder_type in ["gat", "gcn"]
        
        # Apply re-masking strategy
        all_reconstructed = []
        
        if self.remask_method == "random":
            # For GNN decoders: use multi-view re-masking (as in paper)
            # For MLP decoders: skip re-masking (doesn't make sense without message passing)
            num_remasking_iterations = self.num_remasking if is_gnn_decoder else 1
            
            for i in range(num_remasking_iterations):
                if is_gnn_decoder and num_remasking_iterations > 1:
                    # Re-mask for GNN decoder (different mask each iteration)
                    rep_remasked, remask_nodes, rekeep_nodes = self.random_remask(origin_rep, num_nodes, device)
                else:
                    # No re-masking for MLP decoder
                    rep_remasked = origin_rep
                
                # Decode to reconstruct RAW features
                if is_gnn_decoder:
                    # GNN decoder needs edge_index
                    if edge_index is None:
                        raise ValueError("GNN decoder requires edge_index in model_out")
                    if edge_weight is not None:
                        recon_full = self.decoder(rep_remasked, edge_index, edge_weight=edge_weight)
                    else:
                        recon_full = self.decoder(rep_remasked, edge_index)
                else:
                    # MLP/Linear decoder
                    recon_full = self.decoder(rep_remasked)
                
                all_reconstructed.append(recon_full[mask_nodes])
            
            # Average reconstructions across different re-masks
            x_reconstructed = torch.stack(all_reconstructed).mean(dim=0)
            
        elif self.remask_method == "fixed":
            # Fixed re-masking at original positions
            if is_gnn_decoder:
                rep_remasked = self.fixed_remask(origin_rep, mask_nodes)
            else:
                rep_remasked = origin_rep  # No re-masking for MLP
            
            if is_gnn_decoder:
                if edge_index is None:
                    raise ValueError("GNN decoder requires edge_index in model_out")
                if edge_weight is not None:
                    recon_full = self.decoder(rep_remasked, edge_index, edge_weight=edge_weight)
                else:
                    recon_full = self.decoder(rep_remasked, edge_index)
            else:
                recon_full = self.decoder(rep_remasked)
            
            x_reconstructed = recon_full[mask_nodes]
            
        else:
            raise ValueError(f"Unknown remask_method: {self.remask_method}")
        
        # Update model output with RAW feature reconstruction
        model_out["x_reconstructed"] = x_reconstructed  # Reconstructed RAW features
        model_out["x_original"] = x_raw_original[mask_nodes]  # Original RAW features at masked positions
        model_out["num_remasking"] = self.num_remasking if is_gnn_decoder else 1
        
        return model_out
    
    def __repr__(self) -> str:
        is_gnn = self.decoder_type in ["gat", "gcn"]
        remask_info = f"num_remasking={self.num_remasking}" if is_gnn else "num_remasking=1 (disabled for non-GNN)"
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type}, "
            f"{remask_info}, "
            f"remask_rate={self.remask_rate}, "
            f"remask_method={self.remask_method})"
        )

