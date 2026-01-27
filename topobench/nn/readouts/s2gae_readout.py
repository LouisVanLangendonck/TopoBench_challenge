"""S2GAE Readout for cross-layer edge reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import negative_sampling

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class S2GAEReadOut(AbstractZeroCellReadOut):
    r"""S2GAE readout layer for cross-layer edge reconstruction.

    This readout implements the decoder for S2GAE, which performs
    cross-layer feature combination for link prediction. The key idea
    is that different GNN layers capture different structural patterns,
    and their pairwise combinations improve edge reconstruction.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder layers.
    out_channels : int
        Output dimension (typically 1 for binary edge prediction).
    num_layers : int
        Number of GNN layers (for cross-layer combinations).
    decoder_hidden_dim : int, optional
        Hidden dimension for the decoder MLP (default: 256).
    decoder_layers : int, optional
        Number of decoder MLP layers (default: 3).
    task_level : str, optional
        Task level (default: "edge" for link prediction).
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int = 1,
        num_layers: int = 2,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 3,
        task_level: str = "edge",
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,
            **kwargs
        )
        
        self.num_layers = num_layers
        self.decoder_hidden_dim = decoder_hidden_dim
        self.decoder_layers = decoder_layers
        
        # Cross-layer combinations: num_layers^2 possible combinations
        self.num_combinations = num_layers * num_layers
        
        # Build decoder MLP that takes element-wise products as input
        self.decoder = self._build_decoder(
            hidden_dim, decoder_hidden_dim, out_channels, decoder_layers
        )
    
    def _build_decoder(
        self, 
        in_dim: int, 
        hidden_dim: int, 
        out_dim: int, 
        num_layers: int
    ) -> nn.Module:
        """Build the decoder MLP."""
        layers = []
        
        layers.append(nn.Linear(in_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        return nn.ModuleList(layers)
    
    def cross_layer_combination(self, layer_reps, edge_index):
        """Compute cross-layer feature combinations for edge pairs.
        
        For each edge (src, dst), compute pairwise products between
        all layer combinations: layer_i[src] * layer_j[dst].
        
        Parameters
        ----------
        layer_reps : list of torch.Tensor
            List of representations from each GNN layer.
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
            
        Returns
        -------
        torch.Tensor
            Cross-layer combinations of shape (num_edges, num_combinations, hidden_dim).
        """
        src_idx = edge_index[0]
        dst_idx = edge_index[1]
        
        combinations = []
        
        # Compute all pairwise layer combinations
        for i in range(self.num_layers):
            src_rep_i = layer_reps[i][src_idx]  # (num_edges, hidden_dim)
            for j in range(self.num_layers):
                dst_rep_j = layer_reps[j][dst_idx]  # (num_edges, hidden_dim)
                # Element-wise product
                combination = src_rep_i * dst_rep_j  # (num_edges, hidden_dim)
                combinations.append(combination)
        
        # Stack combinations: (num_edges, num_combinations, hidden_dim)
        combinations = torch.stack(combinations, dim=1)
        
        return combinations
    
    def decode_edges(self, cross_layer_feats):
        """Decode edge probabilities from cross-layer features.
        
        Parameters
        ----------
        cross_layer_feats : torch.Tensor
            Cross-layer features of shape (num_edges, num_combinations, hidden_dim).
            
        Returns
        -------
        torch.Tensor
            Edge probabilities of shape (num_edges,).
        """
        # Sum across combinations
        x = cross_layer_feats.sum(dim=1)  # (num_edges, hidden_dim)
        
        # Pass through decoder MLP
        for i, layer in enumerate(self.decoder):
            x = layer(x)
            if i < len(self.decoder) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        
        # Apply sigmoid for binary prediction
        x = torch.sigmoid(x).squeeze(-1)  # (num_edges,)
        
        return x
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for S2GAE edge reconstruction.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - layer_reps: List of layer representations
            - masked_edges: Edges that were masked during training
            - full_edge_index: Original full edge index
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing reconstruction results and losses.
        """
        layer_reps = model_out["layer_reps"]
        masked_edges = model_out.get("masked_edges", None)
        full_edge_index = model_out["full_edge_index"]
        
        device = layer_reps[0].device
        num_nodes = layer_reps[0].size(0)
        
        if masked_edges is not None:
            # Positive edges: masked edges
            pos_edges = masked_edges
            
            # Negative edges: random non-existing edges
            num_neg = pos_edges.size(1)
            neg_edges = negative_sampling(
                edge_index=full_edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg,
            )
            
            # Compute cross-layer features for positive edges
            pos_cross_feats = self.cross_layer_combination(layer_reps, pos_edges)
            pos_pred = self.decode_edges(pos_cross_feats)
            
            # Compute cross-layer features for negative edges
            neg_cross_feats = self.cross_layer_combination(layer_reps, neg_edges)
            neg_pred = self.decode_edges(neg_cross_feats)
            
            # Combine predictions and labels
            edge_pred = torch.cat([pos_pred, neg_pred], dim=0)
            edge_label = torch.cat([
                torch.ones_like(pos_pred),
                torch.zeros_like(neg_pred)
            ], dim=0)
            
            # Update model output
            model_out["edge_pred"] = edge_pred
            model_out["edge_label"] = edge_label
            model_out["pos_edges"] = pos_edges
            model_out["neg_edges"] = neg_edges
        else:
            # No masked edges available (shouldn't happen if mask_during_eval=true)
            model_out["edge_pred"] = None
            model_out["edge_label"] = None
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_layers={self.num_layers}, "
            f"decoder_hidden_dim={self.decoder_hidden_dim}, "
            f"decoder_layers={self.decoder_layers})"
        )



