"""Link Prediction Readout with Decoder for self-supervised pre-training."""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class LinkPredReadOut(AbstractZeroCellReadOut):
    r"""Link Prediction readout layer with decoder for edge prediction.

    This readout implements different decoders for link prediction,
    computing scores for positive and negative edges.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (unused for link pred but kept for compatibility).
    decoder_type : str, optional
        Type of decoder: "dot", "bilinear", "mlp" (default: "dot").
    task_level : str, optional
        Task level (default: "edge").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        decoder_type: str = "dot",
        task_level: str = "edge",
        **kwargs
    ):
        # Set task_level to node since we work with node embeddings
        # The actual task is edge-level but we process node embeddings
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level="node",  # Base class expects node or graph
            logits_linear_layer=False,  # We handle our own output
            **kwargs
        )
        
        self.decoder_type = decoder_type
        # Override task_level for edge-level task
        self.task_level = task_level
        
        # Build decoder
        self.decoder = self._build_decoder(decoder_type, hidden_dim)
    
    def _build_decoder(
        self, 
        decoder_type: str, 
        hidden_dim: int
    ) -> nn.Module:
        """Build the decoder module.
        
        Parameters
        ----------
        decoder_type : str
            Type of decoder ("dot", "bilinear", or "mlp").
        hidden_dim : int
            Hidden dimension.
            
        Returns
        -------
        nn.Module
            The decoder module.
        """
        if decoder_type == "dot":
            # Dot product decoder - no parameters
            return None
        elif decoder_type == "bilinear":
            # Bilinear decoder
            return nn.Bilinear(hidden_dim, hidden_dim, 1)
        elif decoder_type == "mlp":
            # MLP decoder
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'dot', 'bilinear', 'mlp'"
            )
    
    def decode(self, z, edge_index):
        """Decode edges given node embeddings.
        
        Parameters
        ----------
        z : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        edge_index : torch.Tensor
            Edge indices of shape (2, num_edges).
            
        Returns
        -------
        torch.Tensor
            Edge scores of shape (num_edges,).
        """
        src, dst = edge_index[0], edge_index[1]
        z_src, z_dst = z[src], z[dst]
        
        if self.decoder_type == "dot":
            # Simple dot product
            return (z_src * z_dst).sum(dim=-1)
        elif self.decoder_type == "bilinear":
            return self.decoder(z_src, z_dst).squeeze(-1)
        elif self.decoder_type == "mlp":
            edge_features = torch.cat([z_src, z_dst], dim=-1)
            return self.decoder(edge_features).squeeze(-1)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for link prediction decoding.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded node features
            - pos_edge_index: Positive edges to predict
            - neg_edge_index: Negative edges
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - pos_score: Prediction scores for positive edges
            - neg_score: Prediction scores for negative edges
            - logits: Concatenated scores (for compatibility)
            Plus all original model_out keys
        """
        # Get node embeddings and edge indices
        node_emb = model_out["x_0"]
        pos_edge_index = model_out["pos_edge_index"]
        neg_edge_index = model_out["neg_edge_index"]
        
        # Decode edge scores
        pos_score = self.decode(node_emb, pos_edge_index)
        neg_score = self.decode(node_emb, neg_edge_index)
        
        # Update model output
        model_out["pos_score"] = pos_score
        model_out["neg_score"] = neg_score
        
        # Logits for compatibility (concatenate positive and negative scores)
        model_out["logits"] = torch.cat([pos_score, neg_score], dim=0)
        
        # Edge labels: 1 for positive, 0 for negative
        model_out["edge_labels"] = torch.cat([
            torch.ones(pos_score.size(0), device=pos_score.device),
            torch.zeros(neg_score.size(0), device=neg_score.device)
        ], dim=0)
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type}, "
            f"task_level={self.task_level})"
        )

