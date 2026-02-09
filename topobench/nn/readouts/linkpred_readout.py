"""Link Prediction Readout for edge classification."""

import torch
import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class LinkPredReadOut(AbstractZeroCellReadOut):
    r"""Link Prediction readout layer for edge classification.

    This readout takes node embeddings and edge indices (positive and negative),
    combines endpoint node embeddings, and predicts whether edges exist.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder (node embedding dimension).
    out_channels : int
        Output dimension (typically 1 for binary classification or 2 for logits).
    decoder_type : str, optional
        Type of edge decoder: "dot", "concat_mlp", "hadamard_mlp" (default: "concat_mlp").
        - "dot": Simple dot product of node embeddings
        - "concat_mlp": Concatenate node embeddings and pass through MLP
        - "hadamard_mlp": Element-wise product of node embeddings and pass through MLP
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
        out_channels: int = 1,
        decoder_type: str = "concat_mlp",
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
        self.decoder_hidden_dim = decoder_hidden_dim
        
        # Build edge decoder
        self.decoder = self._build_decoder(
            decoder_type, hidden_dim, out_channels, decoder_hidden_dim
        )
    
    def _build_decoder(
        self, 
        decoder_type: str, 
        hidden_dim: int, 
        out_channels: int, 
        decoder_hidden_dim: int
    ) -> nn.Module:
        """Build the edge decoder module."""
        if decoder_type == "dot":
            # Dot product decoder (output is scalar per edge)
            return DotProductDecoder()
        
        elif decoder_type == "concat_mlp":
            # Concatenate node embeddings and pass through MLP
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(decoder_hidden_dim, out_channels)
            )
        
        elif decoder_type == "hadamard_mlp":
            # Element-wise product and pass through MLP
            return nn.Sequential(
                nn.Linear(hidden_dim, decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(decoder_hidden_dim, out_channels)
            )
        
        else:
            raise ValueError(
                f"Unknown decoder type: {decoder_type}. "
                "Available options: 'dot', 'concat_mlp', 'hadamard_mlp'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for Link Prediction edge scoring.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Node embeddings from encoder
            - pos_edge_index: Positive edges to score
            - neg_edge_index: Negative edges to score
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - logits: Edge scores/logits (concatenated positive and negative)
            - labels: Edge labels (1 for positive, 0 for negative)
            - num_pos_edges: Number of positive edges
            - num_neg_edges: Number of negative edges
        """
        node_embeddings = model_out["x_0"]
        pos_edge_index = model_out["pos_edge_index"]
        neg_edge_index = model_out["neg_edge_index"]
        
        num_pos_edges = pos_edge_index.size(1)
        num_neg_edges = neg_edge_index.size(1)
        
        # Combine positive and negative edges
        all_edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        
        # Create labels (1 for positive, 0 for negative)
        pos_labels = torch.ones(num_pos_edges, dtype=torch.float, device=node_embeddings.device)
        neg_labels = torch.zeros(num_neg_edges, dtype=torch.float, device=node_embeddings.device)
        edge_labels = torch.cat([pos_labels, neg_labels], dim=0)
        
        # Get source and target node embeddings
        src_embeddings = node_embeddings[all_edge_index[0]]  # (num_edges, hidden_dim)
        dst_embeddings = node_embeddings[all_edge_index[1]]  # (num_edges, hidden_dim)
        
        # Decode edges based on decoder type
        if self.decoder_type == "dot":
            edge_scores = self.decoder(src_embeddings, dst_embeddings)
        elif self.decoder_type == "concat_mlp":
            edge_repr = torch.cat([src_embeddings, dst_embeddings], dim=-1)
            edge_scores = self.decoder(edge_repr).squeeze(-1)
        elif self.decoder_type == "hadamard_mlp":
            edge_repr = src_embeddings * dst_embeddings  # Element-wise product
            edge_scores = self.decoder(edge_repr).squeeze(-1)
        else:
            raise ValueError(f"Unknown decoder type: {self.decoder_type}")
        
        # Update model output
        model_out["logits"] = edge_scores
        model_out["labels"] = edge_labels
        model_out["num_pos_edges"] = num_pos_edges
        model_out["num_neg_edges"] = num_neg_edges
        model_out["all_edge_index"] = all_edge_index
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"decoder_type={self.decoder_type}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels})"
        )


class DotProductDecoder(nn.Module):
    """Simple dot product decoder for edge prediction."""
    
    def forward(self, src_embeddings, dst_embeddings):
        """Compute dot product between source and destination embeddings.
        
        Parameters
        ----------
        src_embeddings : torch.Tensor
            Source node embeddings of shape (num_edges, hidden_dim).
        dst_embeddings : torch.Tensor
            Destination node embeddings of shape (num_edges, hidden_dim).
            
        Returns
        -------
        torch.Tensor
            Edge scores of shape (num_edges,).
        """
        return (src_embeddings * dst_embeddings).sum(dim=-1)

