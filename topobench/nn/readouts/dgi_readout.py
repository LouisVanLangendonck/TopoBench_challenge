"""DGI Readout with Discriminator for Deep Graph Infomax pre-training."""

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn.inits import uniform

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class Discriminator(nn.Module):
    """Bilinear discriminator for DGI.
    
    Computes the score between node embeddings and graph summary
    using a bilinear transformation.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension of embeddings.
    """
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize weights using uniform distribution."""
        size = self.weight.size(0)
        uniform(size, self.weight)
    
    def forward(self, x, summary):
        """Compute discrimination scores.
        
        Parameters
        ----------
        x : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        summary : torch.Tensor
            Graph summary expanded to match nodes, shape (num_nodes, hidden_dim).
            
        Returns
        -------
        torch.Tensor
            Discrimination scores of shape (num_nodes,).
        """
        h = torch.matmul(summary, self.weight)
        return torch.sum(x * h, dim=1)


class DGIReadOut(AbstractZeroCellReadOut):
    r"""DGI readout layer with discriminator for contrastive learning.

    This readout implements the discriminator for Deep Graph Infomax,
    computing scores for positive and negative node-graph pairs.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output dimension (unused for DGI but kept for compatibility).
    discriminator_type : str, optional
        Type of discriminator: "bilinear" or "mlp" (default: "bilinear").
    task_level : str, optional
        Task level (default: "node").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        discriminator_type: str = "bilinear",
        task_level: str = "node",
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,  # We handle our own output
            **kwargs
        )
        
        self.discriminator_type = discriminator_type
        
        # Build discriminator
        self.discriminator = self._build_discriminator(discriminator_type, hidden_dim)
    
    def _build_discriminator(
        self, 
        discriminator_type: str, 
        hidden_dim: int
    ) -> nn.Module:
        """Build the discriminator module.
        
        Parameters
        ----------
        discriminator_type : str
            Type of discriminator ("bilinear" or "mlp").
        hidden_dim : int
            Hidden dimension.
            
        Returns
        -------
        nn.Module
            The discriminator module.
        """
        if discriminator_type == "bilinear":
            return Discriminator(hidden_dim)
        elif discriminator_type == "mlp":
            return nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            raise ValueError(
                f"Unknown discriminator type: {discriminator_type}. "
                "Available options: 'bilinear', 'mlp'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for DGI discrimination.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Encoded positive node features
            - positive_expanded_summary: Graph summary for each positive node
            - negative_expanded_summary: Graph summary from different graphs
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - positive_score: Discrimination scores for positive pairs
            - negative_score: Discrimination scores for negative pairs
            - logits: Concatenated scores (for compatibility)
            Plus all original model_out keys
        """
        # Get node embeddings and summaries
        node_emb = model_out["x_0"]
        pos_summary = model_out["positive_expanded_summary"]
        neg_summary = model_out["negative_expanded_summary"]
        
        # Compute discrimination scores
        if self.discriminator_type == "bilinear":
            positive_score = self.discriminator(node_emb, pos_summary)
            negative_score = self.discriminator(node_emb, neg_summary)
        else:
            # MLP discriminator expects concatenated input
            pos_input = torch.cat([node_emb, pos_summary], dim=-1)
            neg_input = torch.cat([node_emb, neg_summary], dim=-1)
            positive_score = self.discriminator(pos_input).squeeze(-1)
            negative_score = self.discriminator(neg_input).squeeze(-1)
        
        # Update model output
        model_out["positive_score"] = positive_score
        model_out["negative_score"] = negative_score
        
        # Logits for compatibility (concatenate positive and negative scores)
        model_out["logits"] = torch.cat([positive_score, negative_score], dim=0)
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"discriminator_type={self.discriminator_type}, "
            f"task_level={self.task_level})"
        )

