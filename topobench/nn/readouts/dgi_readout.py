"""DGI Readout with discriminator for Deep Graph Infomax."""

import torch
import torch.nn as nn
import torch_geometric
from torch_scatter import scatter

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class DGIReadOut(AbstractZeroCellReadOut):
    r"""DGI readout layer with discriminator.

    This readout implements the Deep Graph Infomax discriminator that:
    1. Computes graph-level summary vectors from node embeddings (readout)
    2. Discriminates between positive (real) and negative (corrupted) node-summary pairs
    
    The discriminator uses a bilinear scoring function to compute compatibility
    between node embeddings and graph summaries.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder (node embedding dimension).
    out_channels : int
        Output dimension (typically 1 for binary discrimination, but we output
        logits for all nodes so this is set to 1).
    readout_type : str, optional
        Type of readout function: "mean", "sum", "max" (default: "mean").
    task_level : str, optional
        Task level (default: "node").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int = 1,
        readout_type: str = "mean",
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
        
        self.readout_type = readout_type
        
        # Bilinear discriminator: score = h^T W s
        # where h is node embedding, s is graph summary, W is learnable weight
        self.discriminator = nn.Bilinear(hidden_dim, hidden_dim, 1)
        
        # Initialize discriminator weights
        nn.init.xavier_uniform_(self.discriminator.weight.data)
        if self.discriminator.bias is not None:
            self.discriminator.bias.data.fill_(0.0)
        
        # Sigmoid for converting scores to probabilities (used in loss)
        self.sigmoid = nn.Sigmoid()
    
    def graph_readout(self, node_embeddings, batch_indices):
        """Compute graph-level summary vectors from node embeddings.
        
        Parameters
        ----------
        node_embeddings : torch.Tensor
            Node embeddings of shape (num_nodes, hidden_dim).
        batch_indices : torch.Tensor
            Batch assignment for each node.
            
        Returns
        -------
        torch.Tensor
            Graph-level summaries of shape (num_graphs, hidden_dim).
        """
        if self.readout_type == "mean":
            # Average pooling over nodes in each graph
            graph_summaries = scatter(
                node_embeddings,
                batch_indices,
                dim=0,
                reduce="mean"
            )
        elif self.readout_type == "sum":
            # Sum pooling over nodes in each graph
            graph_summaries = scatter(
                node_embeddings,
                batch_indices,
                dim=0,
                reduce="sum"
            )
        elif self.readout_type == "max":
            # Max pooling over nodes in each graph
            graph_summaries = scatter(
                node_embeddings,
                batch_indices,
                dim=0,
                reduce="max"
            )
        else:
            raise ValueError(f"Unknown readout_type: {self.readout_type}")
        
        return graph_summaries
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for DGI discrimination.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_0: Positive node embeddings (from original graph)
            - x_0_corrupted: Negative node embeddings (from corrupted/different graph)
            - batch_0: Batch indices for positive samples
            - batch_0_corrupted: Batch indices for negative samples (for graph_diffusion)
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - logits: Discrimination logits (positive and negative concatenated)
            - labels: Labels (1 for positive, 0 for negative)
            - summary: Graph-level summary vectors
            - num_positive: Number of positive samples
            - num_negative: Number of negative samples
        """
        h_positive = model_out["x_0"]  # (num_positive_nodes, hidden_dim)
        h_negative = model_out["x_0_corrupted"]  # (num_negative_nodes, hidden_dim)
        batch_positive = model_out["batch_0"]
        batch_negative = model_out.get("batch_0_corrupted", batch_positive)  # For compatibility
        
        num_positive_nodes = h_positive.size(0)
        num_negative_nodes = h_negative.size(0)
        
        # Compute graph-level summary from positive embeddings
        # Shape: (num_graphs, hidden_dim)
        graph_summary = self.graph_readout(h_positive, batch_positive)
        
        # Apply sigmoid to summary (as in original DGI paper)
        graph_summary = self.sigmoid(graph_summary)
        
        # Expand summary to match each node with its graph's summary
        # For positive samples: use batch_positive
        # For negative samples: use batch_negative (may be different for graph_diffusion)
        node_summaries_positive = graph_summary[batch_positive]
        node_summaries_negative = graph_summary[batch_negative]
        
        # Compute discrimination scores using bilinear function
        # Positive samples: real node embeddings with real summaries
        scores_positive = self.discriminator(h_positive, node_summaries_positive).squeeze(-1)
        
        # Negative samples: corrupted/different node embeddings with real summaries
        scores_negative = self.discriminator(h_negative, node_summaries_negative).squeeze(-1)
        
        # Concatenate positive and negative scores
        logits = torch.cat([scores_positive, scores_negative], dim=0)
        
        # Create labels (1 for positive, 0 for negative)
        labels_positive = torch.ones(num_positive_nodes, dtype=torch.float, device=h_positive.device)
        labels_negative = torch.zeros(num_negative_nodes, dtype=torch.float, device=h_positive.device)
        labels = torch.cat([labels_positive, labels_negative], dim=0)
        
        # Update model output
        model_out["logits"] = logits
        model_out["labels"] = labels
        model_out["summary"] = graph_summary
        model_out["num_positive"] = num_positive_nodes
        model_out["num_negative"] = num_negative_nodes
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"readout_type={self.readout_type}, "
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels})"
        )


