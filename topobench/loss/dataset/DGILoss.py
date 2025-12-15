"""Loss functions for Deep Graph Infomax (DGI) pre-training."""

import torch
import torch.nn as nn
import torch_geometric

from topobench.loss.base import AbstractLoss


class DGILoss(AbstractLoss):
    r"""Loss function for Deep Graph Infomax (DGI) pre-training.

    This loss computes the binary cross-entropy between positive and negative
    discrimination scores, encouraging the model to distinguish between
    real node-graph pairs and corrupted pairs.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss function. Options: "bce", "margin" (default: "bce").
    margin : float, optional
        Margin for margin-based loss (default: 1.0).
    """

    def __init__(self, loss_type: str = "bce", margin: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
        
        if loss_type == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss_type == "margin":
            self.criterion = None  # We compute margin loss manually
        else:
            raise ValueError(
                f"Invalid loss type '{loss_type}'. "
                "Supported types: 'bce', 'margin'"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the DGI contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - positive_score: Discrimination scores for positive pairs
            - negative_score: Discrimination scores for negative pairs
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (unused but kept for compatibility).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        positive_score = model_out["positive_score"]
        negative_score = model_out["negative_score"]
        
        if self.loss_type == "bce":
            # Binary cross-entropy: positive pairs should score 1, negative should score 0
            pos_loss = self.criterion(
                positive_score, 
                torch.ones_like(positive_score)
            )
            neg_loss = self.criterion(
                negative_score, 
                torch.zeros_like(negative_score)
            )
            loss = pos_loss + neg_loss
        else:
            # Margin loss: positive scores should be higher than negative by margin
            loss = torch.clamp(self.margin - positive_score + negative_score, min=0).mean()
        
        return loss

