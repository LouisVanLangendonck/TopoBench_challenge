"""Loss functions for Link Prediction pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class LinkPredLoss(AbstractLoss):
    r"""Loss function for Link Prediction pre-training.

    This loss computes the binary cross-entropy between predicted edge scores
    and ground truth labels (1 for positive edges, 0 for negative edges).

    Parameters
    ----------
    loss_type : str, optional
        Type of loss function. Options: "bce", "margin", "infonce" (default: "bce").
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
        elif loss_type == "infonce":
            self.criterion = None  # We compute InfoNCE loss manually
        else:
            raise ValueError(
                f"Invalid loss type '{loss_type}'. "
                "Supported types: 'bce', 'margin', 'infonce'"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the link prediction loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - pos_score: Prediction scores for positive edges
            - neg_score: Prediction scores for negative edges
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (unused but kept for compatibility).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        pos_score = model_out["pos_score"]
        neg_score = model_out["neg_score"]
        
        if self.loss_type == "bce":
            # Binary cross-entropy: positive edges should score 1, negative should score 0
            pos_loss = self.criterion(
                pos_score, 
                torch.ones_like(pos_score)
            )
            neg_loss = self.criterion(
                neg_score, 
                torch.zeros_like(neg_score)
            )
            loss = pos_loss + neg_loss
            
        elif self.loss_type == "margin":
            # Margin ranking loss: positive scores should be higher than negative by margin
            # For each positive edge, compare against all negative edges
            loss = torch.clamp(self.margin - pos_score.unsqueeze(1) + neg_score.unsqueeze(0), min=0)
            loss = loss.mean()
            
        elif self.loss_type == "infonce":
            # InfoNCE-style contrastive loss
            # Treat each positive edge as positive and negatives from same batch as negatives
            temperature = 0.1
            
            # Compute logits: positive scores vs all scores
            pos_logits = pos_score / temperature
            neg_logits = neg_score / temperature
            
            # For each positive, softmax over positive + all negatives
            # This is a simplified version
            all_logits = torch.cat([pos_logits.unsqueeze(1), neg_logits.unsqueeze(0).expand(pos_logits.size(0), -1)], dim=1)
            labels = torch.zeros(pos_logits.size(0), dtype=torch.long, device=pos_score.device)
            
            loss = F.cross_entropy(all_logits, labels)
        
        return loss

