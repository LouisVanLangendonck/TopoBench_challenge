"""Link Prediction Loss."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class LinkPredLoss(AbstractLoss):
    r"""Loss function for Link Prediction pre-training.

    Computes binary cross-entropy loss for edge classification (positive vs negative edges).

    Parameters
    ----------
    loss_type : str, optional
        Type of loss function: "bce" (binary cross-entropy) (default: "bce").
    pos_weight : float, optional
        Weight for positive samples to handle class imbalance (default: 1.0).
    """

    def __init__(self, loss_type: str = "bce", pos_weight: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.pos_weight = pos_weight

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the link prediction loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - logits: Edge prediction scores/logits
            - labels: True edge labels (1 for positive, 0 for negative)
        batch : torch_geometric.data.Data
            Batch object (not used directly in this loss).

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        logits = model_out["logits"]
        labels = model_out["labels"]

        if self.loss_type == "bce":
            # Binary cross-entropy with logits
            if self.pos_weight != 1.0:
                pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight_tensor
                )
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

