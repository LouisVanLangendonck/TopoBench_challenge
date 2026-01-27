"""S2GAE Loss for edge reconstruction."""

import torch
import torch.nn.functional as F

from topobench.loss.base import AbstractLoss


class S2GAELoss(AbstractLoss):
    r"""Loss function for S2GAE pre-training.

    Computes binary cross-entropy loss for edge reconstruction,
    balancing positive (masked edges) and negative (non-existing edges).

    Parameters
    ----------
    loss_type : str, optional
        Type of loss: "bce" (binary cross-entropy) or "weighted_bce" (default: "bce").
    pos_weight : float, optional
        Weight for positive samples in weighted BCE (default: 1.0).
    """

    def __init__(
        self,
        loss_type: str = "bce",
        pos_weight: float = 1.0,
        **kwargs
    ):
        super().__init__()
        self.loss_type = loss_type
        self.pos_weight = pos_weight

    def forward(self, model_out: dict, batch) -> torch.Tensor:
        r"""Compute S2GAE loss.

        Parameters
        ----------
        model_out : dict
            Model output dictionary containing:
            - edge_pred: Predicted edge probabilities
            - edge_label: Ground truth edge labels (1 for positive, 0 for negative)
        batch : torch_geometric.data.Data
            Batch data (not used directly).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        edge_pred = model_out.get("edge_pred", None)
        edge_label = model_out.get("edge_label", None)
        
        if edge_pred is None or edge_label is None:
            # No loss during evaluation
            return torch.tensor(0.0, device=batch.x_0.device)
        
        # Compute binary cross-entropy loss
        if self.loss_type == "bce":
            # Standard BCE with logits (but we already applied sigmoid, so use BCE)
            loss = F.binary_cross_entropy(edge_pred, edge_label, reduction='mean')
        
        elif self.loss_type == "weighted_bce":
            # Weighted BCE to handle class imbalance
            weight = torch.where(
                edge_label == 1,
                torch.tensor(self.pos_weight, device=edge_label.device),
                torch.tensor(1.0, device=edge_label.device)
            )
            loss = F.binary_cross_entropy(edge_pred, edge_label, weight=weight, reduction='mean')
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        # Compute accuracy as a metric and store for logging
        with torch.no_grad():
            pred_binary = (edge_pred > 0.5).float()
            accuracy = (pred_binary == edge_label).float().mean()
        
        # Store individual metrics in model_out for logging/evaluation
        model_out["edge_reconstruction_loss"] = loss.item()
        model_out["edge_accuracy"] = accuracy.item()
        
        return loss



