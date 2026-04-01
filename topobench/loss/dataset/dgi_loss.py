"""DGI Loss for Deep Graph Infomax pre-training."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class DGILoss(AbstractLoss):
    r"""Deep Graph Infomax (DGI) loss.

    DGI uses a binary cross-entropy loss to train the discriminator to
    distinguish between positive samples (real node-summary pairs) and
    negative samples (corrupted node-summary pairs).

    The loss maximizes mutual information between local node representations
    and global graph summaries.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss: "bce" for binary cross-entropy (default: "bce").
    reduction : str, optional
        Reduction method: "mean" or "sum" (default: "mean").
    """

    def __init__(
        self,
        loss_type: str = "bce",
        reduction: str = "mean",
    ):
        super().__init__()

        self.loss_type = loss_type
        self.reduction = reduction

        if loss_type not in ["bce"]:
            raise ValueError(f"loss_type must be 'bce', got {loss_type}")

        if reduction not in ["mean", "sum"]:
            raise ValueError(
                f"reduction must be 'mean' or 'sum', got {reduction}"
            )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        """Compute DGI loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - logits: Discrimination logits (positive and negative concatenated)
            - labels: Labels (1 for positive, 0 for negative)
            - num_positive: Number of positive samples
            - num_negative: Number of negative samples
        batch : torch_geometric.data.Data
            Batch object (not used in loss computation).

        Returns
        -------
        torch.Tensor
            The computed total loss.
        """
        logits = model_out["logits"]
        labels = model_out["labels"]
        num_positive = model_out["num_positive"]
        model_out["num_negative"]

        # Binary cross-entropy loss with logits
        if self.loss_type == "bce":
            loss = F.binary_cross_entropy_with_logits(
                logits, labels, reduction=self.reduction
            )

            # Compute separate losses for positive and negative samples
            logits_positive = logits[:num_positive]
            logits_negative = logits[num_positive:]
            labels_positive = labels[:num_positive]
            labels_negative = labels[num_positive:]

            loss_positive = F.binary_cross_entropy_with_logits(
                logits_positive, labels_positive, reduction=self.reduction
            )

            loss_negative = F.binary_cross_entropy_with_logits(
                logits_negative, labels_negative, reduction=self.reduction
            )
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Compute accuracy
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (predictions == labels).float().mean()

        # Store individual losses in model_out for tracking
        model_out["loss_positive"] = loss_positive
        model_out["loss_negative"] = loss_negative
        model_out["accuracy"] = accuracy

        return loss
