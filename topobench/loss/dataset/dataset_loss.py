"""Generic dataset loss wrapper."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class DatasetLoss(AbstractLoss):
    r"""Generic dataset loss for standard supervised tasks.

    This loss function handles standard supervised learning tasks by computing
    the appropriate loss based on the task type (classification or regression).

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing dataset loss configuration with keys:
        - task_type: Type of task ("classification" or "regression")
        - num_classes: Number of classes for classification
        - loss_fn: Optional custom loss function name
    """

    def __init__(self, dataset_loss: dict):
        super().__init__()
        self.task_type = dataset_loss.get("task_type", "classification")
        self.num_classes = dataset_loss.get("num_classes", 2)
        self.loss_fn_name = dataset_loss.get("loss_fn", None)

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the dataset loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with 'logits' key.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data with 'y' labels.

        Returns
        -------
        torch.Tensor
            The computed loss value.
        """
        logits = model_out["logits"]
        labels = model_out.get("labels", batch.y)

        if self.task_type == "classification":
            if self.num_classes == 2 or len(logits.shape) == 1:
                # Binary classification
                if len(logits.shape) > 1 and logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            else:
                # Multi-class classification
                loss = F.cross_entropy(logits, labels.long())
        else:
            # Regression
            loss = F.mse_loss(logits, labels.float())

        return loss

