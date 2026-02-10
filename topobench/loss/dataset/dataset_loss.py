"""Loss module for the topobench package."""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class DatasetLoss(AbstractLoss):
    r"""Defines the default model loss for the given task.

    Parameters
    ----------
    dataset_loss : dict
        Dictionary containing the dataset loss information.
    """

    def __init__(self, dataset_loss):
        super().__init__()
        self.task = dataset_loss["task"]
        self.loss_type = dataset_loss["loss_type"]
        # Dataset loss
        if self.task == "classification":
            assert self.loss_type == "cross_entropy", (
                "Invalid loss type for classification task,TB supports only cross_entropy loss for classification task"
            )
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.task == "multilabel classification":
            assert self.loss_type == "BCE", (
                "Invalid loss type for classification task,TB supports only BCE for multilabel classification task"
            )
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        elif self.task == "regression" and self.loss_type == "mse":
            self.criterion = torch.nn.MSELoss()
        elif self.task == "regression" and self.loss_type == "mae":
            self.criterion = torch.nn.L1Loss()
        else:
            raise Exception("Loss is not defined")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(task={self.task}, loss_type={self.loss_type})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output.
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing the model output with the loss.
        """
        logits = model_out["logits"]
        target = model_out["labels"]

        return self.forward_criterion(logits, target)

    def forward_criterion(self, logits, target):
        r"""Forward pass of the loss function.

        Parameters
        ----------
        logits : torch.Tensor
            Model predictions.
        target : torch.Tensor
            Ground truth labels.

        Returns
        -------
        torch.Tensor
            Loss value.
        """
        if self.task == "regression":
            target = target.unsqueeze(1)
            dataset_loss = self.criterion(logits, target)

        elif self.task == "classification":
            dataset_loss = self.criterion(logits, target)

        elif self.task == "multilabel classification":
            mask = ~torch.isnan(target)
            # Avoid NaN values in the target
            target = torch.where(mask, target, torch.zeros_like(target))
            loss = self.criterion(logits, target)
            # Mask out the loss for NaN values
            loss = loss * mask
            # Take out average
            dataset_loss = (loss.sum(dim=-1) / mask.sum(dim=-1)).mean()

        else:
            raise Exception("Loss is not defined")

        return dataset_loss
# """Generic dataset loss wrapper."""

# import torch
# import torch.nn.functional as F
# import torch_geometric

# from topobench.loss.base import AbstractLoss


# class DatasetLoss(AbstractLoss):
#     r"""Generic dataset loss for standard supervised tasks.

#     This loss function handles standard supervised learning tasks by computing
#     the appropriate loss based on the task type (classification or regression).

#     Parameters
#     ----------
#     dataset_loss : dict
#         Dictionary containing dataset loss configuration with keys:
#         - task_type: Type of task ("classification" or "regression")
#         - num_classes: Number of classes for classification
#         - loss_fn: Optional custom loss function name
#     """

#     def __init__(self, dataset_loss: dict):
#         super().__init__()
#         self.task_type = dataset_loss.get("task_type", "classification")
#         self.num_classes = dataset_loss.get("num_classes", 2)
#         self.loss_fn_name = dataset_loss.get("loss_fn", None)

#     def forward(self, model_out: dict, batch: torch_geometric.data.Data):
#         r"""Compute the dataset loss.

#         Parameters
#         ----------
#         model_out : dict
#             Dictionary containing the model output with 'logits' key.
#         batch : torch_geometric.data.Data
#             Batch object containing the batched domain data with 'y' labels.

#         Returns
#         -------
#         torch.Tensor
#             The computed loss value.
#         """
#         logits = model_out["logits"]
#         labels = model_out.get("labels", batch.y)

#         if self.task_type == "classification":
#             if self.num_classes == 2 or len(logits.shape) == 1:
#                 # Binary classification
#                 if len(logits.shape) > 1 and logits.shape[-1] == 1:
#                     logits = logits.squeeze(-1)
#                 loss = F.binary_cross_entropy_with_logits(logits, labels.float())
#             else:
#                 # Multi-class classification
#                 loss = F.cross_entropy(logits, labels.long())
#         else:
#             # Regression
#             loss = F.mse_loss(logits, labels.float())

#         return loss

