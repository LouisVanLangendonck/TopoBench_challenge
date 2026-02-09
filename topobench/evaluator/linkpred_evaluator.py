"""Evaluator for Link Prediction pre-training."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection, Accuracy, AUROC

from topobench.evaluator import AbstractEvaluator


class LinkPredEvaluator(AbstractEvaluator):
    r"""Evaluator for Link Prediction pre-training tasks.

    This evaluator computes metrics for link prediction self-supervised pre-training,
    tracking edge classification performance.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "loss": Mean binary cross-entropy loss
        - "accuracy": Edge classification accuracy
        - "auroc": Area under ROC curve
        - "precision": Precision for positive edges
        - "recall": Recall for positive edges
        Default: ["loss", "accuracy", "auroc"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["loss", "accuracy", "auroc"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name == "loss":
                metrics_dict[metric_name] = MeanMetric()
            elif metric_name == "accuracy":
                metrics_dict[metric_name] = Accuracy(task="binary")
            elif metric_name == "auroc":
                metrics_dict[metric_name] = AUROC(task="binary")
            elif metric_name in ["precision", "recall"]:
                # Will compute manually
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss', 'accuracy', 'auroc', 'precision', 'recall'"
                )
        
        self.metrics = MetricCollection(metrics_dict)
        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metric_names})"

    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output. It should contain the following keys:
            - logits : torch.Tensor
                Edge prediction logits/scores.
            - labels : torch.Tensor
                True edge labels (1 for positive, 0 for negative).
            - loss : torch.Tensor (optional)
                The loss value.
        """
        logits = model_out["logits"].detach().cpu()
        labels = model_out["labels"].detach().cpu()
        
        # Compute probabilities
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Update loss
        if "loss" in self.metric_names:
            if "loss" in model_out:
                loss_value = model_out["loss"].detach().cpu()
            else:
                # Compute BCE as fallback
                loss_value = F.binary_cross_entropy_with_logits(logits, labels)
            self.metrics["loss"].update(loss_value)
        
        # Update accuracy
        if "accuracy" in self.metric_names:
            self.metrics["accuracy"].update(preds, labels.long())
        
        # Update AUROC
        if "auroc" in self.metric_names:
            self.metrics["auroc"].update(probs, labels.long())
        
        # Update precision
        if "precision" in self.metric_names:
            true_pos = ((preds == 1) & (labels == 1)).sum().float()
            pred_pos = (preds == 1).sum().float()
            precision = true_pos / (pred_pos + 1e-8)
            self.metrics["precision"].update(precision)
        
        # Update recall
        if "recall" in self.metric_names:
            true_pos = ((preds == 1) & (labels == 1)).sum().float()
            actual_pos = (labels == 1).sum().float()
            recall = true_pos / (actual_pos + 1e-8)
            self.metrics["recall"].update(recall)

    def compute(self):
        r"""Compute the metrics.

        Returns
        -------
        dict
            Dictionary containing the computed metrics.
        """
        return self.metrics.compute()

    def reset(self):
        """Reset the metrics.

        This method should be called after each epoch.
        """
        self.metrics.reset()

