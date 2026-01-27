"""Evaluator for S2GAE pre-training."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class S2GAEEvaluator(AbstractEvaluator):
    r"""Evaluator for S2GAE pre-training tasks.

    This evaluator computes metrics for S2GAE self-supervised pre-training,
    tracking edge reconstruction quality.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "edge_accuracy": Edge prediction accuracy
        - "edge_reconstruction_loss": Edge reconstruction loss
        Default: ["edge_accuracy", "edge_reconstruction_loss"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["edge_accuracy", "edge_reconstruction_loss"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["edge_accuracy", "edge_reconstruction_loss"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'edge_accuracy', 'edge_reconstruction_loss'"
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
            - edge_pred : torch.Tensor
                The predicted edge probabilities.
            - edge_label : torch.Tensor
                The ground truth edge labels (1 for positive, 0 for negative).
            - edge_reconstruction_loss : float (optional)
                The edge reconstruction loss value.
            - edge_accuracy : float (optional)
                The edge prediction accuracy value.
        """
        edge_pred = model_out.get("edge_pred", None)
        edge_label = model_out.get("edge_label", None)
        
        if edge_pred is None or edge_label is None:
            # No predictions available (e.g., during evaluation without masked edges)
            return
        
        edge_pred = edge_pred.detach().cpu()
        edge_label = edge_label.detach().cpu()
        
        # Compute and update each metric
        if "edge_accuracy" in self.metric_names:
            if "edge_accuracy" in model_out:
                accuracy_value = torch.tensor(model_out["edge_accuracy"])
            else:
                # Compute accuracy
                pred_binary = (edge_pred > 0.5).float()
                accuracy_value = (pred_binary == edge_label).float().mean()
            self.metrics["edge_accuracy"].update(accuracy_value)
        
        if "edge_reconstruction_loss" in self.metric_names:
            if "edge_reconstruction_loss" in model_out:
                loss_value = torch.tensor(model_out["edge_reconstruction_loss"])
            else:
                # Compute BCE as fallback
                loss_value = F.binary_cross_entropy(edge_pred, edge_label, reduction='mean')
            self.metrics["edge_reconstruction_loss"].update(loss_value)

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


