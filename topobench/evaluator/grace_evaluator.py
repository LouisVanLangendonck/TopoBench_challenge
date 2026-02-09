"""GRACE Evaluator for tracking contrastive learning metrics."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator.base import AbstractEvaluator


class GRACEEvaluator(AbstractEvaluator):
    r"""Evaluator for GRACE contrastive learning.

    Tracks contrastive loss metrics during pre-training.

    Parameters
    ----------
    metrics : list of str, optional
        List of metrics to track. Available options:
        - "loss": Total contrastive loss
        - "loss_view1": Loss for view 1 direction
        - "loss_view2": Loss for view 2 direction
        - "cosine_sim": Cosine similarity between views (optional)
        Default: ["loss", "loss_view1", "loss_view2"]
    **kwargs : dict
        Additional arguments for AbstractEvaluator.
    """

    def __init__(
        self,
        metrics: list = None,
        **kwargs
    ):
        super().__init__()
        
        if metrics is None:
            metrics = ["loss", "loss_view1", "loss_view2"]
        
        self.metric_names = metrics
        
        # Initialize metrics using torchmetrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["loss", "loss_view1", "loss_view2", "cosine_sim"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss', 'loss_view1', 'loss_view2', 'cosine_sim'"
                )
        
        self.metrics = MetricCollection(metrics_dict)
        self.best_metric = {}
    
    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output. Should contain:
            - loss: Total contrastive loss (if tracking)
            - loss_view1: View 1 loss (if tracking)
            - loss_view2: View 2 loss (if tracking)
            - h_1: Projected embeddings from view 1 (for cosine_sim)
            - h_2: Projected embeddings from view 2 (for cosine_sim)
        """
        # Update loss metrics
        if "loss" in self.metric_names and "loss" in model_out:
            loss_value = model_out["loss"].detach().cpu()
            self.metrics["loss"].update(loss_value)
        
        if "loss_view1" in self.metric_names and "loss_view1" in model_out:
            loss_view1 = model_out["loss_view1"].detach().cpu()
            self.metrics["loss_view1"].update(loss_view1)
        
        if "loss_view2" in self.metric_names and "loss_view2" in model_out:
            loss_view2 = model_out["loss_view2"].detach().cpu()
            self.metrics["loss_view2"].update(loss_view2)
        
        # Compute cosine similarity between views if requested
        if "cosine_sim" in self.metric_names:
            if "h_1" in model_out and "h_2" in model_out:
                h_1 = model_out["h_1"].detach().cpu()
                h_2 = model_out["h_2"].detach().cpu()
                
                h_1_norm = F.normalize(h_1, p=2, dim=-1)
                h_2_norm = F.normalize(h_2, p=2, dim=-1)
                cosine_sim = (h_1_norm * h_2_norm).sum(dim=-1).mean()
                self.metrics["cosine_sim"].update(cosine_sim)
    
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
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metric_names})"

