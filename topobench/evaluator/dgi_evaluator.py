"""DGI Evaluator for Deep Graph Infomax pre-training."""

import torch
from torchmetrics import Accuracy, MeanMetric, MetricCollection

from topobench.evaluator.base import AbstractEvaluator


class DGIEvaluator(AbstractEvaluator):
    r"""Evaluator for Deep Graph Infomax (DGI) pre-training.
    
    Tracks the following metrics:
    - loss: Total DGI loss (binary cross-entropy)
    - loss_positive: Loss on positive samples (real node-summary pairs)
    - loss_negative: Loss on negative samples (corrupted node-summary pairs)
    - accuracy: Discrimination accuracy (how well discriminator separates positive/negative)
    
    Parameters
    ----------
    metrics : list of str, optional
        List of metrics to track. Default: ["loss", "loss_positive", "loss_negative", "accuracy"]
    """
    
    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["loss", "loss_positive", "loss_negative", "accuracy"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["loss", "loss_positive", "loss_negative"]:
                metrics_dict[metric_name] = MeanMetric()
            elif metric_name == "accuracy":
                metrics_dict[metric_name] = Accuracy(task="binary")
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss', 'loss_positive', 'loss_negative', 'accuracy'"
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
                Discrimination logits (positive and negative concatenated).
            - labels : torch.Tensor
                Labels (1 for positive, 0 for negative).
            - loss : torch.Tensor (optional)
                The total loss value.
            - loss_positive : torch.Tensor (optional)
                Loss on positive samples.
            - loss_negative : torch.Tensor (optional)
                Loss on negative samples.
        """
        logits = model_out["logits"].detach().cpu()
        labels = model_out["labels"].detach().cpu()
        
        # Compute predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Update loss metrics
        if "loss" in self.metric_names and "loss" in model_out:
            loss_value = model_out["loss"].detach().cpu()
            self.metrics["loss"].update(loss_value)
        
        if "loss_positive" in self.metric_names and "loss_positive" in model_out:
            loss_pos_value = model_out["loss_positive"].detach().cpu()
            self.metrics["loss_positive"].update(loss_pos_value)
        
        if "loss_negative" in self.metric_names and "loss_negative" in model_out:
            loss_neg_value = model_out["loss_negative"].detach().cpu()
            self.metrics["loss_negative"].update(loss_neg_value)
        
        # Update accuracy
        if "accuracy" in self.metric_names:
            self.metrics["accuracy"].update(preds, labels.long())
    
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


