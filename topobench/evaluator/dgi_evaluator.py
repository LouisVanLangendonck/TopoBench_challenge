"""Evaluator for Deep Graph Infomax (DGI) pre-training."""

import torch
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class DGIEvaluator(AbstractEvaluator):
    r"""Evaluator for Deep Graph Infomax (DGI) pre-training tasks.

    This evaluator computes metrics for self-supervised contrastive pre-training,
    tracking discrimination quality between positive and negative pairs.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "pos_score": Average positive discrimination score
        - "neg_score": Average negative discrimination score
        - "discrimination_acc": Discrimination accuracy (positive > 0, negative < 0)
        Default: ["pos_score", "neg_score", "discrimination_acc"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["pos_score", "neg_score", "discrimination_acc"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["pos_score", "neg_score", "discrimination_acc"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'pos_score', 'neg_score', 'discrimination_acc'"
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
            - positive_score : torch.Tensor
                The discrimination scores for positive (real) pairs.
            - negative_score : torch.Tensor
                The discrimination scores for negative (corrupted) pairs.
        """
        positive_score = model_out["positive_score"].detach().cpu()
        negative_score = model_out["negative_score"].detach().cpu()
        
        # Compute and update each metric
        if "pos_score" in self.metric_names:
            # Average positive score (should be high)
            pos_score_mean = positive_score.mean()
            self.metrics["pos_score"].update(pos_score_mean)
        
        if "neg_score" in self.metric_names:
            # Average negative score (should be low)
            neg_score_mean = negative_score.mean()
            self.metrics["neg_score"].update(neg_score_mean)
        
        if "discrimination_acc" in self.metric_names:
            # Discrimination accuracy: percentage of correct classifications
            # Positive should be > 0, negative should be < 0
            pos_correct = (positive_score > 0).float().sum()
            neg_correct = (negative_score < 0).float().sum()
            total = len(positive_score) + len(negative_score)
            acc = (pos_correct + neg_correct) / total
            self.metrics["discrimination_acc"].update(acc)

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

