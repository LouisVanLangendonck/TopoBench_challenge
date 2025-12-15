"""Evaluator for Link Prediction pre-training."""

import torch
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class LinkPredEvaluator(AbstractEvaluator):
    r"""Evaluator for Link Prediction pre-training tasks.

    This evaluator computes metrics for self-supervised link prediction,
    tracking AUC-ROC, Average Precision, and Hits@K metrics.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "auc": Area Under ROC Curve
        - "ap": Average Precision
        - "hits_at_10": Hits@10 metric
        - "hits_at_50": Hits@50 metric
        - "hits_at_100": Hits@100 metric
        - "mrr": Mean Reciprocal Rank
        Default: ["auc", "ap", "hits_at_100"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["auc", "ap", "hits_at_100"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["auc", "ap", "hits_at_10", "hits_at_50", "hits_at_100", "mrr"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'auc', 'ap', 'hits_at_10', 'hits_at_50', 'hits_at_100', 'mrr'"
                )
        
        self.metrics = MetricCollection(metrics_dict)
        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metric_names})"

    def _compute_auc(self, pos_score, neg_score):
        """Compute AUC-ROC score.
        
        Parameters
        ----------
        pos_score : torch.Tensor
            Scores for positive edges.
        neg_score : torch.Tensor
            Scores for negative edges.
            
        Returns
        -------
        torch.Tensor
            AUC score.
        """
        # AUC = P(pos_score > neg_score)
        # Compare each positive with each negative
        diff = pos_score.unsqueeze(1) - neg_score.unsqueeze(0)
        auc = (diff > 0).float().mean() + 0.5 * (diff == 0).float().mean()
        return auc
    
    def _compute_ap(self, pos_score, neg_score):
        """Compute Average Precision.
        
        Parameters
        ----------
        pos_score : torch.Tensor
            Scores for positive edges.
        neg_score : torch.Tensor
            Scores for negative edges.
            
        Returns
        -------
        torch.Tensor
            Average Precision score.
        """
        # Concatenate scores and labels
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([
            torch.ones_like(pos_score),
            torch.zeros_like(neg_score)
        ])
        
        # Sort by score (descending)
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        
        # Compute precision at each threshold
        tp_cumsum = sorted_labels.cumsum(dim=0)
        precision = tp_cumsum / torch.arange(1, len(sorted_labels) + 1, device=scores.device, dtype=scores.dtype)
        
        # Average precision = mean of precision at positive labels
        ap = (precision * sorted_labels).sum() / sorted_labels.sum().clamp(min=1)
        
        return ap
    
    def _compute_hits_at_k(self, pos_score, neg_score, k):
        """Compute Hits@K metric.
        
        Parameters
        ----------
        pos_score : torch.Tensor
            Scores for positive edges.
        neg_score : torch.Tensor
            Scores for negative edges.
        k : int
            Number of top predictions to consider.
            
        Returns
        -------
        torch.Tensor
            Hits@K score.
        """
        # For each positive edge, count how many negatives have higher scores
        # If the positive is in top-K, it's a hit
        ranks = (neg_score.unsqueeze(0) > pos_score.unsqueeze(1)).sum(dim=1) + 1
        hits = (ranks <= k).float().mean()
        return hits
    
    def _compute_mrr(self, pos_score, neg_score):
        """Compute Mean Reciprocal Rank.
        
        Parameters
        ----------
        pos_score : torch.Tensor
            Scores for positive edges.
        neg_score : torch.Tensor
            Scores for negative edges.
            
        Returns
        -------
        torch.Tensor
            MRR score.
        """
        # For each positive edge, compute its rank among all negatives
        ranks = (neg_score.unsqueeze(0) > pos_score.unsqueeze(1)).sum(dim=1) + 1
        mrr = (1.0 / ranks.float()).mean()
        return mrr

    def update(self, model_out: dict):
        r"""Update the metrics with the model output.

        Parameters
        ----------
        model_out : dict
            The model output. It should contain the following keys:
            - pos_score : torch.Tensor
                The prediction scores for positive edges.
            - neg_score : torch.Tensor
                The prediction scores for negative edges.
        """
        pos_score = model_out["pos_score"].detach().cpu()
        neg_score = model_out["neg_score"].detach().cpu()
        
        # Compute and update each metric
        if "auc" in self.metric_names:
            auc = self._compute_auc(pos_score, neg_score)
            self.metrics["auc"].update(auc)
        
        if "ap" in self.metric_names:
            ap = self._compute_ap(pos_score, neg_score)
            self.metrics["ap"].update(ap)
        
        if "hits_at_10" in self.metric_names:
            hits = self._compute_hits_at_k(pos_score, neg_score, k=10)
            self.metrics["hits_at_10"].update(hits)
        
        if "hits_at_50" in self.metric_names:
            hits = self._compute_hits_at_k(pos_score, neg_score, k=50)
            self.metrics["hits_at_50"].update(hits)
        
        if "hits_at_100" in self.metric_names:
            hits = self._compute_hits_at_k(pos_score, neg_score, k=100)
            self.metrics["hits_at_100"].update(hits)
        
        if "mrr" in self.metric_names:
            mrr = self._compute_mrr(pos_score, neg_score)
            self.metrics["mrr"].update(mrr)

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

