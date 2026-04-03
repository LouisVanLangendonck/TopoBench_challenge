"""Evaluator for VGAE edge pretraining (BCE-based edge classification metrics)."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection, Accuracy, AUROC

from topobench.evaluator import AbstractEvaluator


class VGAEEvaluator(AbstractEvaluator):
    r"""Metrics for VGAE-style edge prediction on sampled pos/neg edges.

    **Important:** The training objective is ELBO = edge reconstruction BCE + :math:`\beta`·KL.
    Accuracy and AUROC depend only on **edge logits** (reconstruction term). If ``loss`` is
    taken from ``model_out["loss"]``, it is the **full ELBO**, so it can fall while accuracy
    drifts to chance (posterior collapse: KL pushes :math:`q(z|x)` toward the prior and
    :math:`z_i\!\cdot\!z_j \to 0`). Here, metric ``loss`` is **edge BCE only**, aligned with
    accuracy/AUROC; use ``elbo`` for the scalar actually backpropagated.
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["loss", "elbo", "accuracy", "auroc"]

        self.metric_names = metrics

        metrics_dict = {}
        for metric_name in metrics:
            if metric_name == "loss":
                metrics_dict[metric_name] = MeanMetric()
            elif metric_name == "elbo":
                metrics_dict[metric_name] = MeanMetric()
            elif metric_name == "accuracy":
                metrics_dict[metric_name] = Accuracy(task="binary")
            elif metric_name == "auroc":
                metrics_dict[metric_name] = AUROC(task="binary")
            elif metric_name in ["precision", "recall"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss', 'elbo', 'accuracy', 'auroc', "
                    "'precision', 'recall'"
                )

        self.metrics = MetricCollection(metrics_dict)
        self.best_metric = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metric_names})"

    def update(self, model_out: dict):
        logits = model_out["logits"].detach().cpu()
        labels = model_out["labels"].detach().cpu()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        if "loss" in self.metric_names:
            # Edge BCE only — same signal as accuracy / AUROC (not ELBO).
            recon_bce = F.binary_cross_entropy_with_logits(logits, labels)
            self.metrics["loss"].update(recon_bce)

        if "elbo" in self.metric_names:
            if "loss" in model_out:
                self.metrics["elbo"].update(model_out["loss"].detach().cpu())

        if "accuracy" in self.metric_names:
            self.metrics["accuracy"].update(preds, labels.long())

        if "auroc" in self.metric_names:
            self.metrics["auroc"].update(probs, labels.long())

        if "precision" in self.metric_names:
            true_pos = ((preds == 1) & (labels == 1)).sum().float()
            pred_pos = (preds == 1).sum().float()
            precision = true_pos / (pred_pos + 1e-8)
            self.metrics["precision"].update(precision)

        if "recall" in self.metric_names:
            true_pos = ((preds == 1) & (labels == 1)).sum().float()
            actual_pos = (labels == 1).sum().float()
            recall = true_pos / (actual_pos + 1e-8)
            self.metrics["recall"].update(recall)

    def compute(self):
        return self.metrics.compute()

    def reset(self):
        self.metrics.reset()
