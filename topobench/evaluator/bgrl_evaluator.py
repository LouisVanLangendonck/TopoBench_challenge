"""BGRL evaluator for tracking bootstrap pre-training metrics."""

from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator.base import AbstractEvaluator


class BGRLEvaluator(AbstractEvaluator):
    r"""Evaluator for BGRL pre-training."""

    def __init__(self, metrics=None, **kwargs):
        super().__init__()
        if metrics is None:
            metrics = ["loss", "loss_12", "loss_21", "cosine_sim"]

        self.metric_names = metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["loss", "loss_12", "loss_21", "cosine_sim"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss', 'loss_12', 'loss_21', 'cosine_sim'"
                )
        self.metrics = MetricCollection(metrics_dict)
        self.best_metric = {}

    def update(self, model_out: dict):
        for metric_name in self.metric_names:
            if metric_name in model_out:
                self.metrics[metric_name].update(model_out[metric_name].detach().cpu())

    def compute(self):
        return self.metrics.compute()

    def reset(self):
        self.metrics.reset()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(metrics={self.metric_names})"
