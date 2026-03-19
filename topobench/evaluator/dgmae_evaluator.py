"""Evaluator for DGMAE pre-training.

Based on: https://github.com/zhengziyu77/DGMAE
Paper: "Discrepancy-Aware Graph Mask Auto-Encoder" (KDD 2025)
"""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class DGMAEEvaluator(AbstractEvaluator):
    r"""Evaluator for DGMAE pre-training tasks.

    This evaluator computes metrics for DGMAE self-supervised pre-training,
    tracking reconstruction quality and heterophily/discrepancy loss.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "recon_loss": Mean reconstruction loss
        - "hetero_loss": Mean heterophily/discrepancy loss
        - "total_loss": Combined loss
        - "cosine_sim": Mean cosine similarity between original and reconstructed features
        Default: ["recon_loss", "hetero_loss", "cosine_sim"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["recon_loss", "hetero_loss", "cosine_sim"]
        
        self.metric_names = metrics
        
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["recon_loss", "hetero_loss", "total_loss", "cosine_sim", "mse"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'recon_loss', 'hetero_loss', 'total_loss', 'cosine_sim', 'mse'"
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
            - x_reconstructed : torch.Tensor
                The reconstructed features at masked positions.
            - x_original : torch.Tensor
                The original features at masked positions.
            - loss_recon : torch.Tensor (optional)
                The reconstruction loss value.
            - loss_hetero : torch.Tensor (optional)
                The heterophily/discrepancy loss value.
            - loss : torch.Tensor (optional)
                The total loss value.
        """
        x_reconstructed = model_out["x_reconstructed"].detach().cpu()
        x_original = model_out["x_original"].detach().cpu()
        
        if "recon_loss" in self.metric_names:
            if "loss_recon" in model_out:
                loss_value = model_out["loss_recon"].detach().cpu()
            else:
                loss_value = F.mse_loss(x_reconstructed, x_original)
            self.metrics["recon_loss"].update(loss_value)
        
        if "hetero_loss" in self.metric_names:
            if "loss_hetero" in model_out:
                hetero_value = model_out["loss_hetero"].detach().cpu()
            else:
                hetero_value = torch.tensor(0.0)
            self.metrics["hetero_loss"].update(hetero_value)
        
        if "total_loss" in self.metric_names:
            if "loss" in model_out:
                total_value = model_out["loss"].detach().cpu()
            else:
                total_value = F.mse_loss(x_reconstructed, x_original)
            self.metrics["total_loss"].update(total_value)
        
        if "cosine_sim" in self.metric_names:
            x_reconstructed_norm = F.normalize(x_reconstructed, p=2, dim=-1)
            x_original_norm = F.normalize(x_original, p=2, dim=-1)
            cosine_sim = (x_reconstructed_norm * x_original_norm).sum(dim=-1).mean()
            self.metrics["cosine_sim"].update(cosine_sim)
        
        if "mse" in self.metric_names:
            mse = F.mse_loss(x_reconstructed, x_original)
            self.metrics["mse"].update(mse)

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
