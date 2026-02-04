"""Evaluator for DGMAE pre-training."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class DGMAEEvaluator(AbstractEvaluator):
    r"""Evaluator for DGMAE pre-training tasks.

    This evaluator computes metrics for DGMAE self-supervised pre-training,
    tracking both original feature reconstruction and discrepancy reconstruction quality.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "loss_f": Mean feature reconstruction loss
        - "loss_d": Mean discrepancy reconstruction loss
        - "total_loss": Combined loss
        - "cosine_sim_f": Mean cosine similarity for feature reconstruction
        - "cosine_sim_d": Mean cosine similarity for discrepancy reconstruction
        - "mse_f": MSE for feature reconstruction
        - "mse_d": MSE for discrepancy reconstruction
        Default: ["loss_f", "loss_d", "cosine_sim_f", "cosine_sim_d"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["loss_f", "loss_d", "cosine_sim_f", "cosine_sim_d"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in [
                "loss_f", "loss_d", "total_loss", 
                "cosine_sim_f", "cosine_sim_d", 
                "mse_f", "mse_d"
            ]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'loss_f', 'loss_d', 'total_loss', "
                    "'cosine_sim_f', 'cosine_sim_d', 'mse_f', 'mse_d'"
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
            - x_discrepancy_pred : torch.Tensor
                The predicted feature discrepancy at unmasked positions.
            - x_discrepancy_target : torch.Tensor
                The target feature discrepancy at unmasked positions.
            - loss_f : torch.Tensor (optional)
                The feature reconstruction loss value.
            - loss_d : torch.Tensor (optional)
                The discrepancy reconstruction loss value.
            - loss : torch.Tensor (optional)
                The total loss value.
        """
        x_reconstructed = model_out["x_reconstructed"].detach().cpu()
        x_original = model_out["x_original"].detach().cpu()
        x_discrepancy_pred = model_out["x_discrepancy_pred"].detach().cpu()
        x_discrepancy_target = model_out["x_discrepancy_target"].detach().cpu()
        
        # Feature reconstruction metrics
        if "loss_f" in self.metric_names:
            if "loss_f" in model_out:
                loss_value = model_out["loss_f"].detach().cpu()
            else:
                # Compute MSE as fallback
                loss_value = F.mse_loss(x_reconstructed, x_original)
            self.metrics["loss_f"].update(loss_value)
        
        if "cosine_sim_f" in self.metric_names:
            x_reconstructed_norm = F.normalize(x_reconstructed, p=2, dim=-1)
            x_original_norm = F.normalize(x_original, p=2, dim=-1)
            cosine_sim = (x_reconstructed_norm * x_original_norm).sum(dim=-1).mean()
            self.metrics["cosine_sim_f"].update(cosine_sim)
        
        if "mse_f" in self.metric_names:
            mse = F.mse_loss(x_reconstructed, x_original)
            self.metrics["mse_f"].update(mse)
        
        # Discrepancy reconstruction metrics
        if "loss_d" in self.metric_names:
            if "loss_d" in model_out:
                loss_value = model_out["loss_d"].detach().cpu()
            else:
                # Compute MSE as fallback
                loss_value = F.mse_loss(x_discrepancy_pred, x_discrepancy_target)
            self.metrics["loss_d"].update(loss_value)
        
        if "cosine_sim_d" in self.metric_names:
            x_discrepancy_pred_norm = F.normalize(x_discrepancy_pred, p=2, dim=-1)
            x_discrepancy_target_norm = F.normalize(x_discrepancy_target, p=2, dim=-1)
            cosine_sim = (x_discrepancy_pred_norm * x_discrepancy_target_norm).sum(dim=-1).mean()
            self.metrics["cosine_sim_d"].update(cosine_sim)
        
        if "mse_d" in self.metric_names:
            mse = F.mse_loss(x_discrepancy_pred, x_discrepancy_target)
            self.metrics["mse_d"].update(mse)
        
        # Total loss
        if "total_loss" in self.metric_names:
            if "loss" in model_out:
                total_value = model_out["loss"].detach().cpu()
            else:
                # Compute combined MSE as fallback
                mse_f = F.mse_loss(x_reconstructed, x_original)
                mse_d = F.mse_loss(x_discrepancy_pred, x_discrepancy_target)
                total_value = (mse_f + mse_d) / 2
            self.metrics["total_loss"].update(total_value)

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


