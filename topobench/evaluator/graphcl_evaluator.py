"""Evaluator for Graph Contrastive Learning (GraphCL) pre-training."""

import torch
import torch.nn.functional as F
from torchmetrics import MeanMetric, MetricCollection

from topobench.evaluator import AbstractEvaluator


class GraphCLEvaluator(AbstractEvaluator):
    r"""Evaluator for Graph Contrastive Learning (GraphCL) pre-training tasks.

    This evaluator computes metrics for self-supervised contrastive pre-training,
    tracking alignment and uniformity of learned representations.

    Parameters
    ----------
    metrics : list[str], optional
        List of metrics to compute. Available options:
        - "contrastive_loss": Mean contrastive loss (lower is better)
        - "alignment": Squared L2 distance between positive pairs (lower = better aligned)
        - "uniformity": Avg pairwise similarity on hypersphere (lower = more uniform)
        - "cosine_sim": Mean cosine similarity between positive pairs (higher is better)
        Default: ["contrastive_loss", "alignment", "cosine_sim"]
    """

    def __init__(self, metrics=None, **kwargs):
        if metrics is None:
            metrics = ["contrastive_loss", "alignment", "cosine_sim"]
        
        self.metric_names = metrics
        
        # Initialize metrics
        metrics_dict = {}
        for metric_name in metrics:
            if metric_name in ["contrastive_loss", "alignment", "uniformity", "cosine_sim"]:
                metrics_dict[metric_name] = MeanMetric()
            else:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. "
                    "Available metrics: 'contrastive_loss', 'alignment', 'uniformity', 'cosine_sim'"
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
            - z1_proj : torch.Tensor
                The projected embeddings from view 1.
            - z2_proj : torch.Tensor
                The projected embeddings from view 2.
            - loss : torch.Tensor (optional)
                The contrastive loss value.
        """
        z1 = model_out["z1_proj"].detach().cpu()
        z2 = model_out["z2_proj"].detach().cpu()
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Compute and update each metric
        if "contrastive_loss" in self.metric_names:
            if "loss" in model_out:
                loss_value = model_out["loss"].detach().cpu()
            else:
                # Compute simple contrastive loss
                pos_sim = (z1_norm * z2_norm).sum(dim=1).mean()
                loss_value = -pos_sim  # Simple approximation
            self.metrics["contrastive_loss"].update(loss_value)
        
        if "alignment" in self.metric_names:
            # Alignment loss: average squared L2 distance between positive pairs
            # Lower is better (means positive pairs are closer = better aligned)
            # From Wang & Isola 2020: "Understanding Contrastive Representation Learning"
            # Note: This measures "misalignment" - lower = better alignment achieved
            alignment = ((z1_norm - z2_norm) ** 2).sum(dim=1).mean()
            self.metrics["alignment"].update(alignment)
        
        if "uniformity" in self.metric_names:
            # Uniformity: measures how well embeddings are spread on hypersphere
            # Lower is better (embeddings should be spread out)
            # Using average pairwise similarity as proxy
            all_z = torch.cat([z1_norm, z2_norm], dim=0)
            similarity_matrix = torch.mm(all_z, all_z.t())
            # Exclude diagonal
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool)
            uniformity = similarity_matrix[mask].mean()
            self.metrics["uniformity"].update(uniformity)
        
        if "cosine_sim" in self.metric_names:
            # Mean cosine similarity between positive pairs
            cosine_sim = (z1_norm * z2_norm).sum(dim=1).mean()
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

