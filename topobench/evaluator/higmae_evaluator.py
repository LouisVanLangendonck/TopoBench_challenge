"""Evaluator for Hi-GMAE pretraining tasks."""

import torch
import torch.nn.functional as F
from topobench.evaluator.base import AbstractEvaluator


class HiGMAEEvaluator(AbstractEvaluator):
    r"""Evaluator for Hi-GMAE (Hierarchical Graph Masked Autoencoder) pretraining.

    This evaluator tracks reconstruction loss and cosine similarity between
    reconstructed and original features.

    Parameters
    ----------
    **kwargs
        Additional parameters for the base evaluator.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        self.total_recon_loss = 0.0
        self.total_cosine_sim = 0.0
        self.num_samples = 0

    def update(self, model_out: dict):
        r"""Update metrics with current batch results.

        Parameters
        ----------
        model_out : dict
            Model output containing:
            - x_reconstructed: Reconstructed features
            - x_original: Original features
            - recon_loss: Reconstruction loss
            - batch: Batch data
        """
        x_reconstructed = model_out.get("x_reconstructed")
        x_original = model_out.get("x_original")
        recon_loss = model_out.get("recon_loss")

        if x_reconstructed is None or x_original is None:
            return

        # Track reconstruction loss
        if recon_loss is not None:
            self.total_recon_loss += recon_loss.item()

        # Compute cosine similarity
        x_recon_norm = F.normalize(x_reconstructed, p=2, dim=-1)
        x_orig_norm = F.normalize(x_original, p=2, dim=-1)
        cosine_sim = (x_recon_norm * x_orig_norm).sum(dim=-1).mean()
        self.total_cosine_sim += cosine_sim.item()

        self.num_samples += 1

    def compute(self) -> dict:
        r"""Compute final metrics.

        Returns
        -------
        dict
            Dictionary containing:
            - recon_loss: Mean reconstruction loss
            - cosine_sim: Mean cosine similarity
        """
        if self.num_samples == 0:
            return {
                "recon_loss": 0.0,
                "cosine_sim": 0.0,
            }

        return {
            "recon_loss": self.total_recon_loss / self.num_samples,
            "cosine_sim": self.total_cosine_sim / self.num_samples,
        }

