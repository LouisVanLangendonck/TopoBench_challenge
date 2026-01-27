"""Hi-GMAE Loss for hierarchical masked autoencoding."""

import torch
import torch.nn.functional as F

from topobench.loss.base import AbstractLoss


def sce_loss(x, y, alpha=1):
    """Scaled Cosine Error loss."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class HiGMAELoss(AbstractLoss):
    r"""Loss function for Hi-GMAE (Hierarchical Graph Masked Autoencoder).

    This loss computes the reconstruction error between the predicted
    and original node features using either SCE (Scaled Cosine Error) or MSE.

    Parameters
    ----------
    loss_type : str, optional
        Type of loss: "sce" or "mse" (default: "sce").
    alpha : float, optional
        Alpha parameter for SCE loss (default: 1.0).
    """

    def __init__(self, loss_type: str = "sce", alpha: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha

    def forward(self, model_out: dict, batch) -> torch.Tensor:
        r"""Compute the Hi-GMAE reconstruction loss.

        Parameters
        ----------
        model_out : dict
            Model output containing:
            - x_reconstructed: Reconstructed features at masked positions
            - x_original: Original features at masked positions
        batch : torch_geometric.data.Data
            Batch data (not used directly here).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        x_reconstructed = model_out.get("x_reconstructed")
        x_original = model_out.get("x_original")

        if x_reconstructed is None or x_original is None:
            raise ValueError("Model output must contain 'x_reconstructed' and 'x_original'")

        # Compute reconstruction loss
        if self.loss_type == "sce":
            recon_loss = sce_loss(x_reconstructed, x_original, alpha=self.alpha)
        elif self.loss_type == "mse":
            recon_loss = F.mse_loss(x_reconstructed, x_original)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Store individual loss components in model_out as side effects
        model_out["recon_loss"] = recon_loss.detach()

        # Return scalar loss
        return recon_loss

