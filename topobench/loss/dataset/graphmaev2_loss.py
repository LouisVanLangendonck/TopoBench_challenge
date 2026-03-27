"""GraphMAEv2 Loss for reconstruction-based pre-training."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


def sce_loss(x, y, alpha=1):
    """Scaled Cosine Error loss."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GraphMAEv2Loss(AbstractLoss):
    r"""Loss function for GraphMAEv2 pre-training.

    Computes reconstruction loss and latent representation loss.

    Parameters
    ----------
    loss_type : str, optional
        Type of reconstruction loss: "sce" or "mse" (default: "sce").
    alpha : float, optional
        Alpha parameter for SCE loss (default: 2).
    lam : float, optional
        Weight for latent representation loss (default: 1.0).
    """

    def __init__(self, loss_type: str = "sce", alpha: float = 2, lam: float = 1.0):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.lam = lam

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GraphMAEv2 loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_reconstructed: Reconstructed features
            - x_original: Original features
            - latent_loss: Latent representation loss (optional)
        batch : torch_geometric.data.Data
            Batch object (not used directly).

        Returns
        -------
        torch.Tensor
            The computed total loss.
        """
        x_reconstructed = model_out["x_reconstructed"]
        x_original = model_out["x_original"]
        # Raw node features (e.g. ZINC atom indices) are often integer; SCE/MSE need floats.
        if not x_original.is_floating_point():
            x_original = x_original.to(dtype=x_reconstructed.dtype)
        latent_loss = model_out.get("latent_loss", torch.tensor(0.0, device=x_reconstructed.device))

        # Compute reconstruction loss
        if self.loss_type == "sce":
            recon_loss = sce_loss(x_reconstructed, x_original, alpha=self.alpha)
        elif self.loss_type == "mse":
            recon_loss = F.mse_loss(x_reconstructed, x_original)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Store individual losses in model_out for tracking
        model_out["loss_rec"] = recon_loss
        model_out["loss_latent"] = latent_loss

        # Total loss
        total_loss = recon_loss + self.lam * latent_loss

        return total_loss

