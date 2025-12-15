"""Loss functions for GraphMAEv2 pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


def sce_loss(x, y, alpha=3):
    """Scaled Cosine Error (SCE) loss.
    
    Parameters
    ----------
    x : torch.Tensor
        Predicted features.
    y : torch.Tensor
        Target features.
    alpha : float, optional
        Scaling exponent (default: 3).
        
    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class GraphMAEv2Loss(AbstractLoss):
    r"""Loss function for GraphMAEv2 pre-training.

    This loss combines reconstruction loss and latent representation loss.
    
    Total Loss = Reconstruction Loss + lambda * Latent Loss

    Parameters
    ----------
    loss_type : str, optional
        Type of reconstruction loss. Options: "sce", "mse" (default: "sce").
    alpha : float, optional
        Alpha parameter for SCE loss (default: 2).
    lam : float, optional
        Weight for latent loss (default: 1.0).
    """

    def __init__(
        self, 
        loss_type: str = "sce", 
        alpha: float = 2,
        lam: float = 1.0
    ):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        self.lam = lam
        
        # Set up reconstruction loss criterion
        if loss_type == "sce":
            self.criterion = lambda x, y: sce_loss(x, y, alpha=self.alpha)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"Invalid loss type '{loss_type}'. "
                "Supported types: 'sce', 'mse'"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loss_type={self.loss_type}, alpha={self.alpha}, lam={self.lam})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GraphMAEv2 combined loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - x_reconstructed: Reconstructed features at masked positions
            - x_original: Original features at masked positions
            - latent_loss: Latent representation loss from wrapper
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        x_reconstructed = model_out["x_reconstructed"]
        x_original = model_out["x_original"]
        
        # Reconstruction loss
        loss_rec = self.criterion(x_reconstructed, x_original)
        
        # Latent loss (from wrapper)
        latent_loss = model_out.get("latent_loss", torch.tensor(0.0, device=x_reconstructed.device))
        
        # Combined loss
        total_loss = loss_rec + self.lam * latent_loss
        
        # Store individual losses for logging
        model_out["loss_rec"] = loss_rec
        model_out["loss_latent"] = latent_loss
        
        return total_loss

