"""Loss functions for GraphMAE pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


def sce_loss(x, y, alpha=3):
    """Scaled Cosine Error (SCE) loss.
    
    This loss function normalizes both predictions and targets, then computes
    the scaled cosine distance between them.
    
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
    
    # Compute 1 - cosine similarity, then apply power scaling
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    
    loss = loss.mean()
    return loss


def sig_loss(x, y):
    """Sigmoid loss.
    
    Parameters
    ----------
    x : torch.Tensor
        Predicted features.
    y : torch.Tensor
        Target features.
        
    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    
    loss = (x * y).sum(1)
    loss = torch.sigmoid(-loss)
    loss = loss.mean()
    return loss


class GraphMAELoss(AbstractLoss):
    r"""Loss function for GraphMAE pre-training.

    This loss computes the reconstruction error between the original
    and reconstructed node features at masked positions.

    Parameters
    ----------
    loss_type : str, optional
        Type of reconstruction loss. Options: "sce", "mse", "mae" (default: "sce").
    alpha : float, optional
        Alpha parameter for SCE loss (default: 3).
    """

    def __init__(self, loss_type: str = "sce", alpha: float = 3):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha
        
        # Set up loss criterion
        if loss_type == "sce":
            self.criterion = lambda x, y: sce_loss(x, y, alpha=self.alpha)
        elif loss_type == "mse":
            self.criterion = nn.MSELoss()
        elif loss_type == "mae":
            self.criterion = nn.L1Loss()
        elif loss_type == "sig":
            self.criterion = sig_loss
        else:
            raise ValueError(
                f"Invalid loss type '{loss_type}'. "
                "Supported types: 'sce', 'mse', 'mae', 'sig'"
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(loss_type={self.loss_type}, alpha={self.alpha})"

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GraphMAE reconstruction loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - x_reconstructed: Reconstructed features at masked positions
            - x_original: Original features at masked positions
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (unused but kept for compatibility).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        x_reconstructed = model_out["x_reconstructed"]
        x_original = model_out["x_original"]
        
        # Compute reconstruction loss
        loss = self.criterion(x_reconstructed, x_original)
        
        return loss