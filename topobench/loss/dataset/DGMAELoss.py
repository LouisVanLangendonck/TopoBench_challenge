"""Loss functions for DGMAE pre-training."""

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


class DGMAELoss(AbstractLoss):
    r"""Loss function for DGMAE pre-training.

    This loss combines two components:
    1. Original feature reconstruction loss (L_f)
    2. Feature discrepancy reconstruction loss (L_d)
    
    Total Loss = (1 - λ) * L_f + λ * L_d
    
    Where:
    - L_f learns contextual representation by predicting masked node features
    - L_d preserves discrepancy information between nodes and neighbors

    Parameters
    ----------
    loss_type : str, optional
        Type of reconstruction loss. Options: "sce", "mse" (default: "sce").
    alpha_f : float, optional
        Alpha parameter for SCE loss in feature reconstruction (default: 2).
    alpha_d : float, optional
        Alpha parameter for SCE loss in discrepancy reconstruction (default: 2).
    lam : float, optional
        Weight for discrepancy loss (default: 0.5).
        Controls the balance between feature and discrepancy reconstruction.
    """

    def __init__(
        self, 
        loss_type: str = "sce", 
        alpha_f: float = 2,
        alpha_d: float = 2,
        lam: float = 0.5
    ):
        super().__init__()
        self.loss_type = loss_type
        self.alpha_f = alpha_f
        self.alpha_d = alpha_d
        self.lam = lam
        
        # Set up reconstruction loss criterion
        if loss_type == "sce":
            self.criterion_f = lambda x, y: sce_loss(x, y, alpha=self.alpha_f)
            self.criterion_d = lambda x, y: sce_loss(x, y, alpha=self.alpha_d)
        elif loss_type == "mse":
            self.criterion_f = nn.MSELoss()
            self.criterion_d = nn.MSELoss()
        else:
            raise ValueError(
                f"Invalid loss type '{loss_type}'. "
                "Supported types: 'sce', 'mse'"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loss_type={self.loss_type}, "
            f"alpha_f={self.alpha_f}, "
            f"alpha_d={self.alpha_d}, "
            f"lam={self.lam})"
        )

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the DGMAE combined loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - x_reconstructed: Reconstructed features at masked positions
            - x_original: Original features at masked positions
            - x_discrepancy_pred: Predicted feature discrepancy at unmasked positions
            - x_discrepancy_target: Target feature discrepancy at unmasked positions
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        # Branch 1: Original feature reconstruction loss (L_f)
        x_reconstructed = model_out["x_reconstructed"]
        x_original = model_out["x_original"]
        loss_f = self.criterion_f(x_reconstructed, x_original)
        
        # Branch 2: Feature discrepancy reconstruction loss (L_d)
        x_discrepancy_pred = model_out["x_discrepancy_pred"]
        x_discrepancy_target = model_out["x_discrepancy_target"]
        loss_d = self.criterion_d(x_discrepancy_pred, x_discrepancy_target)
        
        # Combined loss (Eq. 13)
        total_loss = (1 - self.lam) * loss_f + self.lam * loss_d
        
        # Store individual losses for logging
        model_out["loss_f"] = loss_f
        model_out["loss_d"] = loss_d
        
        return total_loss


