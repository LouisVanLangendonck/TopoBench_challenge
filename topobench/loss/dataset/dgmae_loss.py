"""DGMAE Loss for reconstruction-based pre-training with heterophily.

Based on: https://github.com/zhengziyu77/DGMAE
Paper: "Discrepancy-Aware Graph Mask Auto-Encoder" (KDD 2025)
"""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


def sce_loss(x, y, alpha=3):
    """Scaled Cosine Error loss."""
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class DGMAELoss(AbstractLoss):
    r"""Loss function for DGMAE pre-training.

    Computes (Equation 13 from paper):
        L = (1-λ)·L_f + λ·L_d
    
    Where:
    - L_f: Feature reconstruction loss (on masked nodes)
    - L_d: Discrepancy reconstruction loss (on unmasked/keep nodes)
    - λ: Balance parameter (low for homophilic, high for heterophilic graphs)

    The discrepancy loss encourages:
        z^D = z - ẑ ≈ x^D (high-pass filtered features)

    Parameters
    ----------
    loss_type : str, optional
        Type of loss: "sce" or "mse" (default: "sce").
    alpha_recon : float, optional
        Alpha parameter (γ₁) for reconstruction SCE loss (default: 2).
    alpha_hetero : float, optional
        Alpha parameter (γ₂) for discrepancy SCE loss (default: 4).
    lam : float, optional
        Lambda parameter for loss weighting (default: 0.1).
        - Small λ (e.g., 0.1): for homophilic graphs
        - Large λ (e.g., 0.7-0.8): for heterophilic graphs
    """

    def __init__(
        self,
        loss_type: str = "sce",
        alpha_recon: float = 2,
        alpha_hetero: float = 4,
        lam: float = 0.1,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.alpha_recon = alpha_recon
        self.alpha_hetero = alpha_hetero
        self.lam = lam

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the DGMAE loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - x_reconstructed: Reconstructed features at mask_nodes
            - x_original: Original features at mask_nodes
            - x_reconstructed_full: Full reconstruction (all nodes) [optional]
            - high_pred: MLP prediction of high-freq component
            - high_pass_features: High-pass filtered original features
            - keep_nodes: Indices of non-masked nodes
        batch : torch_geometric.data.Data
            Batch object (not used directly).

        Returns
        -------
        torch.Tensor
            The computed total loss.
        """
        x_reconstructed = model_out["x_reconstructed"]
        x_original = model_out["x_original"]
        keep_nodes = model_out["keep_nodes"]
        
        device = x_reconstructed.device

        # 1. Reconstruction loss (on masked nodes)
        if self.loss_type == "sce":
            recon_loss = sce_loss(x_reconstructed, x_original, alpha=self.alpha_recon)
        elif self.loss_type == "mse":
            recon_loss = F.mse_loss(x_reconstructed, x_original)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # 2. Heterophily/Discrepancy loss (on keep nodes)
        hetero_loss = torch.tensor(0.0, device=device)
        
        high_pred = model_out.get("high_pred")
        high_pass_features = model_out.get("high_pass_features")
        x_reconstructed_full = model_out.get("x_reconstructed_full")
        
        if high_pred is not None and high_pass_features is not None:
            if x_reconstructed_full is not None:
                # Full DGMAE heterophily loss
                # diff = high_pred - x_reconstructed_full
                # hetero_loss = sce_loss(diff[keep_nodes], high_pass_features[keep_nodes])
                diff = high_pred - x_reconstructed_full
                
                if self.loss_type == "sce":
                    hetero_loss = sce_loss(
                        diff[keep_nodes],
                        high_pass_features[keep_nodes],
                        alpha=self.alpha_hetero
                    )
                else:
                    hetero_loss = F.mse_loss(
                        diff[keep_nodes],
                        high_pass_features[keep_nodes]
                    )
            else:
                # Simplified heterophily loss when full reconstruction not available
                # Compare high_pred directly to high_pass_features on keep_nodes
                if self.loss_type == "sce":
                    hetero_loss = sce_loss(
                        high_pred[keep_nodes],
                        high_pass_features[keep_nodes],
                        alpha=self.alpha_hetero
                    )
                else:
                    hetero_loss = F.mse_loss(
                        high_pred[keep_nodes],
                        high_pass_features[keep_nodes]
                    )

        # Store individual losses in model_out for tracking
        model_out["loss_recon"] = recon_loss
        model_out["loss_hetero"] = hetero_loss

        # Total loss: Equation (13) from paper
        # L = (1-λ)·L_f + λ·L_d
        total_loss = (1 - self.lam) * recon_loss + self.lam * hetero_loss

        return total_loss
