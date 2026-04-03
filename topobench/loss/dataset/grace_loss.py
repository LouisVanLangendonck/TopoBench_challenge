"""GRACE Loss for contrastive learning."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class GRACELoss(AbstractLoss):
    r"""Loss function for GRACE contrastive learning.

    Implements NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
    for contrastive learning between two augmented views.

    Parameters
    ----------
    tau : float, optional
        Temperature parameter for softmax (default: 0.5).
    batch_size : int, optional
        Batch size for memory-efficient computation (default: None).
        If None, computes full similarity matrix at once.
    """

    def __init__(self, tau: float = 0.5, batch_size: int = None):
        super().__init__()
        self.tau = tau
        self.batch_size = batch_size

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GRACE contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - h_1: Projected embeddings from view 1
            - h_2: Projected embeddings from view 2
        batch : torch_geometric.data.Data
            Batch object (not used directly).

        Returns
        -------
        torch.Tensor
            The computed contrastive loss.
        """
        h_1 = model_out["h_1"]
        h_2 = model_out["h_2"]

        # Normalize embeddings
        h_1 = F.normalize(h_1, p=2, dim=-1)
        h_2 = F.normalize(h_2, p=2, dim=-1)

        # Compute similarity matrices
        num_nodes = h_1.size(0)
        
        # Similarity between view 1 and view 2
        sim_12 = torch.mm(h_1, h_2.t()) / self.tau  # (N, N)
        sim_21 = torch.mm(h_2, h_1.t()) / self.tau  # (N, N)
        
        # Similarity within views (for negatives)
        sim_11 = torch.mm(h_1, h_1.t()) / self.tau  # (N, N)
        sim_22 = torch.mm(h_2, h_2.t()) / self.tau  # (N, N)
        
        # Create masks to exclude self-similarities
        mask = torch.eye(num_nodes, dtype=torch.bool, device=h_1.device)
        
        # Loss for view 1 -> view 2
        pos_sim_12 = torch.diag(sim_12)  # Positive pairs
        neg_sim_1 = torch.cat([sim_12, sim_11.masked_fill(mask, float("-inf"))], dim=1)
        loss_1 = -pos_sim_12 + torch.logsumexp(neg_sim_1, dim=1)
        
        # Loss for view 2 -> view 1
        pos_sim_21 = torch.diag(sim_21)  # Positive pairs
        neg_sim_2 = torch.cat([sim_21, sim_22.masked_fill(mask, float("-inf"))], dim=1)
        loss_2 = -pos_sim_21 + torch.logsumexp(neg_sim_2, dim=1)
        
        # Average loss
        loss = (loss_1.mean() + loss_2.mean()) / 2
        
        # Store individual losses for tracking
        model_out["loss_view1"] = loss_1.mean()
        model_out["loss_view2"] = loss_2.mean()

        return loss

