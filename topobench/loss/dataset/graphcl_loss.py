"""Loss functions for Graph Contrastive Learning (GraphCL) pre-training."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class GraphCLLoss(AbstractLoss):
    r"""NT-Xent loss for Graph Contrastive Learning (GraphCL) pre-training.

    Builds the full 2N x 2N cosine-similarity matrix from two augmented views,
    masks out self-similarities, and applies cross-entropy so that each sample's
    positive (its other view) is pulled closer while all 2(N-1) negatives are
    pushed apart.

    Parameters
    ----------
    temperature : float, optional
        Temperature parameter for scaling similarities (default: 0.1).
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"

    def nt_xent_loss(self, z1, z2):
        """Compute NT-Xent loss between two views using the full 2N x 2N matrix.

        Follows the SimCLR/GraphCL formulation: concatenates both views into a
        2N-sized set, builds the full similarity matrix, masks out
        self-similarities, and treats the matching cross-view sample as the
        positive while all other 2(N-1) samples are negatives.

        Parameters
        ----------
        z1 : torch.Tensor
            Embeddings from first view, shape (batch_size, dim).
        z2 : torch.Tensor
            Embeddings from second view, shape (batch_size, dim).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = z1.size(0)

        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        representations = torch.cat([z1_norm, z2_norm], dim=0)  # (2N, dim)

        sim_matrix = torch.mm(representations, representations.t()) / self.temperature  # (2N, 2N)

        # Positive pair labels: (i, i+N) and (i+N, i)
        labels = torch.cat([
            torch.arange(batch_size, device=z1.device) + batch_size,
            torch.arange(batch_size, device=z1.device),
        ], dim=0)

        # Mask out self-similarities (diagonal)
        self_mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(self_mask, float('-inf'))

        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GraphCL contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - z1_proj: Projected graph embeddings from view 1
            - z2_proj: Projected graph embeddings from view 2
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (unused but kept for compatibility).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        z1 = model_out["z1_proj"]
        z2 = model_out["z2_proj"]
        return self.nt_xent_loss(z1, z2)

