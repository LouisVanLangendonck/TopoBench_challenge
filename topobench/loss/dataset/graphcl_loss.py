"""Loss functions for Graph Contrastive Learning (GraphCL) pre-training."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class GraphCLLoss(AbstractLoss):
    r"""NT-Xent loss for Graph Contrastive Learning (GraphCL) pre-training.

    Builds an N x N cross-view cosine-similarity matrix (z1 vs z2), masks out
    the positive pair on the diagonal, and computes:

        loss_i = -sim(z1_i, z2_i)/tau + log(sum_{j!=i} exp(sim(z1_i, z2_j)/tau))

    matching the official GraphCL implementation and paper Eq. 3.  The
    denominator contains only the N-1 cross-view negatives (excluding the
    positive).  The loss is asymmetric (z1 anchors, z2 targets).

    Parameters
    ----------
    temperature : float, optional
        Temperature parameter for scaling similarities (default: 0.2).
    """

    def __init__(self, temperature: float = 0.2):
        super().__init__()
        self.temperature = temperature

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature})"

    def nt_xent_loss(self, z1, z2):
        """Compute NT-Xent loss using the N x N cross-view-only matrix.

        Matches the official GraphCL ``loss_cal`` and paper Eq. 3: builds an
        N x N similarity matrix between the two views, excludes the positive
        (diagonal) from the denominator, and averages across the batch.

        Parameters
        ----------
        z1 : torch.Tensor
            Embeddings from first view (anchor), shape ``(N, d)``.
        z2 : torch.Tensor
            Embeddings from second view (target), shape ``(N, d)``.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = z1.size(0)

        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)

        sim_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature  # (N, N)

        pos_sim = sim_matrix.diag()  # (N,)

        # Mask diagonal (positive pairs) so they are excluded from the denominator
        diag_mask = torch.eye(batch_size, device=z1.device, dtype=torch.bool)
        neg_sim = sim_matrix.masked_fill(diag_mask, float('-inf'))

        # -log( exp(pos) / sum_neg ) = -pos + logsumexp(neg)
        loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        return loss.mean()

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

