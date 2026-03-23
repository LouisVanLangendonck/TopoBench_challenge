"""BGRL loss for bootstrapped graph representation learning. Code adapted from https://github.com/nerdslab/bgrl/blob/main/bgrl"""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class BGRLLoss(AbstractLoss):
    r"""Symmetric BGRL loss based on negative cosine similarity."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _negative_cosine_similarity(
        pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        pred = F.normalize(pred, p=2, dim=-1)
        target = F.normalize(target, p=2, dim=-1)
        return -(pred * target).sum(dim=-1).mean()

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        pred_h_1 = model_out["pred_h_1"]
        pred_h_2 = model_out["pred_h_2"]
        target_h_1 = model_out["target_h_1"].detach()
        target_h_2 = model_out["target_h_2"].detach()

        loss_12 = self._negative_cosine_similarity(pred_h_1, target_h_2)
        loss_21 = self._negative_cosine_similarity(pred_h_2, target_h_1)
        loss = 0.5 * (loss_12 + loss_21)

        with torch.no_grad():
            cosine_12 = -loss_12
            cosine_21 = -loss_21
            cosine_sim = 0.5 * (cosine_12 + cosine_21)

        model_out["loss_12"] = loss_12
        model_out["loss_21"] = loss_21
        model_out["cosine_sim"] = cosine_sim

        return loss
