"""VGAE ELBO on sampled edges: reconstruction BCE + KL to standard normal."""

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class VAELoss(AbstractLoss):
    r"""BCE on edge logits plus optional :math:`\mathrm{KL}(q(z|x)\|p(z))` over nodes."""

    def __init__(self, kl_weight: float = 1.0, pos_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight
        self.pos_weight = pos_weight

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        logits = model_out["logits"]
        labels = model_out["labels"]

        if self.pos_weight != 1.0:
            pos_weight_tensor = torch.tensor([self.pos_weight], device=logits.device)
            recon = F.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight_tensor
            )
        else:
            recon = F.binary_cross_entropy_with_logits(logits, labels)

        logvar = model_out.get("logvar", None)
        if self.kl_weight != 0.0 and logvar is not None:
            mu = model_out["mu"]
            kl_per_node = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            kl = kl_per_node.mean()
            return recon + self.kl_weight * kl

        return recon
