"""MVGRL Loss for contrastive multi-view pre-training.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
https://arxiv.org/abs/2006.05582
"""

import math

import torch
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class MVGRLLoss(AbstractLoss):
    r"""MVGRL pre-training loss using Jensen-Shannon Divergence.

    Computes the contrastive loss between local (node) and global (graph)
    representations across two structural views.

    From the paper (Equation 5), the loss maximizes:
    - I(h^(1)_i, s^(2)): MI between node embeddings from view 1 and graph summary from view 2
    - I(h^(2)_i, s^(1)): MI between node embeddings from view 2 and graph summary from view 1

    The discriminator is implemented as a dot product (Section 3.2):
    D(h_n, h_g) = h_n · h_g^T

    Parameters
    ----------
    weight : float, optional
        Weight for this loss term (default: 1.0).
    """

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    def get_positive_expectation(self, p_samples, average=True):
        """Positive part of JS Divergence: log(2) - softplus(-x)"""
        log_2 = math.log(2.0)
        Ep = log_2 - F.softplus(-p_samples)
        return Ep.mean() if average else Ep

    def get_negative_expectation(self, q_samples, average=True):
        """Negative part of JS Divergence: softplus(-x) + x - log(2)"""
        log_2 = math.log(2.0)
        Eq = F.softplus(-q_samples) + q_samples - log_2
        return Eq.mean() if average else Eq

    def local_global_loss(self, l_enc, g_enc, batch_indices, num_graphs):
        """Compute local-global contrastive loss using JSD estimator.

        For cross_graph (inductive): negatives are nodes paired with graphs
        they don't belong to.
        """
        num_nodes = l_enc.shape[0]
        device = l_enc.device

        # Create masks for positive/negative pairs
        pos_mask = torch.zeros((num_nodes, num_graphs), device=device)
        neg_mask = torch.ones((num_nodes, num_graphs), device=device)

        for nodeidx, graphidx in enumerate(batch_indices):
            pos_mask[nodeidx][graphidx] = 1.0
            neg_mask[nodeidx][graphidx] = 0.0

        # Dot product discriminator (from paper Section 3.2)
        res = torch.mm(l_enc, g_enc.t())

        # JSD estimator
        E_pos = self.get_positive_expectation(
            res * pos_mask, average=False
        ).sum()
        E_pos = E_pos / num_nodes

        E_neg = self.get_negative_expectation(
            res * neg_mask, average=False
        ).sum()
        E_neg = (
            E_neg / (num_nodes * (num_graphs - 1))
            if num_graphs > 1
            else E_neg / num_nodes
        )

        return E_neg - E_pos

    def local_global_loss_with_shuffle(
        self, l_enc_real, l_enc_shuffled, g_enc, batch_indices, num_graphs
    ):
        """Compute loss for transductive setting with shuffled negatives.

        Positives: real node representations with graph summary
        Negatives: shuffled node representations with graph summary
        """
        num_nodes = l_enc_real.shape[0]
        device = l_enc_real.device

        # Create mask for which nodes belong to which graphs
        pos_mask = torch.zeros((num_nodes, num_graphs), device=device)
        for nodeidx, graphidx in enumerate(batch_indices):
            pos_mask[nodeidx][graphidx] = 1.0

        # Dot product scores
        res_positive = torch.mm(l_enc_real, g_enc.t())
        res_negative = torch.mm(l_enc_shuffled, g_enc.t())

        # Only consider scores where node belongs to graph
        E_pos = self.get_positive_expectation(
            res_positive * pos_mask, average=False
        ).sum()
        E_pos = E_pos / num_nodes

        E_neg = self.get_negative_expectation(
            res_negative * pos_mask, average=False
        ).sum()
        E_neg = E_neg / num_nodes

        return E_neg - E_pos

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> torch.Tensor:
        r"""Compute the MVGRL contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary from MVGRLWrapper containing:
            - lv1_proj: Projected node embeddings from view 1
            - lv2_proj: Projected node embeddings from view 2
            - gv1_proj: Projected graph embeddings from view 1
            - gv2_proj: Projected graph embeddings from view 2
            - batch_0: Batch indices for nodes
            - num_graphs: Number of graphs in batch
            - negative_sampling: "cross_graph" or "feature_shuffle"
            - lv1_shuf_proj, lv2_shuf_proj: (only for feature_shuffle)
        batch : torch_geometric.data.Data
            Batch object.

        Returns
        -------
        torch.Tensor
            The JSD-based contrastive loss (sum of two terms).
        """
        lv1_proj = model_out["lv1_proj"]
        lv2_proj = model_out["lv2_proj"]
        gv1_proj = model_out["gv1_proj"]
        gv2_proj = model_out["gv2_proj"]
        batch_indices = model_out["batch_0"]
        num_graphs = model_out["num_graphs"]
        negative_sampling = model_out.get("negative_sampling", "cross_graph")

        if negative_sampling == "cross_graph":
            # Inductive: negatives from other graphs in batch
            # Term 1: I(h^(1), s^(2)) - nodes from view 1 with graph from view 2
            loss1 = self.local_global_loss(
                lv1_proj, gv2_proj, batch_indices, num_graphs
            )
            # Term 2: I(h^(2), s^(1)) - nodes from view 2 with graph from view 1
            loss2 = self.local_global_loss(
                lv2_proj, gv1_proj, batch_indices, num_graphs
            )
        else:
            # Transductive: negatives from shuffled features
            lv1_shuf_proj = model_out["lv1_shuf_proj"]
            lv2_shuf_proj = model_out["lv2_shuf_proj"]

            # Term 1: real view1 nodes vs shuffled view2 nodes, with graph from view 2
            loss1 = self.local_global_loss_with_shuffle(
                lv1_proj, lv2_shuf_proj, gv2_proj, batch_indices, num_graphs
            )
            # Term 2: real view2 nodes vs shuffled view1 nodes, with graph from view 1
            loss2 = self.local_global_loss_with_shuffle(
                lv2_proj, lv1_shuf_proj, gv1_proj, batch_indices, num_graphs
            )

        # Store individual losses for tracking
        model_out["loss_term1"] = loss1
        model_out["loss_term2"] = loss2

        total_loss = loss1 + loss2
        return self.weight * total_loss

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight})"


# Aliases for backward compatibility
MVGRLInductiveLoss = MVGRLLoss
MVGRLTransductiveLoss = MVGRLLoss
