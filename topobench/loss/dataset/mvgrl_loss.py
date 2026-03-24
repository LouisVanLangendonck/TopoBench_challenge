"""MVGRL Loss for contrastive multi-view pre-training.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
"""

import torch
import torch_geometric

from topobench.loss.base import AbstractLoss


class MVGRLLoss(AbstractLoss):
    r"""Loss function for MVGRL pre-training.

    The contrastive loss is computed in the wrapper using the JSD-based
    mutual information estimator. This loss module simply extracts and
    returns that loss.

    The loss maximizes mutual information between:
    - Node representations from view 1 and graph representation from view 2
    - Node representations from view 2 and graph representation from view 1

    Parameters
    ----------
    None - loss is computed in the wrapper using the measure parameter.
    """

    def __init__(self):
        super().__init__()

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Return the MVGRL contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - contrastive_loss: The MI-based contrastive loss computed in wrapper
        batch : torch_geometric.data.Data
            Batch object (not used directly).

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
        loss = model_out["contrastive_loss"]
        
        # Store for tracking
        model_out["loss_contrastive"] = loss
        
        return loss
