"""MVGRL Readout for contrastive multi-view pre-training.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
"""

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.utils import scatter

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class MVGRLReadOut(AbstractZeroCellReadOut):
    r"""MVGRL readout layer for contrastive pre-training.

    This readout handles the output from MVGRLGNNWrapper.
    The wrapper already computes graph-level representations using
    JK-Net style pooling (sum + concatenate across layers) followed by MLP projection.

    For pre-training, it passes through the representations.
    For downstream tasks, it computes logits from the graph representations.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder (after MLP projection).
    out_channels : int
        Number of output classes (for downstream tasks).
    task_level : str, optional
        Task level: "node" or "graph" (default: "graph").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        task_level: str = "graph",
        **kwargs
    ):
        # Use logits_linear_layer=True to create the linear projection for downstream tasks
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            pooling_type="sum",
            logits_linear_layer=True,
            **kwargs
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for MVGRL readout.

        The wrapper provides:
        - x_0: Node representations from view 1 (final GCN layer)
        - x_graph: Combined projected graph representations (gv1_proj + gv2_proj)
        - contrastive_loss: The JSD-based contrastive loss

        Parameters
        ----------
        model_out : dict
            Dictionary from MVGRLGNNWrapper containing representations and loss.
        batch : torch_geometric.data.Data
            Batch object.

        Returns
        -------
        dict
            Updated model output dictionary with logits for downstream tasks.
        """
        # Ensure x_graph is available for graph-level tasks
        if "x_graph" not in model_out and self.task_level == "graph":
            # Fallback: compute graph representation from node representations
            h = model_out["x_0"]
            batch_indices = model_out["batch_0"]
            model_out["x_graph"] = scatter(h, batch_indices, dim=0, reduce="sum")
        
        # Compute logits for downstream tasks
        # Override the base class behavior to use x_graph directly for graph tasks
        if self.task_level == "graph":
            # Use the pre-computed graph representations (already pooled)
            model_out["logits"] = self.linear(model_out["x_graph"])
        else:
            # For node-level tasks, use x_0 (node representations)
            model_out["logits"] = self.linear(model_out["x_0"])
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.linear.out_features}, "
            f"task_level={self.task_level})"
        )
