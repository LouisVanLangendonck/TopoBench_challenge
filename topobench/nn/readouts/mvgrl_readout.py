"""MVGRL Readout for contrastive multi-view pre-training.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
"""

import torch_geometric
from torch_geometric.utils import scatter

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class MVGRLReadOut(AbstractZeroCellReadOut):
    r"""MVGRL readout layer for contrastive pre-training.

    This readout handles the output from MVGRLWrapper.
    The wrapper already computes both node and graph representations.

    For pre-training, it passes through the representations.
    For downstream tasks, it computes logits from the appropriate representations.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Number of output classes.
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

        Parameters
        ----------
        model_out : dict
            Dictionary from MVGRLWrapper containing:
            - x_0: Node representations (h1 + h2)
            - x_graph: Graph representations (gv1_proj + gv2_proj)
            - contrastive_loss: The JSD-based contrastive loss
        batch : torch_geometric.data.Data
            Batch object.

        Returns
        -------
        dict
            Updated model output dictionary with logits for downstream tasks.
        """
        # Ensure x_graph is available for graph-level tasks
        if "x_graph" not in model_out and self.task_level == "graph":
            h = model_out["x_0"]
            batch_indices = model_out["batch_0"]
            model_out["x_graph"] = scatter(h, batch_indices, dim=0, reduce="sum")
        
        # Compute logits for downstream tasks
        if self.task_level == "graph":
            model_out["logits"] = self.linear(model_out["x_graph"])
        else:
            model_out["logits"] = self.linear(model_out["x_0"])
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.linear.out_features}, "
            f"task_level={self.task_level})"
        )


# Aliases for backward compatibility
MVGRLInductiveReadOut = MVGRLReadOut
MVGRLTransductiveReadOut = MVGRLReadOut
