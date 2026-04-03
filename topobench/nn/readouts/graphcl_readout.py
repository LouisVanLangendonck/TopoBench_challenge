"""GraphCL Readout with Projection Head for Graph Contrastive Learning."""

import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class GraphCLReadOut(AbstractZeroCellReadOut):
    r"""GraphCL readout layer with projection head for contrastive learning.

    This readout implements the projection head for GraphCL,
    projecting graph embeddings to a lower-dimensional space for contrastive loss.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension from the encoder.
    out_channels : int
        Output projection dimension.
    projection_type : str, optional
        Type of projection head: "linear", "mlp", or "none" (default: "mlp").
    projection_hidden_dim : int, optional
        Hidden dimension for MLP projection head (default: 256).
    task_level : str, optional
        Task level (default: "graph").
    **kwargs : dict
        Additional arguments.
    """

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        projection_type: str = "mlp",
        projection_hidden_dim: int = None,
        task_level: str = "graph",
        **kwargs
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,  # We handle our own projection
            **kwargs
        )
        
        self.projection_type = projection_type
        
        # Default projection hidden dim to same as hidden_dim
        if projection_hidden_dim is None:
            projection_hidden_dim = hidden_dim
        
        # Build projection head
        self.projection_head = self._build_projection_head(
            projection_type, hidden_dim, out_channels, projection_hidden_dim
        )
    
    def _build_projection_head(
        self,
        projection_type: str,
        in_dim: int,
        out_dim: int,
        hidden_dim: int
    ) -> nn.Module:
        """Build the projection head module.
        
        Parameters
        ----------
        projection_type : str
            Type of projection head ("linear", "mlp", or "none").
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Hidden dimension for MLP.
            
        Returns
        -------
        nn.Module
            The projection head module.
        """
        if projection_type == "none":
            return nn.Identity()
        elif projection_type == "linear":
            return nn.Linear(in_dim, out_dim)
        elif projection_type == "mlp":
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            raise ValueError(
                f"Unknown projection type: {projection_type}. "
                "Available options: 'none', 'linear', 'mlp'"
            )
    
    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        r"""Forward pass for GraphCL projection.

        Parameters
        ----------
        model_out : dict
            Dictionary containing:
            - z1: Graph embedding from first augmented view
            - z2: Graph embedding from second augmented view
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data.

        Returns
        -------
        dict
            Dictionary containing:
            - z1_proj: Projected embedding from view 1
            - z2_proj: Projected embedding from view 2
            - logits: z1_proj (for compatibility)
            Plus all original model_out keys
        """
        # Get graph embeddings from both views
        z1 = model_out["z1"]
        z2 = model_out["z2"]
        
        # Project through projection head
        z1_proj = self.projection_head(z1)
        z2_proj = self.projection_head(z2)
        
        # Update model output
        model_out["z1_proj"] = z1_proj
        model_out["z2_proj"] = z2_proj
        
        # Logits for compatibility
        model_out["logits"] = z1_proj
        
        return model_out
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"projection_type={self.projection_type}, "
            f"task_level={self.task_level})"
        )

