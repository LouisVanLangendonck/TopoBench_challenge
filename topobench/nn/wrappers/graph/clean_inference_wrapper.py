"""Clean inference wrapper for downstream evaluation.

This wrapper preserves the architectural components (LayerNorms, residual connections)
that were present during pre-training, but removes all pre-training specific logic
(masking, augmentation, EMA, etc.).

This ensures the backbone sees the same architecture/feature distribution as during
pre-training, preventing negative transfer.
"""

import torch.nn as nn


class CleanInferenceWrapper(nn.Module):
    """Clean wrapper that preserves architecture but removes pre-training logic.

    This wrapper:
    - ✅ Keeps LayerNorm layers (ln_i)
    - ✅ Keeps residual connections
    - ❌ Removes masking, augmentation, EMA, projectors, etc.

    Parameters
    ----------
    backbone : nn.Module
        The GNN backbone.
    out_channels : int
        Output dimension.
    num_cell_dimensions : int
        Number of cell dimensions (for creating ln_i layers).
    residual_connections : bool, optional
        Whether to use residual connections (default: True).
    manual_iteration : bool, optional
        Whether to manually iterate through backbone.convs (S2GAE-style).
        If False, calls backbone() directly (default for DGI, GraphCL, etc.).
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_channels: int,
        num_cell_dimensions: int,
        residual_connections: bool = True,
        manual_iteration: bool = False,
    ):
        super().__init__()
        self.backbone = backbone
        self.out_channels = out_channels
        self.dimensions = range(num_cell_dimensions)
        self.residual_connections = residual_connections
        self.manual_iteration = manual_iteration

        # Create LayerNorm layers (same as AbstractWrapper)
        for i in self.dimensions:
            setattr(self, f"ln_{i}", nn.LayerNorm(out_channels))

    def forward(self, x, edge_index, batch=None, edge_weight=None):
        """Forward pass - clean encoding without pre-training logic.

        Parameters
        ----------
        x : torch.Tensor
            Node features.
        edge_index : torch.Tensor
            Edge indices.
        batch : torch.Tensor, optional
            Batch indices.
        edge_weight : torch.Tensor, optional
            Edge weights.

        Returns
        -------
        torch.Tensor
            Node embeddings.
        """
        # DEBUG: Print once to verify correct forward path
        if not hasattr(self, "_debug_printed"):
            print("[DEBUG CleanInferenceWrapper]")
            print(f"  Input shape: {x.shape}")
            print(f"  Backbone type: {type(self.backbone).__name__}")
            print(
                f"  Has 'convs' attribute: {hasattr(self.backbone, 'convs')}"
            )
            print(f"  Residual connections: {self.residual_connections}")
            self._debug_printed = True

        # Store input for residual connection
        x_input = x

        # Backbone encoding (no masking, no augmentation)
        # IMPORTANT: We must use the SAME forward path as during pre-training
        # - S2GAE: manually iterates through layers with intermediate activations
        # - DGI/GraphCL/etc.: calls backbone() directly
        if self.manual_iteration and hasattr(self.backbone, "convs"):
            # Use S2GAE's manual layer iteration (same as pre-training)
            import torch.nn.functional as F

            x_out = x
            for i, conv in enumerate(self.backbone.convs):
                try:
                    # Try with edge_weight first (for GCN, SAGE, etc.)
                    if edge_weight is not None:
                        x_out = conv(
                            x_out, edge_index, edge_weight=edge_weight
                        )
                    else:
                        x_out = conv(x_out, edge_index)
                except TypeError:
                    # If edge_weight not supported (GIN, etc.), call without it
                    x_out = conv(x_out, edge_index)

                # Apply activation and dropout between layers (same as S2GAE)
                if i < len(self.backbone.convs) - 1:
                    x_out = F.relu(x_out)
                    if hasattr(self.backbone, "dropout"):
                        x_out = F.dropout(
                            x_out,
                            p=self.backbone.dropout,
                            training=self.training,
                        )
        else:
            # For non-S2GAE models (DGI, GraphCL, etc.), use standard forward
            x_out = self.backbone(
                x, edge_index, batch=batch, edge_weight=edge_weight
            )

        # DEBUG: Check backbone output
        if not hasattr(self, "_debug_backbone_printed"):
            print(f"  After backbone: {x_out.shape}")
            self._debug_backbone_printed = True

        # Residual connection + LayerNorm (if enabled)
        if self.residual_connections:
            # Add residual
            x_out = x_out + x_input
            # Apply LayerNorm
            x_out = self.ln_0(x_out)

            # DEBUG: Check after residual + LayerNorm
            if not hasattr(self, "_debug_residual_printed"):
                print(f"  After residual + LayerNorm: {x_out.shape}")
                print(
                    f"  Output mean: {x_out.mean().item():.4f}, std: {x_out.std().item():.4f}"
                )
                self._debug_residual_printed = True

        return x_out
