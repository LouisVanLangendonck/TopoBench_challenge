"""Minimal feature projection for graph (0-cell) inputs."""

import torch
import torch_geometric

from topobench.nn.encoders.base import AbstractFeatureEncoder


class LinearFeatureEncoder(AbstractFeatureEncoder):
    r"""One linear map per cell dimension: ``x_i`` :math:`\mapsto` ``out_channels``.

    Same config contract as :class:`~topobench.nn.encoders.all_cell_encoder.AllCellFeatureEncoder`
    (``in_channels`` list, scalar ``out_channels``, optional ``selected_dimensions``), but each
    cell encoder is a single ``nn.Linear`` plus optional dropout—no GraphNorm or second linear.

    For plain graphs, only ``x`` / ``x_0`` is used; missing ``x_0`` or ``batch_0`` is filled
    from ``x`` / ``batch`` like the all-cell encoder path expects.

    Parameters
    ----------
    in_channels : list[int]
        Per-dimension input widths (node features are ``in_channels[0]``).
    out_channels : int
        Output width for every encoded dimension.
    proj_dropout : float, optional
        Dropout applied after the linear (default 0).
    selected_dimensions : list[int], optional
        Subset of dimensions to encode; default is all ``len(in_channels)`` indices.
    **kwargs : dict
        Ignored.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        proj_dropout=0,
        selected_dimensions=None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = torch.nn.Dropout(proj_dropout)
        self.dimensions = (
            selected_dimensions
            if selected_dimensions is not None
            else range(len(self.in_channels))
        )
        for i in self.dimensions:
            in_i = int(self.in_channels[i])
            setattr(
                self,
                f"encoder_{i}",
                torch.nn.Linear(in_i, int(out_channels)),
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"dimensions={list(self.dimensions)})"
        )

    def forward(
        self, data: torch_geometric.data.Data
    ) -> torch_geometric.data.Data:
        if not hasattr(data, "x_0") and hasattr(data, "x"):
            data.x_0 = data.x
        if getattr(data, "batch", None) is not None:
            data.batch_0 = data.batch
        elif not hasattr(data, "batch_0"):
            data.batch_0 = torch.zeros(
                data.x_0.size(0),
                dtype=torch.long,
                device=data.x_0.device,
            )

        for i in self.dimensions:
            if hasattr(data, f"x_{i}") and hasattr(self, f"encoder_{i}"):
                h = getattr(self, f"encoder_{i}")(data[f"x_{i}"])
                data[f"x_{i}"] = self.dropout(h)
        return data
