"""BGRL readout with node-level predictor. Code adapted from https://github.com/nerdslab/bgrl/blob/main/bgrl"""

import torch.nn as nn
import torch_geometric

from topobench.nn.readouts.base import AbstractZeroCellReadOut


class BGRLReadOut(AbstractZeroCellReadOut):
    r"""BGRL readout that applies an MLP predictor on online embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        out_channels: int,
        predictor_hidden_dim: int = 512,
        task_level: str = "node",
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            out_channels=out_channels,
            task_level=task_level,
            logits_linear_layer=False,
            **kwargs,
        )
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.predictor_hidden_dim = predictor_hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, predictor_hidden_dim, bias=True),
            nn.PReLU(num_parameters=1),
            nn.Linear(predictor_hidden_dim, out_channels, bias=True),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()
            elif isinstance(module, nn.PReLU):
                module.weight.data.fill_(0.25)

    def forward(
        self, model_out: dict, batch: torch_geometric.data.Data
    ) -> dict:
        online_h_1 = model_out["online_h_1"]
        online_h_2 = model_out["online_h_2"]
        target_h_1 = model_out["target_h_1"]
        target_h_2 = model_out["target_h_2"]

        model_out["pred_h_1"] = self.predictor(online_h_1)
        model_out["pred_h_2"] = self.predictor(online_h_2)
        model_out["target_h_1"] = target_h_1
        model_out["target_h_2"] = target_h_2
        model_out["x_0"] = online_h_1
        return model_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"hidden_dim={self.hidden_dim}, "
            f"out_channels={self.out_channels}, "
            f"predictor_hidden_dim={self.predictor_hidden_dim})"
        )
