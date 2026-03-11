"""Dataset-specific loss functions."""

from .linkpred_loss import LinkPredLoss
from .graphmaev2_loss import GraphMAEv2Loss
from .grace_loss import GRACELoss
from .dgi_loss import DGILoss
from .graphcl_loss import GraphCLLoss
from .dataset_loss import DatasetLoss

__all__ = [
    "LinkPredLoss",
    "GraphMAEv2Loss",
    "GRACELoss",
    "DGILoss",
    "GraphCLLoss",
    "DatasetLoss",
]

