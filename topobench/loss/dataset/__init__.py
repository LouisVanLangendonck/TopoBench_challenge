"""Dataset-specific loss functions."""

from .linkpred_loss import LinkPredLoss
from .graphmaev2_loss import GraphMAEv2Loss
from .dgmae_loss import DGMAELoss
from .grace_loss import GRACELoss
from .dgi_loss import DGILoss
from .dataset_loss import DatasetLoss

__all__ = [
    "LinkPredLoss",
    "GraphMAEv2Loss",
    "DGMAELoss",
    "GRACELoss",
    "DGILoss",
    "DatasetLoss",
]

