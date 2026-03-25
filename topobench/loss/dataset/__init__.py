"""Dataset-specific loss functions."""

from .vgae_loss import VAELoss
from .graphmaev2_loss import GraphMAEv2Loss
from .dgmae_loss import DGMAELoss
from .grace_loss import GRACELoss
from .dgi_loss import DGILoss
from .graphcl_loss import GraphCLLoss
from .bgrl_loss import BGRLLoss
from .dataset_loss import DatasetLoss

__all__ = [
    "VAELoss",
    "GraphMAEv2Loss",
    "DGMAELoss",
    "GRACELoss",
    "DGILoss",
    "GraphCLLoss",
    "BGRLLoss",
    "DatasetLoss",
]

