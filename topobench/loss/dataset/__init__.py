"""Dataset-specific loss functions."""

from .bgrl_loss import BGRLLoss
from .dataset_loss import DatasetLoss
from .dgi_loss import DGILoss
from .dgmae_loss import DGMAELoss
from .grace_loss import GRACELoss
from .graphcl_loss import GraphCLLoss
from .graphmaev2_loss import GraphMAEv2Loss
from .mvgrl_loss import MVGRLInductiveLoss, MVGRLLoss, MVGRLTransductiveLoss
from .vgae_loss import VAELoss

__all__ = [
    "VAELoss",
    "GraphMAEv2Loss",
    "DGMAELoss",
    "MVGRLLoss",
    "MVGRLInductiveLoss",  # Alias for backward compatibility
    "MVGRLTransductiveLoss",  # Alias for backward compatibility
    "GRACELoss",
    "DGILoss",
    "GraphCLLoss",
    "BGRLLoss",
    "DatasetLoss",
]
