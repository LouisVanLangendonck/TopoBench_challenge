"""Loss functions for Graph Contrastive Learning (GraphCL) pre-training."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric

from topobench.loss.base import AbstractLoss


class GraphCLLoss(AbstractLoss):
    r"""Loss function for Graph Contrastive Learning (GraphCL) pre-training.

    This loss computes the NT-Xent (Normalized Temperature-scaled Cross Entropy)
    contrastive loss between two augmented views of graphs.

    Parameters
    ----------
    temperature : float, optional
        Temperature parameter for scaling similarities (default: 0.1).
    loss_type : str, optional
        Type of loss function. Options: "nt_xent", "infonce" (default: "nt_xent").
    """

    def __init__(self, temperature: float = 0.1, loss_type: str = "nt_xent"):
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(temperature={self.temperature}, loss_type={self.loss_type})"

    def nt_xent_loss(self, z1, z2):
        """Compute NT-Xent loss between two views.
        
        Parameters
        ----------
        z1 : torch.Tensor
            Embeddings from first view, shape (batch_size, dim).
        z2 : torch.Tensor
            Embeddings from second view, shape (batch_size, dim).
            
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Compute similarity matrix between all pairs
        # sim[i, j] = cosine_similarity(z1[i], z2[j])
        sim_matrix = torch.mm(z1_norm, z2_norm.t()) / self.temperature
        
        # Apply exp to get similarity scores
        sim_matrix = torch.exp(sim_matrix)
        
        # Positive pairs are on the diagonal (same graph, different views)
        pos_sim = sim_matrix.diag()
        
        # Negative pairs: all others in the batch
        # For each z1[i], negatives are all z2[j] where j != i
        loss = -torch.log(pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-8))
        
        return loss.mean()

    def infonce_loss(self, z1, z2):
        """Compute InfoNCE loss between two views.
        
        Parameters
        ----------
        z1 : torch.Tensor
            Embeddings from first view, shape (batch_size, dim).
        z2 : torch.Tensor
            Embeddings from second view, shape (batch_size, dim).
            
        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        batch_size = z1.size(0)
        
        # Normalize embeddings
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        
        # Concatenate for computing full similarity matrix
        representations = torch.cat([z1_norm, z2_norm], dim=0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(representations, representations.t()) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

    def forward(self, model_out: dict, batch: torch_geometric.data.Data):
        r"""Compute the GraphCL contrastive loss.

        Parameters
        ----------
        model_out : dict
            Dictionary containing the model output with keys:
            - z1_proj: Projected graph embeddings from view 1
            - z2_proj: Projected graph embeddings from view 2
        batch : torch_geometric.data.Data
            Batch object containing the batched domain data (unused but kept for compatibility).

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        z1 = model_out["z1_proj"]
        z2 = model_out["z2_proj"]
        
        if self.loss_type == "nt_xent":
            loss = self.nt_xent_loss(z1, z2)
        elif self.loss_type == "infonce":
            loss = self.infonce_loss(z1, z2)
        else:
            raise ValueError(
                f"Invalid loss type '{self.loss_type}'. "
                "Supported types: 'nt_xent', 'infonce'"
            )
        
        return loss

