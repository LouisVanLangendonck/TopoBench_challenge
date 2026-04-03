"""Personalized PageRank (PPR) Diffusion Transform.

Based on: https://github.com/kavehhassani/mvgrl
Paper: "Contrastive Multi-View Representation Learning on Graphs" (ICML 2020)
"""

import numpy as np
import torch
from scipy.linalg import fractional_matrix_power, inv
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_adj, add_self_loops


class PPRDiffusion(BaseTransform):
    r"""Personalized PageRank (PPR) Diffusion transform.

    Computes the PPR diffusion matrix and stores it as sparse edge_index
    and edge_weight attributes on the data object. This is used for
    multi-view contrastive learning methods like MVGRL.

    The PPR diffusion matrix is computed as:
        S^{PPR} = α(I - (1-α)Ã)^{-1}
    
    where Ã = D^{-1/2} A D^{-1/2} is the symmetrically normalized adjacency.

    Parameters
    ----------
    alpha : float, optional
        Teleport probability (default: 0.2).
    threshold : float, optional
        Minimum edge weight to keep after sparsification (default: 1e-4).
    self_loop : bool, optional
        Whether to add self-loops before computing diffusion (default: True).
    attr_name_edge_index : str, optional
        Attribute name for storing diffusion edge indices (default: "edge_index_diff").
    attr_name_edge_weight : str, optional
        Attribute name for storing diffusion edge weights (default: "edge_weight_diff").
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        alpha: float = 0.2,
        threshold: float = 1e-4,
        self_loop: bool = True,
        attr_name_edge_index: str = "edge_index_diff",
        attr_name_edge_weight: str = "edge_weight_diff",
        **kwargs,
    ):
        self.alpha = alpha
        self.threshold = threshold
        self.self_loop = self_loop
        self.attr_name_edge_index = attr_name_edge_index
        self.attr_name_edge_weight = attr_name_edge_weight

    def forward(self, data: Data) -> Data:
        """Compute the PPR diffusion and store as sparse edge format.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with PPR diffusion edges added.
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = edge_index.device

        # Handle empty graphs
        if edge_index.size(1) == 0 or num_nodes <= 1:
            setattr(data, self.attr_name_edge_index, edge_index)
            setattr(data, self.attr_name_edge_weight, torch.ones(edge_index.size(1), device=device))
            return data

        # Compute PPR diffusion matrix
        diff_edge_index, diff_edge_weight = self._compute_ppr_sparse(
            edge_index, num_nodes, device
        )

        # Store on data object
        setattr(data, self.attr_name_edge_index, diff_edge_index)
        setattr(data, self.attr_name_edge_weight, diff_edge_weight)

        return data

    def _compute_ppr(self, adj: np.ndarray) -> np.ndarray:
        """Compute PPR diffusion matrix.
        
        Matches original MVGRL implementation:
        S^{PPR} = α(I - (1-α)Ã)^{-1}

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix (with self-loops if self_loop=True).

        Returns
        -------
        np.ndarray
            PPR diffusion matrix.
        """
        n = adj.shape[0]
        d = np.diag(np.sum(adj, axis=1))
        
        # Handle isolated nodes (degree 0)
        d_diag = np.diag(d)
        d_diag[d_diag == 0] = 1  # Prevent division by zero
        d = np.diag(d_diag)
        
        dinv = fractional_matrix_power(d, -0.5)
        at = np.matmul(np.matmul(dinv, adj), dinv)  # Ã = D^{-1/2} A D^{-1/2}
        
        return self.alpha * inv(np.eye(n) - (1 - self.alpha) * at)

    def _compute_ppr_sparse(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> tuple:
        """Compute PPR and convert to sparse edge format.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.

        Returns
        -------
        tuple
            (edge_index_diff, edge_weight_diff)
        """
        # Convert to dense adjacency
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).cpu().numpy()
        
        # Add self-loops if requested
        if self.self_loop:
            adj = adj + np.eye(num_nodes)
        
        # Compute PPR diffusion
        ppr_matrix = self._compute_ppr(adj)
        
        # Convert to torch tensor
        ppr_tensor = torch.from_numpy(ppr_matrix).float().to(device)
        
        # Sparsify: keep edges above threshold
        mask = ppr_tensor > self.threshold
        edge_index_diff = mask.nonzero(as_tuple=False).t().contiguous()
        edge_weight_diff = ppr_tensor[mask]
        
        return edge_index_diff, edge_weight_diff

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"alpha={self.alpha}, "
            f"threshold={self.threshold}, "
            f"self_loop={self.self_loop})"
        )


class HeatDiffusion(BaseTransform):
    r"""Heat Kernel Diffusion transform.

    Computes the heat kernel diffusion matrix and stores it as sparse
    edge_index and edge_weight attributes on the data object.

    The heat kernel diffusion matrix is computed as:
        S^{heat} = exp(t * (AD^{-1} - 1))  (element-wise)

    Parameters
    ----------
    t : float, optional
        Diffusion time (default: 5.0).
    threshold : float, optional
        Minimum edge weight to keep after sparsification (default: 1e-4).
    self_loop : bool, optional
        Whether to add self-loops before computing diffusion (default: True).
    attr_name_edge_index : str, optional
        Attribute name for storing diffusion edge indices (default: "edge_index_diff").
    attr_name_edge_weight : str, optional
        Attribute name for storing diffusion edge weights (default: "edge_weight_diff").
    **kwargs : dict
        Additional arguments (not used).
    """

    def __init__(
        self,
        t: float = 5.0,
        threshold: float = 1e-4,
        self_loop: bool = True,
        attr_name_edge_index: str = "edge_index_diff",
        attr_name_edge_weight: str = "edge_weight_diff",
        **kwargs,
    ):
        self.t = t
        self.threshold = threshold
        self.self_loop = self_loop
        self.attr_name_edge_index = attr_name_edge_index
        self.attr_name_edge_weight = attr_name_edge_weight

    def forward(self, data: Data) -> Data:
        """Compute the heat kernel diffusion and store as sparse edge format.

        Parameters
        ----------
        data : Data
            Input graph data object.

        Returns
        -------
        Data
            Graph data object with heat diffusion edges added.
        """
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        device = edge_index.device

        # Handle empty graphs
        if edge_index.size(1) == 0 or num_nodes <= 1:
            setattr(data, self.attr_name_edge_index, edge_index)
            setattr(data, self.attr_name_edge_weight, torch.ones(edge_index.size(1), device=device))
            return data

        # Compute heat diffusion matrix
        diff_edge_index, diff_edge_weight = self._compute_heat_sparse(
            edge_index, num_nodes, device
        )

        # Store on data object
        setattr(data, self.attr_name_edge_index, diff_edge_index)
        setattr(data, self.attr_name_edge_weight, diff_edge_weight)

        return data

    def _compute_heat(self, adj: np.ndarray) -> np.ndarray:
        """Compute heat kernel diffusion matrix.
        
        Matches original MVGRL implementation:
        S^{heat} = exp(t * (AD^{-1} - 1))  (element-wise)

        Parameters
        ----------
        adj : np.ndarray
            Adjacency matrix (with self-loops if self_loop=True).

        Returns
        -------
        np.ndarray
            Heat kernel diffusion matrix.
        """
        d = np.diag(np.sum(adj, axis=1))
        
        # Handle isolated nodes (degree 0)
        d_diag = np.diag(d)
        d_diag[d_diag == 0] = 1  # Prevent division by zero
        d = np.diag(d_diag)
        
        return np.exp(self.t * (np.matmul(adj, inv(d)) - 1))

    def _compute_heat_sparse(
        self, edge_index: torch.Tensor, num_nodes: int, device: torch.device
    ) -> tuple:
        """Compute heat kernel and convert to sparse edge format.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge indices of the graph.
        num_nodes : int
            Number of nodes in the graph.
        device : torch.device
            Device to use.

        Returns
        -------
        tuple
            (edge_index_diff, edge_weight_diff)
        """
        # Convert to dense adjacency
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).cpu().numpy()
        
        # Add self-loops if requested
        if self.self_loop:
            adj = adj + np.eye(num_nodes)
        
        # Compute heat diffusion
        heat_matrix = self._compute_heat(adj)
        
        # Convert to torch tensor
        heat_tensor = torch.from_numpy(heat_matrix).float().to(device)
        
        # Sparsify: keep edges above threshold
        mask = heat_tensor > self.threshold
        edge_index_diff = mask.nonzero(as_tuple=False).t().contiguous()
        edge_weight_diff = heat_tensor[mask]
        
        return edge_index_diff, edge_weight_diff

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"t={self.t}, "
            f"threshold={self.threshold}, "
            f"self_loop={self.self_loop})"
        )
