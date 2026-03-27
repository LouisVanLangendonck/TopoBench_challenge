"""VGAE pretraining: edge masking, within-graph negative sampling, variational latent z."""

import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling

from topobench.nn.wrappers.base import AbstractWrapper


class _EdgeSamplingGNNWrapper(AbstractWrapper):
    r"""Mask random edges, run the GNN on the subgraph, sample negatives per graph.

    Internal base for :class:`VAEGNNWrapper`.
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_sample_ratio: float = 0.5,
        neg_sample_ratio: float = 1.0,
        sampling_method: str = "sparse",
        **kwargs,
    ):
        super().__init__(backbone, **kwargs)

        self.edge_sample_ratio = edge_sample_ratio
        self.neg_sample_ratio = neg_sample_ratio
        self.sampling_method = sampling_method

        if not 0.0 < edge_sample_ratio < 1.0:
            raise ValueError(f"edge_sample_ratio must be in (0, 1), got {edge_sample_ratio}")
        if neg_sample_ratio <= 0:
            raise ValueError(f"neg_sample_ratio must be positive, got {neg_sample_ratio}")
        if sampling_method not in ["sparse", "dense"]:
            raise ValueError(f"sampling_method must be 'sparse' or 'dense', got {sampling_method}")

    def sample_edges(self, edge_index, batch_indices, num_nodes, device):
        num_edges = edge_index.size(1)

        if num_edges == 0:
            return (
                torch.empty((2, 0), dtype=edge_index.dtype, device=device),
                torch.empty((2, 0), dtype=edge_index.dtype, device=device),
            )

        num_pos_samples = max(1, int(self.edge_sample_ratio * num_edges))

        perm = torch.randperm(num_edges, device=device)
        pos_indices = perm[:num_pos_samples]
        remain_indices = perm[num_pos_samples:]

        remaining_edge_index = edge_index[:, remain_indices]
        pos_edge_index = edge_index[:, pos_indices]

        return remaining_edge_index, pos_edge_index

    def sample_negative_edges(
        self, remaining_edge_index, num_pos_edges, batch_indices, num_nodes, device
    ):
        num_neg_samples = int(num_pos_edges * self.neg_sample_ratio)

        if num_neg_samples == 0:
            return torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)

        batch_ids = batch_indices.unique()
        neg_edges_list = []

        samples_per_graph = {}
        total_nodes = 0
        for batch_id in batch_ids:
            graph_node_mask = batch_indices == batch_id
            num_graph_nodes = graph_node_mask.sum().item()
            samples_per_graph[batch_id.item()] = num_graph_nodes
            total_nodes += num_graph_nodes

        for batch_id in batch_ids:
            graph_node_mask = batch_indices == batch_id
            graph_nodes = torch.where(graph_node_mask)[0]
            num_graph_nodes = len(graph_nodes)

            if num_graph_nodes < 2:
                continue

            src_in_graph = torch.isin(remaining_edge_index[0], graph_nodes)
            dst_in_graph = torch.isin(remaining_edge_index[1], graph_nodes)
            graph_edge_mask = src_in_graph & dst_in_graph
            graph_edges = remaining_edge_index[:, graph_edge_mask]

            max_possible_negs = num_graph_nodes * (num_graph_nodes - 1) - graph_edges.size(1)

            graph_neg_samples = max(
                1, int(num_neg_samples * samples_per_graph[batch_id.item()] / total_nodes)
            )
            graph_neg_samples = min(graph_neg_samples, max(1, max_possible_negs))

            node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
            node_mapping[graph_nodes] = torch.arange(num_graph_nodes, device=device)

            local_edges = node_mapping[graph_edges]

            if (local_edges < 0).any() or (local_edges >= num_graph_nodes).any():
                src_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                dst_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                global_neg_edges = torch.stack([graph_nodes[src_idx], graph_nodes[dst_idx]], dim=0)
                neg_edges_list.append(global_neg_edges)
                continue

            try:
                local_neg_edges = negative_sampling(
                    edge_index=local_edges,
                    num_nodes=num_graph_nodes,
                    num_neg_samples=graph_neg_samples,
                    method=self.sampling_method,
                )

                if (local_neg_edges >= num_graph_nodes).any() or (local_neg_edges < 0).any():
                    raise ValueError("Invalid local indices from negative_sampling")

                global_neg_edges = torch.stack(
                    [
                        graph_nodes[local_neg_edges[0]],
                        graph_nodes[local_neg_edges[1]],
                    ],
                    dim=0,
                )

                if (global_neg_edges >= num_nodes).any() or (global_neg_edges < 0).any():
                    raise ValueError("Global negative edges out of bounds!")

                neg_edges_list.append(global_neg_edges)

            except Exception:
                src_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                dst_idx = torch.randint(0, num_graph_nodes, (graph_neg_samples,), device=device)
                mask = src_idx != dst_idx
                if mask.sum() > 0:
                    global_neg_edges = torch.stack(
                        [graph_nodes[src_idx[mask]], graph_nodes[dst_idx[mask]]], dim=0
                    )
                    neg_edges_list.append(global_neg_edges)

        if len(neg_edges_list) > 0:
            neg_edge_index = torch.cat(neg_edges_list, dim=1)
        else:
            neg_edge_index = torch.empty((2, 0), dtype=remaining_edge_index.dtype, device=device)

        return neg_edge_index

    def forward(self, batch):
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        device = x_0.device

        remaining_edge_index, pos_edge_index = self.sample_edges(
            edge_index, batch_indices, num_nodes, device
        )

        node_embeddings = self.backbone(
            x_0,
            remaining_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight if edge_weight is not None else None,
        )

        neg_edge_index = self.sample_negative_edges(
            remaining_edge_index,
            pos_edge_index.size(1),
            batch_indices,
            num_nodes,
            device,
        )

        return {
            "x_0": node_embeddings,
            "pos_edge_index": pos_edge_index,
            "neg_edge_index": neg_edge_index,
            "edge_index": edge_index,
            "remaining_edge_index": remaining_edge_index,
            "batch_0": batch_indices,
            "labels": batch.y if hasattr(batch, "y") else None,
        }


class VAEGNNWrapper(_EdgeSamplingGNNWrapper):
    r"""VGAE-style encoder: GNN :math:`\rightarrow` :math:`\mu`, :math:`\log\sigma^2` :math:`\rightarrow` sample :math:`z`.

    Passes ``z`` as ``x_0`` to the readout for inner-product edge logits. Set
    ``residual_connections: false`` in config (latent space :math:`\neq` input features).

    Parameters
    ----------
    latent_dim : int
        Dimension of :math:`z` per node.
    variational : bool
        If True, use reparameterization; if False, :math:`z=\mu` (GAE-style).
    """

    def __init__(
        self,
        backbone: nn.Module,
        edge_sample_ratio: float = 0.5,
        neg_sample_ratio: float = 1.0,
        sampling_method: str = "sparse",
        latent_dim: int = 32,
        variational: bool = True,
        **kwargs,
    ):
        super().__init__(
            backbone,
            edge_sample_ratio,
            neg_sample_ratio,
            sampling_method,
            **kwargs,
        )
        self.latent_dim = latent_dim
        self.variational = variational
        enc_dim = getattr(backbone, "out_channels", None)
        if enc_dim is None:
            enc_dim = getattr(backbone, "hidden_dim", None)
        if enc_dim is None:
            enc_dim = kwargs["out_channels"]
        self._encoder_dim = int(enc_dim)
        self.fc_mu = nn.Linear(self._encoder_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._encoder_dim, latent_dim) if variational else None

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        num_nodes = x_0.size(0)
        device = x_0.device

        remaining_edge_index, pos_edge_index = self.sample_edges(
            edge_index, batch_indices, num_nodes, device
        )

        h = self.backbone(
            x_0,
            remaining_edge_index,
            batch=batch_indices,
            edge_weight=edge_weight if edge_weight is not None else None,
        )

        mu = self.fc_mu(h)
        if self.variational and self.fc_logvar is not None:
            logvar = self.fc_logvar(h)
            z = self.reparameterize(mu, logvar)
        else:
            logvar = None
            z = mu

        neg_edge_index = self.sample_negative_edges(
            remaining_edge_index,
            pos_edge_index.size(1),
            batch_indices,
            num_nodes,
            device,
        )

        return {
            "x_0": z,
            "mu": mu,
            "logvar": logvar,
            "pos_edge_index": pos_edge_index,
            "neg_edge_index": neg_edge_index,
            "edge_index": edge_index,
            "remaining_edge_index": remaining_edge_index,
            "batch_0": batch_indices,
            "labels": batch.y if hasattr(batch, "y") else None,
        }
