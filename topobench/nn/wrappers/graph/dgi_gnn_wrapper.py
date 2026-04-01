"""Wrapper for DGI (Deep Graph Infomax) pre-training with GNN models."""

import torch
import torch.nn as nn

from topobench.nn.wrappers.base import AbstractWrapper


class DGIGNNWrapper(AbstractWrapper):
    r"""Wrapper for Deep Graph Infomax (DGI) pre-training with GNN models.

    DGI maximizes mutual information between patch representations (node embeddings)
    and graph-level summary vectors. This is achieved by:
    1. Encoding the original graph to get node embeddings (positive samples)
    2. Creating negative samples (corrupted version or different graphs)
    3. Encoding the negative samples
    4. Using a discriminator to distinguish between positive and negative node-summary pairs

    Two corruption strategies are supported:
    - "feature_shuffle": Shuffles node features within each graph
      * Works for both transductive (single graph) and inductive (multiple graphs)
      * For transductive: the only corruption option
      * For inductive: one of two options

    - "graph_diffusion": Uses node embeddings from different graphs as negatives
      * Only for inductive setting (requires batch_size >= 2)
      * As described in DGI paper: "our corruption function simply samples
        a different graph from the training set"
      * Each graph is encoded, then paired with a different graph's node embeddings

    Parameters
    ----------
    backbone : torch.nn.Module
        The encoder backbone model (GNN).
    corruption_type : str, optional
        Type of corruption to apply (default: "feature_shuffle").
        - "feature_shuffle": Randomly permute node features (transductive & inductive)
        - "graph_diffusion": Use different graphs as negatives (inductive only)
    **kwargs : dict
        Additional arguments for the AbstractWrapper base class.

    Notes
    -----
    For "graph_diffusion", the batch must contain at least 2 graphs (batch_size >= 2).
    """

    def __init__(
        self,
        backbone: nn.Module,
        corruption_type: str = "feature_shuffle",
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(backbone, **kwargs)

        self.corruption_type = corruption_type
        self.verbose = verbose
        self.forward_count = 0  # Track number of forward passes

        if corruption_type not in ["feature_shuffle", "graph_diffusion"]:
            raise ValueError(
                f"corruption_type must be 'feature_shuffle' or 'graph_diffusion', got {corruption_type}"
            )

    def corrupt_graph(self, x, batch_indices):
        """Corrupt node features to create negative samples.

        For "feature_shuffle": Shuffle features within each graph (transductive).
        This is used in the forward pass to create corrupted input features.

        Parameters
        ----------
        x : torch.Tensor
            Node features of shape (num_nodes, num_features).
        batch_indices : torch.Tensor
            Batch assignment for each node.

        Returns
        -------
        torch.Tensor
            Corrupted node features with same shape as input.
        """
        if self.corruption_type != "feature_shuffle":
            raise ValueError(
                "corrupt_graph should only be called with feature_shuffle. "
                "graph_diffusion uses different graphs directly."
            )

        # Shuffle features within each graph separately (transductive)
        batch_ids = batch_indices.unique()
        corrupted_x = x.clone()

        for batch_id in batch_ids:
            # Get nodes for this graph
            graph_mask = batch_indices == batch_id
            graph_indices = torch.where(graph_mask)[0]

            # Shuffle node indices within this graph
            perm = torch.randperm(len(graph_indices), device=x.device)
            shuffled_indices = graph_indices[perm]

            # Replace features with shuffled features
            corrupted_x[graph_indices] = x[shuffled_indices]

        return corrupted_x

    def forward(self, batch):
        r"""Forward pass for DGI encoding.

        Two corruption strategies:
        - "feature_shuffle": Corrupts features within each graph, encodes corrupted version
        - "graph_diffusion": Uses different graphs as negatives (for inductive learning)

        For downstream tasks, use the standard GNNWrapper instead by loading
        only the backbone encoder without this wrapper.

        Parameters
        ----------
        batch : torch_geometric.data.Data
            Batch object containing the batched data.

        Returns
        -------
        dict
            Dictionary containing:
            - x_0: Node embeddings from positive (original) graph
            - x_0_corrupted: Node embeddings from negative samples
            - batch_0: Batch indices for positive samples
            - batch_0_corrupted: Batch indices for negative samples (for graph_diffusion)
            - edge_index: Edge indices (for readout)
        """
        # Input features AFTER feature encoding
        x_0 = batch.x_0
        edge_index = batch.edge_index
        edge_weight = batch.get("edge_weight", None)
        batch_indices = batch.batch_0

        if self.corruption_type == "feature_shuffle":
            # Transductive: corrupt features, keep same graph structure
            # Encode original graph (positive samples)
            h_positive = self.backbone(
                x_0,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )

            # Create corrupted graph (negative samples)
            x_corrupted = self.corrupt_graph(x_0, batch_indices)

            # Encode corrupted graph (negative samples)
            h_negative = self.backbone(
                x_corrupted,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )

            # Return both positive and negative embeddings for discriminator
            model_out = {
                "x_0": h_positive,  # Positive node embeddings
                "x_0_corrupted": h_negative,  # Negative node embeddings
                "batch_0": batch_indices,
                "batch_0_corrupted": batch_indices,  # Same batch structure
                "edge_index": edge_index,
                "labels": batch.y if hasattr(batch, "y") else None,
            }

        elif self.corruption_type == "graph_diffusion":
            # Inductive: use different graphs as negatives
            batch_ids = batch_indices.unique()
            num_graphs = len(batch_ids)

            if num_graphs < 2:
                raise ValueError(
                    "graph_diffusion requires at least 2 graphs in the batch, "
                    f"but got {num_graphs}. Increase batch size or use feature_shuffle."
                )

            # Logging for verification
            self.forward_count += 1
            if (
                self.verbose or self.forward_count <= 3
            ):  # Always log first 3 batches
                print(f"\n{'=' * 80}")
                print(
                    f"DGI graph_diffusion - Forward pass #{self.forward_count}"
                )
                print(f"Mode: {'TRAINING' if self.training else 'VALIDATION'}")
                print(
                    f"Batch contains {num_graphs} graphs with IDs: {batch_ids.tolist()}"
                )
                # Count nodes per graph
                for gid in batch_ids:
                    n_nodes = (batch_indices == gid).sum().item()
                    print(f"  Graph {gid}: {n_nodes} nodes")

            # Encode ALL graphs in the batch
            h_all = self.backbone(
                x_0,
                edge_index,
                batch=batch_indices,
                edge_weight=edge_weight,
            )

            # For each graph, select a different graph as negative
            batch_ids_list = batch_ids.tolist()
            positive_embeddings = []
            negative_embeddings = []
            positive_batch_indices = []
            negative_batch_indices = []

            # Track pairings for logging
            pairings = []

            for graph_id in batch_ids_list:
                # Get nodes for this graph (positive samples)
                pos_mask = batch_indices == graph_id
                pos_indices = torch.where(pos_mask)[0]

                # Select a different graph as negative
                other_graphs = [g for g in batch_ids_list if g != graph_id]
                neg_graph_id = other_graphs[
                    torch.randint(0, len(other_graphs), (1,)).item()
                ]

                pairings.append((graph_id, neg_graph_id))

                # Get nodes for negative graph
                neg_mask = batch_indices == neg_graph_id
                neg_indices = torch.where(neg_mask)[0]

                # Store embeddings
                positive_embeddings.append(h_all[pos_indices])
                negative_embeddings.append(h_all[neg_indices])

                # Create batch indices (renumber to be consecutive)
                new_graph_idx = len(positive_batch_indices)
                positive_batch_indices.append(
                    torch.full(
                        (len(pos_indices),),
                        new_graph_idx,
                        dtype=batch_indices.dtype,
                        device=batch_indices.device,
                    )
                )
                negative_batch_indices.append(
                    torch.full(
                        (len(neg_indices),),
                        new_graph_idx,
                        dtype=batch_indices.dtype,
                        device=batch_indices.device,
                    )
                )

            # Log pairings
            if self.verbose or self.forward_count <= 3:
                print("\nGraph pairings (positive -> negative):")
                for pos_id, neg_id in pairings:
                    print(
                        f"  Graph {pos_id} paired with Graph {neg_id} (as negative)"
                    )
                print(f"{'=' * 80}\n")

            # Concatenate all positive and negative samples
            h_positive = torch.cat(positive_embeddings, dim=0)
            h_negative = torch.cat(negative_embeddings, dim=0)
            batch_positive = torch.cat(positive_batch_indices, dim=0)
            batch_negative = torch.cat(negative_batch_indices, dim=0)

            model_out = {
                "x_0": h_positive,  # Positive node embeddings (all graphs)
                "x_0_corrupted": h_negative,  # Negative node embeddings (from different graphs)
                "batch_0": batch_positive,  # Batch indices for positive samples
                "batch_0_corrupted": batch_negative,  # Batch indices for negative samples
                "edge_index": edge_index,  # Original edge index (may not be used)
                "labels": batch.y if hasattr(batch, "y") else None,
            }

        else:
            raise ValueError(
                f"Unknown corruption_type: {self.corruption_type}"
            )

        return model_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"corruption_type={self.corruption_type})"
        )
