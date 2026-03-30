"""
Analyze AQSOL, NCI1, and IMDB-MULTI with the same pipeline as analyze_datasets.ipynb:
load → sample inspection → graph statistics → plots → node communities → community stats.

IMDB-MULTI: sweeps K ∈ {2,…,11} (10 values), mean silhouette per K, saves a curve plot,
then prompts for K (or ``--imdb-k``) before community statistics. A final
GraphUniverse-style report is appended to ``graphuniverse_equivalence_report.txt``.
"""

from __future__ import annotations

import argparse
import os
import sys
import shutil
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter
from torch_geometric.datasets import AQSOL, TUDataset

# sklearn KMeans can warn on tiny graphs in IMDB-MULTI
warnings.filterwarnings(
    "ignore",
    message="Number of distinct clusters .* found smaller than n_clusters",
    category=UserWarning,
    module="sklearn",
)

DATASETS = ("AQSOL", "NCI1", "IMDB-MULTI")


def load_aqsol_dataset(root: str = "./data", force_reload: bool = False):
    print("Loading AQSOL dataset...")
    aqsol_root = os.path.join(root, "AQSOL")
    if force_reload and os.path.exists(aqsol_root):
        processed_dir = os.path.join(aqsol_root, "processed")
        if os.path.exists(processed_dir):
            shutil.rmtree(processed_dir)
    try:
        train_dataset = AQSOL(root=aqsol_root, split="train", force_reload=force_reload)
        val_dataset = AQSOL(root=aqsol_root, split="val", force_reload=force_reload)
        test_dataset = AQSOL(root=aqsol_root, split="test", force_reload=force_reload)
    except Exception as e:
        print(f"Error loading AQSOL: {e}\nAttempting force reload...")
        if os.path.exists(aqsol_root):
            shutil.rmtree(aqsol_root)
        train_dataset = AQSOL(root=aqsol_root, split="train", force_reload=True)
        val_dataset = AQSOL(root=aqsol_root, split="val", force_reload=True)
        test_dataset = AQSOL(root=aqsol_root, split="test", force_reload=True)

    dataset = train_dataset + val_dataset + test_dataset
    print("Dataset loaded successfully!")
    print(
        f"Number of graphs: {len(dataset)} (train: {len(train_dataset)}, "
        f"val: {len(val_dataset)}, test: {len(test_dataset)})"
    )
    nc = getattr(train_dataset, "num_classes", None)
    print(f"Number of classes/tasks: {nc if nc is not None else 'N/A (regression)'}")
    print(f"Number of node features: {train_dataset.num_node_features}")
    print(f"Number of edge features: {train_dataset.num_edge_features}\n")
    return dataset


def load_nci1_dataset(root: str = "./data"):
    print("Loading NCI1 dataset...")
    dataset = TUDataset(root=root, name="NCI1")
    print("Dataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes/tasks: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}\n")
    return dataset


def load_imdbmulti_dataset(root: str = "./data"):
    print("Loading IMDB-MULTI dataset...")
    dataset = TUDataset(root=root, name="IMDB-MULTI", use_node_attr=True, use_edge_attr=True)
    print("Dataset loaded successfully!")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes/tasks: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")
    print(f"Number of edge features: {dataset.num_edge_features}\n")
    return dataset


def load_dataset(name: str, root: str):
    if name == "AQSOL":
        return load_aqsol_dataset(root=root)
    if name == "NCI1":
        return load_nci1_dataset(root=root)
    if name == "IMDB-MULTI":
        return load_imdbmulti_dataset(root=root)
    raise ValueError(f"Unknown dataset: {name}")


def inspect_sample_data(dataset, num_samples: int = 3):
    print("=" * 80)
    print("INSPECTING SAMPLE DATA")
    print("=" * 80)
    for i in range(min(num_samples, len(dataset))):
        data = dataset[i]
        print(f"\n--- Sample {i + 1} ---")
        print(f"Graph object: {data}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        if hasattr(data, "x") and data.x is not None:
            print(f"\nNode features shape: {data.x.shape}")
            print(f"First 3 nodes features:\n{data.x[:3]}")
        else:
            print("\nNode features: None")
        print(f"\nTarget (y) shape: {data.y.shape}")
        print(f"Target value: {data.y}")
        print(f"\nEdge index shape: {data.edge_index.shape}")
        print(f"First 5 edges:\n{data.edge_index[:, :5]}")
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            print(f"\nEdge features shape: {data.edge_attr.shape}")
            print(f"First 3 edge features:\n{data.edge_attr[:3]}")
        else:
            print("\nEdge features: None")
        print("-" * 80)


def calculate_graph_statistics(dataset):
    print("\n" + "=" * 80)
    print("CALCULATING GRAPH STATISTICS")
    print("=" * 80)
    num_nodes_list = []
    num_edges_list = []
    node_degrees_list = []
    avg_degrees_per_graph = []
    target_values = []
    print("Processing all graphs...")
    for i, data in enumerate(dataset):
        num_nodes = data.num_nodes
        num_edges = data.num_edges
        num_nodes_list.append(num_nodes)
        num_edges_list.append(num_edges)
        target_values.append(data.y.item() if data.y.numel() == 1 else data.y.numpy())
        edge_index = data.edge_index
        degrees = torch.bincount(edge_index[0], minlength=num_nodes)
        node_degrees_list.extend(degrees.tolist())
        avg_degree = degrees.float().mean().item() if num_nodes > 0 else 0
        avg_degrees_per_graph.append(avg_degree)
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset)} graphs...")
    print(f"Processed all {len(dataset)} graphs!\n")
    stats = {
        "num_nodes": num_nodes_list,
        "num_edges": num_edges_list,
        "node_degrees": node_degrees_list,
        "avg_degrees_per_graph": avg_degrees_per_graph,
        "targets": target_values,
    }
    stats["num_nodes_5p"] = np.percentile(num_nodes_list, 5)
    stats["num_nodes_95p"] = np.percentile(num_nodes_list, 95)
    stats["avg_deg_5p"] = np.percentile(avg_degrees_per_graph, 5)
    stats["avg_deg_95p"] = np.percentile(avg_degrees_per_graph, 95)
    return stats


def print_statistics(stats):
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print("\n--- Graph Sizes (Number of Nodes) ---")
    print(f"Mean: {np.mean(stats['num_nodes']):.2f}")
    print(f"Std: {np.std(stats['num_nodes']):.2f}")
    print(f"Min: {np.min(stats['num_nodes'])}")
    print(f"Max: {np.max(stats['num_nodes'])}")
    print(f"Median: {np.median(stats['num_nodes']):.2f}")
    print(f"25th percentile: {np.percentile(stats['num_nodes'], 25):.2f}")
    print(f"75th percentile: {np.percentile(stats['num_nodes'], 75):.2f}")
    print(
        f"90% coverage (from 5th to 95th percentile): "
        f"{stats['num_nodes_5p']:.2f} to {stats['num_nodes_95p']:.2f}"
    )
    print("\n--- Number of Edges per Graph ---")
    print(f"Mean: {np.mean(stats['num_edges']):.2f}")
    print(f"Std: {np.std(stats['num_edges']):.2f}")
    print(f"Min: {np.min(stats['num_edges'])}")
    print(f"Max: {np.max(stats['num_edges'])}")
    print(f"Median: {np.median(stats['num_edges']):.2f}")
    print("\n--- Node Degrees (across all nodes in all graphs) ---")
    print(f"Mean: {np.mean(stats['node_degrees']):.2f}")
    print(f"Std: {np.std(stats['node_degrees']):.2f}")
    print(f"Min: {np.min(stats['node_degrees'])}")
    print(f"Max: {np.max(stats['node_degrees'])}")
    print(f"Median: {np.median(stats['node_degrees']):.2f}")
    print("\n--- Average Degree Per Graph ---")
    print(f"Mean: {np.mean(stats['avg_degrees_per_graph']):.2f}")
    print(f"Std: {np.std(stats['avg_degrees_per_graph']):.2f}")
    print(f"Min: {np.min(stats['avg_degrees_per_graph']):.2f}")
    print(f"Max: {np.max(stats['avg_degrees_per_graph']):.2f}")
    print(f"Median: {np.median(stats['avg_degrees_per_graph']):.2f}")
    print(
        f"90% coverage (from 5th to 95th percentile): "
        f"{stats['avg_deg_5p']:.2f} to {stats['avg_deg_95p']:.2f}"
    )
    degree_counts = Counter(stats["node_degrees"])
    print("\nDegree distribution (degree: count):")
    for degree in sorted(degree_counts.keys())[:10]:
        print(f"  Degree {degree}: {degree_counts[degree]} nodes")
    if len(degree_counts) > 10:
        print(f"  ... ({len(degree_counts)} unique degree values total)")
    print("\n--- Target Values (y) ---")
    targets_array = np.array(stats["targets"])
    print(f"Mean: {np.mean(targets_array):.4f}")
    print(f"Std: {np.std(targets_array):.4f}")
    print(f"Min: {np.min(targets_array):.4f}")
    print(f"Max: {np.max(targets_array):.4f}")
    print(f"Median: {np.median(targets_array):.4f}")


def plot_distributions(stats, save_path: str, target_xlabel: str = "Target value (y)"):
    print("\n" + "=" * 80)
    print("CREATING DISTRIBUTION PLOTS")
    print("=" * 80)
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    ax = axes[0, 0]
    ax.hist(stats["num_nodes"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Graph Sizes (Number of Nodes)")
    ax.grid(True, alpha=0.3)
    ax.axvline(stats["num_nodes_5p"], color="red", linestyle="--", label="5th percentile")
    ax.axvline(stats["num_nodes_95p"], color="blue", linestyle="--", label="95th percentile")
    ax.text(
        0.01,
        0.97,
        f"90% covers [{stats['num_nodes_5p']:.2f}, {stats['num_nodes_95p']:.2f}]",
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
    )
    ax.legend()
    ax = axes[0, 1]
    ax.hist(stats["num_edges"], bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Number of Edges")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Number of Edges")
    ax.grid(True, alpha=0.3)
    ax = axes[0, 2]
    nd = stats["node_degrees"]
    hi = min(max(nd) + 2, 20)
    ax.hist(nd, bins=range(min(nd), hi), edgecolor="black", alpha=0.7, color="green")
    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Node Degrees")
    ax.grid(True, alpha=0.3)
    ax = axes[0, 3]
    ax.boxplot(stats["num_nodes"], vert=True)
    ax.set_ylabel("Number of Nodes")
    ax.set_title("Box Plot of Graph Sizes")
    ax.grid(True, alpha=0.3)
    xlims = ax.get_xlim()
    ax.plot([xlims[0], xlims[1]], [stats["num_nodes_5p"], stats["num_nodes_5p"]], color="red", linestyle="--")
    ax.plot([xlims[0], xlims[1]], [stats["num_nodes_95p"], stats["num_nodes_95p"]], color="blue", linestyle="--")
    ax.text(
        0.98,
        0.02,
        f"90% [{stats['num_nodes_5p']:.2f}, {stats['num_nodes_95p']:.2f}]",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6),
    )
    ax = axes[1, 1]
    ax.hist(stats["targets"], bins=50, edgecolor="black", alpha=0.7, color="red")
    ax.set_xlabel(target_xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Target Values")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 2]
    ax.scatter(stats["num_nodes"], stats["num_edges"], alpha=0.3, s=10)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Relationship between Nodes and Edges")
    ax.grid(True, alpha=0.3)
    ax = axes[1, 3]
    ax.hist(stats["avg_degrees_per_graph"], bins=40, color="slateblue", alpha=0.75, edgecolor="black")
    ax.axvline(stats["avg_deg_5p"], color="red", linestyle="--", lw=1)
    ax.axvline(stats["avg_deg_95p"], color="blue", linestyle="--", lw=1)
    ax.set_title("Avg Degree")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.5,
        0.94,
        f"90%\n[{stats['avg_deg_5p']:.2f}, {stats['avg_deg_95p']:.2f}]",
        ha="center",
        va="top",
        fontsize=8,
        transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7),
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {save_path}")
    plt.close(fig)


def extract_node_communities_nci(nci_dataset):
    annotated = []
    for graph in nci_dataset:
        community_labels = graph.x.argmax(dim=1)
        graph["community_labels"] = community_labels
        annotated.append(graph)
    return annotated


def _imdb_node_features(graph):
    """Degree + local clustering coefficient per node (same as original IMDB pipeline)."""
    import networkx as nx
    from torch_geometric.utils import degree, to_networkx

    num_nodes = graph.num_nodes
    if num_nodes == 0:
        return None, 0
    deg = degree(graph.edge_index[0], num_nodes=num_nodes, dtype=torch.float)
    G = to_networkx(graph, to_undirected=True)
    clustering_coeffs = nx.clustering(G)
    clustering = torch.tensor([clustering_coeffs[i] for i in range(num_nodes)], dtype=torch.float)
    features = torch.stack([deg, clustering], dim=1).numpy()
    return features, num_nodes


def _imdb_cluster_labels_for_graph(graph, n_clusters: int, random_state: int = 42):
    """
    Node cluster ids for one graph. Mirrors previous logic: KMeans when possible,
    else degree binning / zeros.
    """
    from sklearn.cluster import KMeans

    features, num_nodes = _imdb_node_features(graph)
    if features is None:
        return np.zeros(0, dtype=np.int64)
    degrees = torch.tensor(features[:, 0], dtype=torch.float)
    if num_nodes >= n_clusters:
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        return kmeans.fit_predict(features)
    if degrees.max() > degrees.min():
        num_bins = min(n_clusters, num_nodes)
        if num_bins > 1:
            degree_bins = torch.quantile(
                degrees.float(), torch.linspace(0, 1, num_bins + 1)[1:-1]
            )
            return torch.bucketize(degrees, degree_bins).numpy()
        return np.zeros(num_nodes, dtype=np.int64)
    return np.zeros(num_nodes, dtype=np.int64)


def _mean_silhouette_imdb_k(dataset, n_clusters: int, random_state: int = 42) -> tuple[float, int]:
    """Mean silhouette over graphs where the score is defined."""
    from sklearn.metrics import silhouette_score

    scores: list[float] = []
    used = 0
    for graph in dataset:
        features, num_nodes = _imdb_node_features(graph)
        if features is None or num_nodes < 2:
            continue
        labels = _imdb_cluster_labels_for_graph(graph, n_clusters, random_state=random_state)
        if len(labels) < 2:
            continue
        n_unique = len(np.unique(labels))
        if n_unique < 2 or n_unique >= num_nodes:
            continue
        try:
            s = float(silhouette_score(features, labels, metric="euclidean"))
            scores.append(s)
            used += 1
        except ValueError:
            continue
    if not scores:
        return float("nan"), 0
    return float(np.mean(scores)), used


def imdb_silhouette_sweep(
    dataset, k_values: tuple[int, ...] = tuple(range(2, 12))
) -> tuple[list[int], list[float], list[int]]:
    """Ten K values (default 2..11): mean silhouette and #graphs used per K."""
    ks: list[int] = []
    means: list[float] = []
    counts: list[int] = []
    for k in k_values:
        m, n = _mean_silhouette_imdb_k(dataset, k)
        ks.append(k)
        means.append(m)
        counts.append(n)
    return ks, means, counts


def plot_imdb_silhouette_vs_k(
    k_vals: list[int],
    sil_means: list[float],
    save_path: str,
    suggested_k: int | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    valid = [(k, s) for k, s in zip(k_vals, sil_means) if np.isfinite(s)]
    if valid:
        k_plot, s_plot = zip(*valid)
        ax.plot(k_plot, s_plot, "o-", lw=2, ms=8, color="steelblue")
    ax.set_xlabel("K (number of clusters)")
    ax.set_ylabel("Mean silhouette score (per graph, then averaged)")
    ax.set_title("IMDB-MULTI: node clustering — silhouette vs K")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_vals)
    if suggested_k is not None and suggested_k in k_vals:
        ax.axvline(suggested_k, color="gray", linestyle="--", alpha=0.8, label=f"max silhouette K={suggested_k}")
        ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"IMDB silhouette sweep plot saved to: {save_path}")


def _suggest_k_max_silhouette(k_vals: list[int], sil_means: list[float]) -> int | None:
    pairs = [(k, s) for k, s in zip(k_vals, sil_means) if np.isfinite(s)]
    if not pairs:
        return None
    return max(pairs, key=lambda t: t[1])[0]


def prompt_imdb_k(
    k_vals: list[int],
    sil_means: list[float],
    counts: list[int],
    preset: int | None,
    non_interactive: bool,
) -> int:
    if preset is not None:
        if preset not in k_vals:
            raise SystemExit(
                f"--imdb-k={preset} is not in sweep K set {list(k_vals)}. "
                "Use one of these values."
            )
        print(f"Using IMDB node-clustering K from CLI: {preset}")
        return preset
    if non_interactive:
        sug = _suggest_k_max_silhouette(k_vals, sil_means)
        if sug is None:
            raise SystemExit(
                "Non-interactive run: could not infer K (no valid silhouette). "
                "Pass --imdb-k explicitly."
            )
        print(f"Non-interactive: using K with highest mean silhouette: {sug}")
        return sug

    sug = _suggest_k_max_silhouette(k_vals, sil_means)
    print("\n" + "=" * 80)
    print("IMDB-MULTI: choose K for node clustering (see silhouette plot for 'knee')")
    print("=" * 80)
    print(f"{'K':>4}  {'mean silhouette':>18}  {'graphs used':>12}")
    for k, s, n in zip(k_vals, sil_means, counts):
        ss = f"{s:.4f}" if np.isfinite(s) else "n/a"
        print(f"{k:>4}  {ss:>18}  {n:>12}")
    if sug is not None:
        print(f"\nHeuristic maximum mean silhouette at K = {sug} (you may still pick a knee).")
    while True:
        raw = input(f"Enter K in {list(k_vals)} (default {sug}): ").strip()
        if not raw:
            if sug is None:
                print("No default available; type an integer K.")
                continue
            return sug
        try:
            k = int(raw)
        except ValueError:
            print("Invalid integer.")
            continue
        if k not in k_vals:
            print(f"K must be one of {list(k_vals)}.")
            continue
        return k


def extract_node_communities_imdb(imdb_dataset, n_clusters: int):
    annotated_dataset = []
    for graph in imdb_dataset:
        num_nodes = graph.num_nodes
        if num_nodes == 0:
            graph["community_labels"] = torch.tensor([], dtype=torch.long)
            annotated_dataset.append(graph)
            continue
        community_labels = _imdb_cluster_labels_for_graph(graph, n_clusters)
        graph["community_labels"] = torch.tensor(community_labels, dtype=torch.long)
        annotated_dataset.append(graph)
    return annotated_dataset


def extract_node_communities_aqsol(aqsol_dataset):
    annotated_dataset = []
    for graph in aqsol_dataset:
        graph["community_labels"] = graph.x
        annotated_dataset.append(graph)
    return annotated_dataset


def _community_rows(community_labels: torch.Tensor):
    """Hashable per-node community ids (1D class index or multidimensional features)."""
    if community_labels.dim() == 1:
        return community_labels.tolist()
    return [tuple(r.tolist()) for r in community_labels]


def _same_community(community_labels: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    if community_labels.dim() == 1:
        return community_labels[src] == community_labels[dst]
    return (community_labels[src] == community_labels[dst]).all(dim=-1)


def annotate_communities(dataset_name: str, dataset, imdb_k: int | None = None):
    if dataset_name == "AQSOL":
        return extract_node_communities_aqsol(dataset)
    if dataset_name == "NCI1":
        return extract_node_communities_nci(dataset)
    if dataset_name == "IMDB-MULTI":
        if imdb_k is None:
            raise ValueError("imdb_k is required for IMDB-MULTI (run silhouette sweep first).")
        return extract_node_communities_imdb(dataset, n_clusters=imdb_k)
    raise ValueError(dataset_name)


def calculate_node_community_statistics(community_annotated_dataset):
    unique_communities = set()
    for graph in community_annotated_dataset:
        cl = graph["community_labels"]
        unique_communities.update(_community_rows(cl))
    num_unique_communities = len(unique_communities)
    print(f"(showing up to 20) {list(unique_communities)[:20]}...")
    print(f"Number of unique communities: {num_unique_communities}")
    num_unique_comms_per_graph = [
        len(set(_community_rows(graph["community_labels"]))) for graph in community_annotated_dataset
    ]
    distribution_of_unique_communities = Counter(num_unique_comms_per_graph)
    unique_comms_5th = np.percentile(num_unique_comms_per_graph, 5)
    unique_comms_95th = np.percentile(num_unique_comms_per_graph, 95)
    print(f"5th percentile of # unique communities per graph: {unique_comms_5th}")
    print(f"95th percentile of # unique communities per graph: {unique_comms_95th}")
    distribution_of_community_size = {}
    total_nodes = 0
    for graph in community_annotated_dataset:
        community_labels = graph["community_labels"]
        total_nodes += len(community_labels)
        for label in _community_rows(community_labels):
            distribution_of_community_size[label] = distribution_of_community_size.get(label, 0) + 1
    sorted_communities_and_sizes = sorted(
        distribution_of_community_size.items(), key=lambda x: x[1], reverse=True
    )
    cum_sum = 0
    num_communities_for_90pct = 0
    for _label, size in sorted_communities_and_sizes:
        cum_sum += size
        num_communities_for_90pct += 1
        if cum_sum / total_nodes >= 0.95:
            break
    print(
        f"To cover 95% of all nodes, need top {num_communities_for_90pct} "
        f"communities out of {num_unique_communities}"
    )
    community_homophily = []
    for graph in community_annotated_dataset:
        community_labels = graph["community_labels"].squeeze()
        edge_index = graph.edge_index
        src = edge_index[0]
        dst = edge_index[1]
        same_community = _same_community(community_labels, src, dst)
        if len(same_community) > 0:
            homophily = same_community.float().mean().item()
        else:
            homophily = float("nan")
        community_homophily.append(homophily)
    homophily_array = np.array([h for h in community_homophily if not np.isnan(h)])
    if len(homophily_array) > 0:
        homophily_5th = np.percentile(homophily_array, 5)
        homophily_95th = np.percentile(homophily_array, 95)
    else:
        homophily_5th = float("nan")
        homophily_95th = float("nan")
    print(f"5th percentile of homophily: {homophily_5th}")
    print(f"95th percentile of homophily: {homophily_95th}")
    return {
        "num_unique_communities": num_unique_communities,
        "distribution_of_unique_communities": distribution_of_unique_communities,
        "num_unique_comms_per_graph": num_unique_comms_per_graph,
        "unique_comms_5th": unique_comms_5th,
        "unique_comms_95th": unique_comms_95th,
        "distribution_of_community_homophily": community_homophily,
        "homophily_5th": homophily_5th,
        "homophily_95th": homophily_95th,
        "distribution_of_community_size": distribution_of_community_size,
        "sorted_communities_and_sizes": sorted_communities_and_sizes,
        "num_communities_for_90pct": num_communities_for_90pct,
        "total_nodes": total_nodes,
    }


def plot_community_statistics(stats, save_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    bins = sorted(stats["distribution_of_unique_communities"].keys())
    axes[0].bar(bins, [stats["distribution_of_unique_communities"][k] for k in bins])
    axes[0].set_xlabel("Number of Unique Communities")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Unique Communities")
    axes[0].axvline(stats["unique_comms_5th"], color="red", linestyle="--", label="5th pct")
    axes[0].axvline(stats["unique_comms_95th"], color="red", linestyle="-", label="95th pct")
    axes[0].legend()
    homophily_no_nan = [h for h in stats["distribution_of_community_homophily"] if not np.isnan(h)]
    axes[1].hist(homophily_no_nan, bins=20, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Community Homophily")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Community Homophily")
    axes[1].axvline(stats["homophily_5th"], color="red", linestyle="--", label="5th pct")
    axes[1].axvline(stats["homophily_95th"], color="red", linestyle="-", label="95th pct")
    axes[1].legend()
    sorted_communities_and_sizes = stats["sorted_communities_and_sizes"]
    sizes_sorted = [v for _k, v in sorted_communities_and_sizes]
    axes[2].bar(range(len(sizes_sorted)), sizes_sorted)
    axes[2].set_xlabel("Community (sorted)")
    axes[2].set_ylabel("Node Count")
    axes[2].set_title("Distribution of Community Size (sorted)")
    num_90 = stats["num_communities_for_90pct"]
    axes[2].axvline(num_90 - 0.5, color="green", linestyle="--", label=f"Top {num_90} (~95% nodes)")
    axes[2].legend()
    ax2b = axes[2].twinx()
    cum_sizes = np.cumsum(sizes_sorted)
    ax2b.plot(range(len(sizes_sorted)), cum_sizes / stats["total_nodes"], color="orange")
    ax2b.set_ylabel("Cumulative Fraction of Nodes")
    ax2b.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Community plot saved to: {save_path}")
    plt.close(fig)


def target_xlabel_for(dataset_name: str) -> str:
    if dataset_name == "AQSOL":
        return "Target (solubility)"
    if dataset_name == "NCI1":
        return "Graph class label"
    return "Graph class label"


def graphuniverse_graph_level_k_description(dataset, dataset_name: str) -> str:
    if dataset_name == "AQSOL":
        return "N/A (regression — molecular solubility; no discrete graph-class K)"
    nc = getattr(dataset, "num_classes", None)
    if nc is not None and nc > 0:
        return str(int(nc))
    return "unknown"


def append_graphuniverse_report_section(
    report_path: str,
    dataset_name: str,
    dataset,
    stats: dict,
    community_stats: dict,
    imdb_chosen_k: int | None,
) -> None:
    """Append one dataset block; includes assumed feature/structural signal labels."""
    n_graphs = len(dataset)
    k_desc = graphuniverse_graph_level_k_description(dataset, dataset_name)

    if dataset_name == "IMDB-MULTI":
        feature_signal = "Low"
        structural_signal = "High"
        assumption_paragraph = (
            "For IMDB-MULTI we assume **High** structural signal and **Low** feature signal "
            "in GraphUniverse terms: the benchmark has no natural node attributes; node "
            "information used for clustering is derived from topology (degree + local "
            "clustering coefficient)."
        )
    else:
        feature_signal = "Medium"
        structural_signal = "Medium"
        assumption_paragraph = (
            "For this dataset we assume **Medium** feature signal and **Medium** structural "
            "signal for GraphUniverse-style alignment (explicit assumption, not estimated "
            "by this script)."
        )

    h5 = community_stats["homophily_5th"]
    h95 = community_stats["homophily_95th"]
    h5s = f"{h5:.4f}" if np.isfinite(h5) else "n/a"
    h95s = f"{h95:.4f}" if np.isfinite(h95) else "n/a"

    lines = [
        "",
        "=" * 72,
        f" GraphUniverse-style equivalence — {dataset_name}",
        "=" * 72,
        "",
        f"n_graphs:                    {n_graphs}",
        f"Graph-level K (classes):    {k_desc}",
    ]
    if dataset_name == "IMDB-MULTI" and imdb_chosen_k is not None:
        lines.append(
            f"Node clustering K (IMDB):   {imdb_chosen_k}  "
            "(KMeans on degree + local clustering; chosen after silhouette sweep)"
        )
    lines.extend(
        [
            "",
            "Percentiles (5th — 95th) for empirical summaries:",
            f"  Homophily (community edges):     {h5s} — {h95s}",
            f"  Avg degree per graph:            {stats['avg_deg_5p']:.4f} — {stats['avg_deg_95p']:.4f}",
            f"  Graph size (nodes):              {stats['num_nodes_5p']:.2f} — {stats['num_nodes_95p']:.2f}",
            f"  Unique communities per graph:    {community_stats['unique_comms_5th']:.2f} — "
            f"{community_stats['unique_comms_95th']:.2f}",
            "",
            "Assumed GraphUniverse signal levels (for reporting):",
            f"  Feature signal:    {feature_signal}",
            f"  Structural signal: {structural_signal}",
            "",
            assumption_paragraph,
            "",
        ]
    )
    block = "\n".join(lines)
    with open(report_path, "a", encoding="utf-8") as f:
        f.write(block)
    print(block)


def run_one(
    dataset_name: str,
    root: str,
    out_dir: str,
    imdb_k: int | None = None,
    no_prompt: bool = False,
    report_path: str | None = None,
):
    os.makedirs(out_dir, exist_ok=True)
    safe = dataset_name.replace("-", "_")
    print("\n" + "#" * 80)
    print(f"# {dataset_name}")
    print("#" * 80 + "\n")
    dataset = load_dataset(dataset_name, root=root)
    inspect_sample_data(dataset, num_samples=3)
    stats = calculate_graph_statistics(dataset)
    print_statistics(stats)
    plot_distributions(
        stats,
        save_path=os.path.join(out_dir, f"{safe}_distributions.png"),
        target_xlabel=target_xlabel_for(dataset_name),
    )

    imdb_chosen_k: int | None = None
    if dataset_name == "IMDB-MULTI":
        k_vals, sil_means, counts = imdb_silhouette_sweep(dataset)
        sug = _suggest_k_max_silhouette(k_vals, sil_means)
        plot_imdb_silhouette_vs_k(
            k_vals,
            sil_means,
            os.path.join(out_dir, f"{safe}_silhouette_vs_k.png"),
            suggested_k=sug,
        )
        imdb_chosen_k = prompt_imdb_k(
            k_vals, sil_means, counts, preset=imdb_k, non_interactive=no_prompt
        )

    community_annotated = annotate_communities(
        dataset_name, dataset, imdb_k=imdb_chosen_k
    )
    print("\nSample graph after community annotation:\n", community_annotated[0])
    community_stats = calculate_node_community_statistics(community_annotated)
    plot_community_statistics(
        community_stats, save_path=os.path.join(out_dir, f"{safe}_communities.png")
    )

    if report_path is not None:
        append_graphuniverse_report_section(
            report_path,
            dataset_name,
            dataset,
            stats,
            community_stats,
            imdb_chosen_k=imdb_chosen_k,
        )


def main():
    p = argparse.ArgumentParser(description="Analyze AQSOL, NCI1, IMDB-MULTI (PyG).")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=list(DATASETS),
        choices=list(DATASETS),
        help="Subset to run (default: all three).",
    )
    p.add_argument("--root", default="./data", help="Data root directory.")
    p.add_argument(
        "--out-dir",
        default="./analysis_output",
        help="Where to save PNG figures.",
    )
    p.add_argument(
        "--imdb-k",
        type=int,
        default=None,
        help="IMDB-MULTI only: node-clustering K (must be in 2..11). Skips interactive prompt.",
    )
    p.add_argument(
        "--no-prompt",
        action="store_true",
        help="IMDB-MULTI only: non-interactive — pick K with highest mean silhouette.",
    )
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "graphuniverse_equivalence_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("GraphUniverse equivalence report\n")
        f.write("Empirical percentiles are 5th–95th unless noted.\n")
        f.write("IMDB node-clustering K: silhouette sweep K∈{2,…,11}, then user or --imdb-k / --no-prompt.\n")

    for name in args.datasets:
        run_one(
            name,
            root=args.root,
            out_dir=args.out_dir,
            imdb_k=args.imdb_k,
            no_prompt=args.no_prompt,
            report_path=report_path,
        )

    print(f"\nCombined report written to: {report_path}")


if __name__ == "__main__":
    main()
