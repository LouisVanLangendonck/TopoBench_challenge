"""
Graph Property Computation for Downstream Tasks

This module computes various graph properties that can be used as targets
for property reconstruction tasks. These properties capture different aspects
of graph structure, community organization, and degree distributions.

Properties computed:
- Simple (scalar per graph): homophily, avg_degree, size, gini, diameter
- Complex (vector/matrix per graph): community_presence, edge_prob_matrix

IMPORTANT: The edge_prob_matrix is ALWAYS K×K (indexed 0 to K-1), where K is
the total number of communities in the universe. Rows/columns for communities
that are not present in a specific graph will be all zeros. This ensures
consistent tensor shapes across all graphs for property reconstruction tasks.

Usage:
    from graph_properties import GraphPropertyComputer
    
    # Initialize with K communities (from config)
    computer = GraphPropertyComputer(K=10)
    
    # Compute all properties for a batch of graphs
    properties = computer.compute_all_properties(data_list)
    
    # Or compute individual properties
    homophily = computer.compute_homophily(data)

Testing:
    Run this file directly to test with GraphUniverse-generated graphs:
    python tutorials/graph_properties.py
"""

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_networkx
from typing import Dict, List, Union
from pathlib import Path


class GraphPropertyComputer:
    """
    Computes graph properties for property reconstruction tasks.
    
    Parameters
    ----------
    K : int
        Number of communities in the GraphUniverse dataset (from universe_parameters).
    """
    
    def __init__(self, K: int):
        self.K = K
    
    # =========================================================================
    # Simple Properties (scalar per graph)
    # =========================================================================
    
    def compute_homophily(self, data: Data) -> float:
        """
        Fraction of edges connecting nodes with the same label.
        
        Homophily measures the tendency of nodes to connect with similar nodes.
        Range: [0, 1], where 1 = all edges connect same-label nodes.
        
        Parameters
        ----------
        data : Data
            PyG Data object with edge_index and y (node labels).
        
        Returns
        -------
        float
            Homophily score in [0, 1].
        """
        edge_index = data.edge_index
        y = data.y  # node labels
        
        if edge_index.size(1) == 0:
            return 0.0
        
        same_label = (y[edge_index[0]] == y[edge_index[1]]).sum().item()
        total_edges = edge_index.size(1)
        
        return same_label / total_edges
    
    def compute_avg_degree(self, data: Data) -> float:
        """
        Mean number of neighbors per node.
        
        Parameters
        ----------
        data : Data
            PyG Data object with edge_index and num_nodes.
        
        Returns
        -------
        float
            Average degree.
        """
        num_nodes = data.num_nodes
        num_edges = data.edge_index.size(1)
        
        return num_edges / num_nodes if num_nodes > 0 else 0.0
    
    def compute_size(self, data: Data) -> float:
        """
        Total number of nodes in the graph.
        
        Parameters
        ----------
        data : Data
            PyG Data object.
        
        Returns
        -------
        float
            Number of nodes (as float for consistency).
        """
        return float(data.num_nodes)
    
    def compute_gini(self, data: Data) -> float:
        """
        GINI coefficient of degree distribution.
        
        Measures degree inequality. Range: [0, 1]
        - 0 = perfectly equal degree distribution
        - 1 = maximally unequal (one node has all edges)
        
        Parameters
        ----------
        data : Data
            PyG Data object with edge_index.
        
        Returns
        -------
        float
            GINI coefficient in [0, 1].
        """
        deg = degree(data.edge_index[0], num_nodes=data.num_nodes).cpu().numpy()
        
        if len(deg) == 0 or deg.sum() == 0:
            return 0.0
        
        deg_sorted = np.sort(deg)
        n = len(deg_sorted)
        cumsum = np.cumsum(deg_sorted)
        
        if cumsum[-1] == 0:
            return 0.0
        
        gini = (2 * np.sum((np.arange(1, n+1)) * deg_sorted)) / (n * cumsum[-1]) - (n+1)/n
        return float(gini)
    
    def compute_diameter(self, data: Data) -> float:
        """
        Graph diameter: longest shortest path in the largest connected component.
        
        If graph is disconnected, uses the largest connected component.
        
        Parameters
        ----------
        data : Data
            PyG Data object.
        
        Returns
        -------
        float
            Diameter (longest shortest path length).
        """
        G = to_networkx(data, to_undirected=True)
        
        if len(G) == 0:
            return 0.0
        
        # Use largest connected component if disconnected
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            if len(components) == 0:
                return 0.0
            G = G.subgraph(max(components, key=len))
        
        if len(G) == 0:
            return 0.0
        
        try:
            return float(nx.diameter(G))
        except:
            return 0.0
    
    # =========================================================================
    # Complex Properties (vector/matrix per graph)
    # =========================================================================
    
    def compute_community_presence(self, data: Data) -> torch.Tensor:
        """
        Binary vector indicating which communities are present in the graph.
        
        Returns a K-dimensional binary vector where entry i is 1 if community i
        has at least one node in the graph, 0 otherwise.
        
        Parameters
        ----------
        data : Data
            PyG Data object with y (node labels).
        
        Returns
        -------
        torch.Tensor
            Binary vector of shape [K] with 1s for present communities.
        """
        y = data.y  # node labels [N]
        presence = torch.zeros(self.K, dtype=torch.float32)
        
        unique_communities = torch.unique(y)
        presence[unique_communities.long()] = 1.0
        
        return presence
    
    def compute_edge_prob_matrix(self, data: Data) -> torch.Tensor:
        """
        Normalized inter-community edge probability matrix.
        
        Returns a K×K symmetric matrix where entry [i,j] represents the
        fraction of possible edges between community i and j that actually exist.
        
        IMPORTANT: The matrix is ALWAYS K×K (indexed 0 to K-1), even if some
        communities are not present in the graph. Rows/columns for absent
        communities will be all zeros.
        
        For within-community edges (i=j):
            matrix[i,i] = actual_edges / max_possible_edges
            max_possible_edges = n_i * (n_i - 1) / 2  (undirected, no self-loops)
        
        For between-community edges (i≠j):
            matrix[i,j] = actual_edges / max_possible_edges
            max_possible_edges = n_i * n_j
        
        Entries are 0 if either community i or j is not present in the graph.
        Values are in [0, 1] representing the fraction of possible edges that exist.
        
        Parameters
        ----------
        data : Data
            PyG Data object with edge_index and y (node labels).
        
        Returns
        -------
        torch.Tensor
            Symmetric matrix of shape [K, K] with normalized edge probabilities in [0, 1].
            Matrix is indexed by community ID (0 to K-1).
            Rows/columns for absent communities are all zeros.
        """
        y = data.y  # node labels [N]
        edge_index = data.edge_index  # [2, E]
        
        # Initialize K×K matrix (always full size, regardless of which communities are present)
        edge_counts = torch.zeros(self.K, self.K, dtype=torch.float32)
        
        # Count UNIQUE edges between each community pair
        # For undirected graphs, edge_index typically contains both (u,v) and (v,u)
        # We need to count each unique edge only once
        if edge_index.size(1) > 0:
            src_labels = y[edge_index[0]]
            dst_labels = y[edge_index[1]]
            
            for i in range(edge_index.size(1)):
                src, dst = int(src_labels[i].item()), int(dst_labels[i].item())
                # Only count each edge once by checking src <= dst
                # This handles the undirected graph representation
                if src <= dst:
                    edge_counts[src, dst] += 1
        
        # Make the matrix symmetric
        edge_counts = edge_counts + edge_counts.T - torch.diag(edge_counts.diag())
        
        # Count nodes in each community (ALWAYS length K, with 0s for absent communities)
        community_sizes = torch.bincount(y.long(), minlength=self.K).float()
        
        # Compute maximum possible edges between each pair (K×K matrix)
        max_edges = torch.zeros(self.K, self.K)
        for i in range(self.K):
            for j in range(i, self.K):
                if i == j:
                    # Within-community edges (undirected, no self-loops)
                    # Max edges = n*(n-1)/2 for undirected graphs
                    max_edges[i, i] = community_sizes[i] * (community_sizes[i] - 1) / 2
                else:
                    # Between-community edges (count each edge once)
                    max_edges[i, j] = community_sizes[i] * community_sizes[j]
                    max_edges[j, i] = max_edges[i, j]
        
        # Normalize: edge_prob[i,j] = actual_edges / max_possible_edges
        # Where max_edges is 0 (community not present), result will be 0
        edge_prob = torch.where(
            max_edges > 0,
            edge_counts / max_edges,
            torch.zeros_like(edge_counts)
        )
        
        # Clamp to [0, 1] to handle any numerical issues
        edge_prob = torch.clamp(edge_prob, 0.0, 1.0)
        
        # Result is K×K matrix with:
        # - Non-zero entries for present community pairs (based on actual edge density)
        # - Zero entries for absent communities (entire row/column is 0)
        # - All values in [0, 1]
        return edge_prob
    
    # =========================================================================
    # Batch Processing
    # =========================================================================
    
    def compute_all_properties(
        self,
        data_list: List[Data],
        include_complex: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all graph properties for a list of graphs.
        
        Parameters
        ----------
        data_list : List[Data]
            List of PyG Data objects.
        include_complex : bool
            Whether to include complex properties (vectors/matrices).
            Set to False for faster computation if only scalars are needed.
        
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with property names as keys and tensors as values.
            
            Simple properties have shape [batch_size, 1]:
                - homophily, avg_degree, size, gini, diameter
            
            Complex properties have shapes:
                - community_presence: [batch_size, K]
                - edge_prob_matrix: [batch_size, K, K]
        """
        batch_size = len(data_list)
        
        # Initialize storage for simple properties
        properties = {
            'homophily': torch.zeros(batch_size, 1),
            'avg_degree': torch.zeros(batch_size, 1),
            'size': torch.zeros(batch_size, 1),
            'gini': torch.zeros(batch_size, 1),
            'diameter': torch.zeros(batch_size, 1),
        }
        
        # Initialize storage for complex properties
        if include_complex:
            properties['community_presence'] = torch.zeros(batch_size, self.K)
            properties['edge_prob_matrix'] = torch.zeros(batch_size, self.K, self.K)
        
        # Compute properties for each graph
        for i, data in enumerate(data_list):
            # Simple properties
            properties['homophily'][i, 0] = self.compute_homophily(data)
            properties['avg_degree'][i, 0] = self.compute_avg_degree(data)
            properties['size'][i, 0] = self.compute_size(data)
            properties['gini'][i, 0] = self.compute_gini(data)
            properties['diameter'][i, 0] = self.compute_diameter(data)
            
            # Complex properties
            if include_complex:
                properties['community_presence'][i] = self.compute_community_presence(data)
                properties['edge_prob_matrix'][i] = self.compute_edge_prob_matrix(data)
        
        return properties
    
    def compute_simple_properties_only(self, data_list: List[Data]) -> Dict[str, torch.Tensor]:
        """
        Compute only simple (scalar) properties for faster processing.
        
        Equivalent to compute_all_properties(data_list, include_complex=False).
        """
        return self.compute_all_properties(data_list, include_complex=False)


# =============================================================================
# Utility Functions for Adding Properties to Dataset
# =============================================================================

def add_properties_to_dataset(
    data_list: List[Data],
    K: int,
    include_complex: bool = True,
    verbose: bool = True,
) -> List[Data]:
    """
    Add computed properties as attributes to each Data object.
    
    This modifies the Data objects in-place by adding property attributes.
    
    Parameters
    ----------
    data_list : List[Data]
        List of PyG Data objects to augment.
    K : int
        Number of communities (from GraphUniverse config).
    include_complex : bool
        Whether to include complex properties.
    verbose : bool
        Whether to print progress.
    
    Returns
    -------
    List[Data]
        The same data_list with added property attributes.
    """
    computer = GraphPropertyComputer(K=K)
    
    if verbose:
        print(f"Computing properties for {len(data_list)} graphs...")
    
    properties = computer.compute_all_properties(data_list, include_complex=include_complex)
    
    # Add properties as attributes to each Data object
    for i, data in enumerate(data_list):
        # Simple properties
        data.property_homophily = properties['homophily'][i]
        data.property_avg_degree = properties['avg_degree'][i]
        data.property_size = properties['size'][i]
        data.property_gini = properties['gini'][i]
        data.property_diameter = properties['diameter'][i]
        
        # Complex properties
        if include_complex:
            data.property_community_presence = properties['community_presence'][i]
            data.property_edge_prob_matrix = properties['edge_prob_matrix'][i]
    
    if verbose:
        print("✓ Properties added to dataset")
    
    return data_list


def extract_K_from_config(config: dict) -> int:
    """
    Extract K (number of communities) from a TopoBench config.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (from wandb or yaml).
    
    Returns
    -------
    int
        Number of communities K.
    """
    # Try GraphUniverse-specific path first
    try:
        gen_params = config["dataset"]["loader"]["parameters"]["generation_parameters"]
        K = gen_params["universe_parameters"]["K"]
        return K
    except KeyError:
        pass
    
    # Try general num_classes
    try:
        K = config["dataset"]["parameters"]["num_classes"]
        return K
    except KeyError:
        pass
    
    raise ValueError("Could not extract K from config. Check config structure.")


# =============================================================================
# Testing & Visualization
# =============================================================================

def create_graphuniverse_test_graphs(K: int = 3, n_graphs: int = 5) -> List[Data]:
    """
    Create test graphs using GraphUniverse for realistic community structure.
    
    Parameters
    ----------
    K : int
        Number of communities.
    n_graphs : int
        Number of test graphs to create.
    
    Returns
    -------
    List[Data]
        List of test graphs generated by GraphUniverse (converted to PyG Data format).
    """
    try:
        from graph_universe import GraphUniverse, GraphFamilyGenerator
    except ImportError:
        raise ImportError(
            "GraphUniverse not installed. Install with: pip install graph-universe"
        )
    
    print(f"   Using GraphUniverse to generate {n_graphs} graphs with K={K} communities...")
    
    # Create universe with detailed parameters
    universe = GraphUniverse(
        K=K, 
        edge_propensity_variance=0.3, 
        feature_dim=10,
        seed=42
    )
    
    # Generate family with full parameter control
    family = GraphFamilyGenerator(
        universe=universe,
        min_n_nodes=25, 
        max_n_nodes=50,
        min_communities=2,
        max_communities=min(K, 7),  # Don't exceed K
        homophily_range=(0.2, 0.8),
        avg_degree_range=(2.0, 10.0),
        degree_distribution="power_law",
        power_law_exponent_range=(2.0, 5.0),
        degree_separation_range=(0.1, 0.7),
        seed=42
    )
    
    # Generate graphs (stores in family.graphs)
    family.generate_family(n_graphs=n_graphs, show_progress=False)
    
    # Convert to PyG format using built-in method
    pyg_graphs = family.to_pyg_graphs(task="community_detection")
    
    print(f"   ✓ Generated {len(pyg_graphs)} graphs using GraphUniverse")
    
    return pyg_graphs


def visualize_graph_with_properties(
    data: Data, 
    properties: Dict[str, torch.Tensor], 
    graph_idx: int = 0,
    K: int = None
):
    """
    Visualize a graph with its computed properties.
    
    Parameters
    ----------
    data : Data
        PyG Data object.
    properties : Dict[str, torch.Tensor]
        Properties computed by GraphPropertyComputer.
    graph_idx : int
        Index of the graph in the batch (for extracting from batched properties).
    K : int, optional
        Number of communities. If None, inferred from edge_prob_matrix shape.
    """
    import matplotlib.pyplot as plt
    
    # Infer K from edge_prob_matrix if not provided
    if K is None:
        K = properties['edge_prob_matrix'].shape[1]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 1. Graph visualization
    G = to_networkx(data, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by community
    node_colors = data.y.cpu().numpy()
    
    nx.draw(
        G, pos, 
        node_color=node_colors, 
        cmap='tab10',
        node_size=300,
        with_labels=True,
        ax=axes[0]
    )
    unique_comms = torch.unique(data.y).cpu().numpy()
    axes[0].set_title(f"Graph Structure\n(communities present: {list(unique_comms)})")
    
    # 2. Simple properties (bar plot)
    simple_props = {
        'Homophily': properties['homophily'][graph_idx, 0].item(),
        'Avg Degree': properties['avg_degree'][graph_idx, 0].item(),
        'Size': properties['size'][graph_idx, 0].item(),
        'GINI': properties['gini'][graph_idx, 0].item(),
        'Diameter': properties['diameter'][graph_idx, 0].item(),
    }
    
    axes[1].bar(range(len(simple_props)), list(simple_props.values()))
    axes[1].set_xticks(range(len(simple_props)))
    axes[1].set_xticklabels(simple_props.keys(), rotation=45, ha='right')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Simple Properties (5 scalars)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. Edge probability matrix (heatmap) - ALWAYS K×K
    edge_prob = properties['edge_prob_matrix'][graph_idx].cpu().numpy()
    
    im = axes[2].imshow(edge_prob, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    axes[2].set_xlabel('Community ID')
    axes[2].set_ylabel('Community ID')
    axes[2].set_title(f'Inter-Community Edge Probability\n({K}×{K} matrix, zeros for absent communities)')
    
    # Set ticks to show all community IDs
    tick_positions = list(range(K))
    axes[2].set_xticks(tick_positions)
    axes[2].set_yticks(tick_positions)
    axes[2].set_xticklabels(tick_positions)
    axes[2].set_yticklabels(tick_positions)
    
    # Add colorbar
    plt.colorbar(im, ax=axes[2], label='Edge Probability')
    
    # Add values to heatmap (only if K is small enough to read)
    if K <= 10:
        fontsize = max(6, 10 - K)  # Smaller font for larger K
        for i in range(K):
            for j in range(K):
                val = edge_prob[i, j]
                # Only show non-zero values to avoid clutter
                if val > 0.001:
                    text = axes[2].text(j, i, f'{val:.2f}',
                                       ha="center", va="center", 
                                       color="black" if val < 0.5 else "white",
                                       fontsize=fontsize)
    
    plt.tight_layout()
    plt.savefig('graph_properties_test.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to 'graph_properties_test.png'")
    plt.close()


def test_property_computation():
    """
    Test property computation with simple synthetic graphs.
    """
    print("\n" + "=" * 80)
    print("TESTING GRAPH PROPERTY COMPUTATION")
    print("=" * 80)
    
    # Create test graphs using GraphUniverse
    K = 3
    n_graphs = 5
    print(f"\n1. Creating {n_graphs} test graphs with K={K} communities using GraphUniverse...")
    graphs = create_graphuniverse_test_graphs(K=K, n_graphs=n_graphs)
    print(f"   ✓ Created {len(graphs)} graphs")
    print(f"   Example graph: {graphs[0].num_nodes} nodes, {graphs[0].edge_index.size(1)} edges")
    
    # Show which communities are present in first graph
    unique_comms = torch.unique(graphs[0].y).cpu().numpy()
    print(f"   Example graph communities: {list(unique_comms)}")
    
    # Initialize property computer
    print(f"\n2. Computing properties...")
    computer = GraphPropertyComputer(K=K)
    properties = computer.compute_all_properties(graphs, include_complex=True)
    
    print("\n3. Property shapes:")
    for name, tensor in properties.items():
        print(f"   {name:25s}: {list(tensor.shape)}")
    
    print("\n4. Sample values (first graph):")
    print(f"   Homophily:      {properties['homophily'][0, 0]:.4f}")
    print(f"   Avg Degree:     {properties['avg_degree'][0, 0]:.4f}")
    print(f"   Size:           {properties['size'][0, 0]:.0f}")
    print(f"   GINI:           {properties['gini'][0, 0]:.4f}")
    print(f"   Diameter:       {properties['diameter'][0, 0]:.0f}")
    print(f"   Community Presence: {properties['community_presence'][0]}")
    print(f"   Edge Prob Matrix:\n{properties['edge_prob_matrix'][0]}")
    
    print("\n5. Creating visualization (K=3 example)...")
    visualize_graph_with_properties(graphs[0], properties, graph_idx=0, K=K)
    
    # Test with GraphUniverse-like config
    print("\n6. Testing with mock GraphUniverse config...")
    mock_config = {
        "dataset": {
            "loader": {
                "parameters": {
                    "generation_parameters": {
                        "universe_parameters": {
                            "K": 10
                        }
                    }
                }
            },
            "parameters": {
                "num_classes": 10
            }
        }
    }
    
    K_extracted = extract_K_from_config(mock_config)
    print(f"   ✓ Extracted K={K_extracted} from config")
    
    # Test with incomplete communities (K=10, generate graphs that may not have all communities)
    print("\n7. Testing with K=10 (realistic GraphUniverse graphs)...")
    K_large = 10
    computer_large = GraphPropertyComputer(K=K_large)
    
    # Generate graphs with K=10 but allow min_communities < K
    # This naturally creates graphs where not all communities are present
    print("   Generating graphs with GraphUniverse (K=10, but min/max communities 3-6)...")
    graphs_large = create_graphuniverse_test_graphs(K=K_large, n_graphs=3)
    
    # Pick the first graph that doesn't have all communities present
    data_incomplete = None
    for g in graphs_large:
        unique_comms = torch.unique(g.y).cpu().numpy()
        if len(unique_comms) < K_large:
            data_incomplete = g
            active_comms = list(unique_comms)
            break
    
    # If all graphs have all communities, use the first one anyway
    if data_incomplete is None:
        data_incomplete = graphs_large[0]
        active_comms = list(torch.unique(data_incomplete.y).cpu().numpy())
    
    print(f"   Selected graph with {data_incomplete.num_nodes} nodes, {data_incomplete.edge_index.size(1)} edges")
    print(f"   Graph has communities: {active_comms} (out of K={K_large} possible)")
    
    # Compute properties
    props_incomplete = computer_large.compute_all_properties([data_incomplete], include_complex=True)
    
    print(f"\n   Computing properties for K={K_large} graph...")
    print(f"   Community presence vector shape: {props_incomplete['community_presence'].shape}")
    print(f"   Community presence: {props_incomplete['community_presence'][0]}")
    print(f"   Edge prob matrix shape: {props_incomplete['edge_prob_matrix'].shape}")
    
    edge_prob_incomplete = props_incomplete['edge_prob_matrix'][0]
    
    # Verify that absent communities have zero rows/columns
    absent_comms = [c for c in range(K_large) if c not in active_comms]
    
    if len(absent_comms) > 0:
        print(f"   Absent communities: {absent_comms}")
        for c in absent_comms:
            row_sum = edge_prob_incomplete[c].sum().item()
            col_sum = edge_prob_incomplete[:, c].sum().item()
            assert row_sum == 0.0, f"Row {c} should be all zeros (absent community)"
            assert col_sum == 0.0, f"Col {c} should be all zeros (absent community)"
        
        print(f"   ✓ Verified: Rows/columns {absent_comms} are all zeros (absent communities)")
    else:
        print(f"   Note: All {K_large} communities are present in this graph")
    
    # Verify that present communities have non-zero diagonal entries
    for c in active_comms:
        diag_val = edge_prob_incomplete[c, c].item()
        assert diag_val > 0.0, f"Diagonal entry [{c},{c}] should be non-zero (community {c} is present)"
    
    print(f"   ✓ Verified: Diagonal entries for present communities {active_comms} are non-zero")
    
    # Create visualization for the K=10 example
    print("\n8. Creating visualization for K=10 example (showing full 10×10 matrix)...")
    visualize_graph_with_properties(data_incomplete, props_incomplete, graph_idx=0, K=K_large)
    
    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
    print("\nKey takeaways:")
    print("  ✓ Edge probability matrix is ALWAYS K×K (indexed 0 to K-1)")
    print("  ✓ Rows/columns for ABSENT communities are all zeros")
    print("  ✓ This ensures consistent shape regardless of which communities appear in a graph")
    print("  ✓ Example: K=10, only communities [2,5,7] present → 10×10 matrix with zeros at rows/cols [0,1,3,4,6,8,9]")
    print(f"\n  📊 Check 'graph_properties_test.png' to see the full {K_large}×{K_large} matrix visualization!")
    print("\nUsage example:")
    print("```python")
    print("from graph_properties import GraphPropertyComputer, extract_K_from_config")
    print("")
    print("# Load config and dataset")
    print("config = load_wandb_config(run_dir)")
    print("K = extract_K_from_config(config)")
    print("")
    print("# Compute properties")
    print("computer = GraphPropertyComputer(K=K)")
    print("properties = computer.compute_all_properties(data_list)")
    print("")
    print("# Properties dict contains:")
    print("#   - Simple: homophily, avg_degree, size, gini, diameter [batch_size, 1]")
    print("#   - Complex: community_presence [batch_size, K], edge_prob_matrix [batch_size, K, K]")
    print("```\n")


if __name__ == "__main__":
    test_property_computation()

