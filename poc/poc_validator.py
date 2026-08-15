"""
POC Step 2: GNN Validator Concept

This script proves we can:
1. Load an extracted graph
2. Manually corrupt it (simulate defects)
3. Convert both to PyTorch Geometric format
4. Pass them through a simple GNN
5. Get distinguishable embeddings/scores

This is NOT training. Just proving the tensor conversion and forward pass work.

Run: python poc_validator.py
"""

import os
import json
import random
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt


class SimpleGNN(torch.nn.Module):
    """
    Simple 2-layer GCN for graph classification.
    
    Architecture:
    - 2 GCN layers with ReLU activation
    - Global mean pooling to get graph-level embedding
    - MLP classifier head with 4 outputs:
      * consistency_score
      * missing_entity_prob
      * contradiction_prob  
      * fragmentation_prob
    """
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        
        # GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Classifier heads
        self.fc_consistency = torch.nn.Linear(hidden_dim, 1)
        self.fc_missing = torch.nn.Linear(hidden_dim, 1)
        self.fc_contradiction = torch.nn.Linear(hidden_dim, 1)
        self.fc_fragmentation = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, batch):
        """
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity [2, num_edges]
            batch: Batch assignment [num_nodes] (which graph each node belongs to)
        
        Returns:
            dict with 4 scores, each in [0, 1]
        """
        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Global pooling (one embedding per graph in batch)
        x = global_mean_pool(x, batch)  # [num_graphs, hidden_dim]
        
        # Classifier heads
        consistency = torch.sigmoid(self.fc_consistency(x))
        missing = torch.sigmoid(self.fc_missing(x))
        contradiction = torch.sigmoid(self.fc_contradiction(x))
        fragmentation = torch.sigmoid(self.fc_fragmentation(x))
        
        return {
            'consistency': consistency.squeeze(),
            'missing_entities': missing.squeeze(),
            'contradictions': contradiction.squeeze(),
            'fragmentation': fragmentation.squeeze()
        }


def load_extracted_graph():
    """Load the graph from POC step 1."""
    graph_path = "data/raw_extraction_response.json"

    if not os.path.exists(graph_path):
        print(f"Error: {graph_path} not found. Run poc_extraction.py first.")
        return None

    # Read JSON file
    with open(graph_path, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    # Create NetworkX graph
    G = nx.DiGraph()  # Directed graph for relations

    # Add nodes (entities) with attributes
    if 'entities' in graph_data:
        for entity in graph_data['entities']:
            entity_id = entity['id']
            # Add node with all attributes except 'id'
            node_attrs = {k: v for k, v in entity.items() if k != 'id'}
            G.add_node(entity_id, **node_attrs)

    # Add edges (relations) with attributes
    if 'relations' in graph_data:
        for relation in graph_data['relations']:
            source = relation['source']
            target = relation['target']
            # Add edge with all attributes except 'source' and 'target'
            edge_attrs = {k: v for k, v in relation.items() if k not in ['source', 'target']}
            G.add_edge(source, target, **edge_attrs)

    print(f"✓ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G

def corrupt_graph(G, corruption_type="missing_entities"):
    """
    Manually corrupt a graph to simulate defects.
    
    Corruption types:
    - missing_entities: Remove high-degree nodes
    - contradictions: Flip edge directions randomly
    - fragmentation: Remove bridge edges to disconnect graph
    """
    G_corrupted = G.copy()
    
    if corruption_type == "missing_entities":
        # Remove 20% of highest-degree nodes
        degrees = dict(G_corrupted.degree())
        sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        num_to_remove = max(1, int(0.2 * len(sorted_nodes)))
        nodes_to_remove = [node for node, _ in sorted_nodes[:num_to_remove]]
        G_corrupted.remove_nodes_from(nodes_to_remove)
        print(f"  - Removed {num_to_remove} high-degree nodes")
    
    elif corruption_type == "contradictions":
        # Flip 30% of edge directions randomly
        edges = list(G_corrupted.edges())
        num_to_flip = max(1, int(0.3 * len(edges)))
        edges_to_flip = random.sample(edges, num_to_flip)
        
        for u, v in edges_to_flip:
            data = G_corrupted.edges[u, v]
            G_corrupted.remove_edge(u, v)
            G_corrupted.add_edge(v, u, **data)  # Reversed
        
        print(f"  - Flipped {num_to_flip} edge directions")
    
    elif corruption_type == "fragmentation":
        # Remove edges to increase number of connected components
        edges = list(G_corrupted.edges())
        num_to_remove = max(1, int(0.3 * len(edges)))
        edges_to_remove = random.sample(edges, num_to_remove)
        G_corrupted.remove_edges_from(edges_to_remove)
        print(f"  - Removed {num_to_remove} edges")
    
    return G_corrupted


def build_node_features(G):
    """
    Build simple node features for GNN input.
    
    For POC, we use basic features:
    - One-hot entity type (5 types)
    - Document frequency (normalized)
    - Confidence score
    - Node degree (normalized)
    
    Returns: [num_nodes, feature_dim] tensor
    """
    entity_types = ["PERSON", "ORGANIZATION", "LOCATION", "EVENT", "CONCEPT"]
    type_to_idx = {t: i for i, t in enumerate(entity_types)}
    
    features = []
    max_degree = max(dict(G.degree()).values()) if G.number_of_nodes() > 0 else 1
    max_doc_freq = max(G.nodes[n].get("document_frequency", 1) for n in G.nodes()) if G.number_of_nodes() > 0 else 1
    
    for node in G.nodes():
        attrs = G.nodes[node]
        
        # One-hot entity type
        entity_type = attrs.get("type", "CONCEPT")
        type_vec = [0.0] * 5
        type_vec[type_to_idx[entity_type]] = 1.0
        
        # Numerical features
        doc_freq_norm = attrs.get("document_frequency", 1) / max_doc_freq
        confidence = attrs.get("confidence", 1.0)
        degree_norm = G.degree(node) / max_degree
        
        # Combine into feature vector
        feature_vec = type_vec + [doc_freq_norm, confidence, degree_norm]
        features.append(feature_vec)
    
    return torch.tensor(features, dtype=torch.float)


def networkx_to_pyg(G):
    """
    Convert NetworkX graph to PyTorch Geometric Data object.
    
    Returns:
        PyG Data object with:
        - x: node features [num_nodes, feature_dim]
        - edge_index: connectivity [2, num_edges]
        - num_nodes: int
    """
    # Node features
    x = build_node_features(G)
    
    # Edge index (convert node IDs to indices)
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}
    edge_index = []
    
    for u, v in G.edges():
        edge_index.append([node_to_idx[u], node_to_idx[v]])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, num_nodes=G.number_of_nodes())
    
    return data


def visualize_comparison(G_clean, G_corrupted, corruption_type):
    """Visualize clean vs corrupted graph side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for ax, G, title in zip(axes, [G_clean, G_corrupted], ["Clean Graph", f"Corrupted ({corruption_type})"]):
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "Empty graph", ha="center", va="center", fontsize=12)
            ax.axis("off")
            continue
        
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Color by type
        type_colors = {
            "PERSON": "#ff6b6b",
            "ORGANIZATION": "#4ecdc4",
            "LOCATION": "#45b7d1",
            "EVENT": "#ffa07a",
            "CONCEPT": "#98d8c8"
        }
        node_colors = [type_colors.get(G.nodes[node].get("type", "CONCEPT"), "#cccccc") for node in G.nodes()]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.9, ax=ax)
        nx.draw_networkx_edges(G, pos, edge_color="#666", arrows=True, arrowsize=15, width=1.5, alpha=0.6, ax=ax)
        
        labels = {node: G.nodes[node]["name"][:15] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)
        
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig("data/graph_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ Comparison visualization saved to data/graph_comparison.png")
    plt.show()


def main():
    """Run the validator POC."""
    print("\n" + "="*80)
    print("POC STEP 2: GNN VALIDATOR CONCEPT")
    print("="*80 + "\n")
    
    # Step 1: Load clean graph
    G_clean = load_extracted_graph()
    if G_clean is None:
        return
    
    print(f"\nClean graph stats:")
    print(f"  Nodes: {G_clean.number_of_nodes()}")
    print(f"  Edges: {G_clean.number_of_edges()}")
    print(f"  Connected components: {nx.number_weakly_connected_components(G_clean)}")
    
    # Step 2: Create corrupted versions
    print(f"\nCreating corrupted graphs...")
    corruption_types = ["missing_entities", "contradictions", "fragmentation"]
    corrupted_graphs = {}
    
    for ctype in corruption_types:
        print(f"\nCorruption type: {ctype}")
        G_corrupted = corrupt_graph(G_clean, ctype)
        corrupted_graphs[ctype] = G_corrupted
        
        print(f"  Corrupted graph stats:")
        print(f"    Nodes: {G_corrupted.number_of_nodes()}")
        print(f"    Edges: {G_corrupted.number_of_edges()}")
        print(f"    Connected components: {nx.number_weakly_connected_components(G_corrupted)}")
    
    # Step 3: Convert to PyTorch Geometric
    print(f"\n{'='*80}")
    print("Converting graphs to PyTorch Geometric format...")
    print(f"{'='*80}\n")
    
    data_clean = networkx_to_pyg(G_clean)
    print(f"Clean graph PyG Data:")
    print(f"  Node features: {data_clean.x.shape}")
    print(f"  Edge index: {data_clean.edge_index.shape}")
    
    data_corrupted = {}
    for ctype, G in corrupted_graphs.items():
        data_corrupted[ctype] = networkx_to_pyg(G)
        print(f"\n{ctype} PyG Data:")
        print(f"  Node features: {data_corrupted[ctype].x.shape}")
        print(f"  Edge index: {data_corrupted[ctype].edge_index.shape}")
    
    # Step 4: Initialize GNN model
    print(f"\n{'='*80}")
    print("Initializing GNN model...")
    print(f"{'='*80}\n")
    
    input_dim = data_clean.x.shape[1]  # Feature dimension
    model = SimpleGNN(input_dim=input_dim, hidden_dim=64)
    model.eval()  # Evaluation mode (no training yet)
    
    print(f"Model architecture:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: 64")
    print(f"  Output: 4 scores (consistency, missing, contradiction, fragmentation)")
    
    # Step 5: Forward pass
    print(f"\n{'='*80}")
    print("Running forward pass (random weights, no training)...")
    print(f"{'='*80}\n")
    
    with torch.no_grad():
        # Create batch tensor (all nodes belong to graph 0)
        batch_clean = torch.zeros(data_clean.num_nodes, dtype=torch.long)
        scores_clean = model(data_clean.x, data_clean.edge_index, batch_clean)
        
        print("Clean graph scores:")
        for key, value in scores_clean.items():
            print(f"  {key}: {value.item():.4f}")
        
        print("\nCorrupted graph scores:")
        for ctype, data in data_corrupted.items():
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            scores = model(data.x, data.edge_index, batch)
            
            print(f"\n  {ctype}:")
            for key, value in scores.items():
                print(f"    {key}: {value.item():.4f}")
    
    # Step 6: Visualize one corruption
    print(f"\n{'='*80}")
    print("Visualizing clean vs corrupted (missing_entities)...")
    print(f"{'='*80}\n")
    
    visualize_comparison(G_clean, corrupted_graphs["missing_entities"], "missing_entities")
    
    print(f"\n{'='*80}")
    print("✓ POC STEP 2 COMPLETE")
    print(f"{'='*80}\n")
    
    print("Key takeaways:")
    print("  • NetworkX → PyG conversion works")
    print("  • GNN forward pass produces scores")
    print("  • Corruption functions create distinguishable defects")
    print("  • Next step: Generate training data and actually train the model")


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    main()
