"""
POC: Knowledge Graph Extraction using kg-gen library

This script tests the kg-gen library for KG extraction on Multi-News clusters
and compares it to our manual LLM extraction approach.

Features tested:
1. Basic single-text KG generation
2. Chunked large-text KG generation
3. Multi-document aggregation + clustering
4. Visualization with kg-gen's built-in visualizer

Install: pip install kg-gen

Run: python poc_kggen_extraction.py
"""

import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
from kg_gen import KGGen

# Load environment variables
load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────
# kg-gen uses LiteLLM model strings: "{provider}/{model_name}"
# See https://docs.litellm.ai/docs/providers for format details

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "cerebras")

MODEL_MAP = {
    "cerebras": f"cerebras/{os.getenv('CEREBRAS_MODEL', 'llama-3.3-70b')}",
    "anthropic": f"anthropic/{os.getenv('ANTHROPIC_MODEL', 'claude-sonnet-4-20250514')}",
    "openai": f"openai/{os.getenv('OPENAI_MODEL', 'gpt-4-turbo-preview')}",
    "gemini": f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')}",
    "groq": f"groq/{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
}

API_KEY_MAP = {
    "cerebras": os.getenv("CEREBRAS_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "openai": os.getenv("OPENAI_API_KEY"),
    "gemini": os.getenv("GEMINI_API_KEY"),
    "groq": os.getenv("GROQ_API_KEY"),
}

MODEL = MODEL_MAP.get(LLM_PROVIDER, MODEL_MAP["anthropic"])
API_KEY = API_KEY_MAP.get(LLM_PROVIDER)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_single_cluster(cluster_idx=0):
    """Load one document cluster from Multi-News test set."""
    print("Loading Multi-News dataset...")
    dataset = load_dataset("multi_news", split="test", trust_remote_code=True)

    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")
    reference_summary = cluster["summary"]

    print(f"\n{'='*80}")
    print(f"Loaded cluster {cluster_idx}")
    print(f"Number of documents: {len(documents)}")
    print(f"Reference summary length: {len(reference_summary)} chars")
    print(f"{'='*80}\n")

    return documents, reference_summary


# ── KG-Gen Tests ─────────────────────────────────────────────────────────────

def test_basic_extraction(kg, text, context="News article"):
    """Test 1: Basic single-text KG generation."""
    print("\n" + "─"*80)
    print("TEST 1: Basic Single-Text Extraction")
    print("─"*80)

    graph = kg.generate(
        input_data=text,
        context=context,
    )

    print(f"\n✓ Graph generated:")
    print(f"  Entities ({len(graph.entities)}): {graph.entities}")
    print(f"  Edge types ({len(graph.edges)}): {graph.edges}")
    print(f"  Relations ({len(graph.relations)}):")
    for subj, rel, obj in sorted(graph.relations):
        print(f"    • {subj} → [{rel}] → {obj}")

    return graph


def test_chunked_extraction(kg, text, context="News article cluster"):
    """Test 2: Chunked large-text KG generation with clustering."""
    print("\n" + "─"*80)
    print("TEST 2: Chunked Extraction with Clustering")
    print("─"*80)

    graph = kg.generate(
        input_data=text,
        chunk_size=3000,
        cluster=True,
        context=context,
    )

    print(f"\n✓ Chunked + clustered graph generated:")
    print(f"  Entities ({len(graph.entities)}): {graph.entities}")
    print(f"  Edge types ({len(graph.edges)}): {graph.edges}")
    print(f"  Relations ({len(graph.relations)}):")
    for subj, rel, obj in sorted(graph.relations):
        print(f"    • {subj} → [{rel}] → {obj}")

    if hasattr(graph, "entity_clusters") and graph.entity_clusters:
        print(f"\n  Entity clusters:")
        for canonical, aliases in graph.entity_clusters.items():
            if len(aliases) > 1:
                print(f"    • {canonical}: {aliases}")

    if hasattr(graph, "edge_clusters") and graph.edge_clusters:
        print(f"\n  Edge clusters:")
        for canonical, aliases in graph.edge_clusters.items():
            if len(aliases) > 1:
                print(f"    • {canonical}: {aliases}")

    return graph


def test_multi_doc_aggregation(kg, documents, context="Multi-document news cluster"):
    """Test 3: Generate KG per document, then aggregate and cluster."""
    print("\n" + "─"*80)
    print("TEST 3: Multi-Document Aggregation")
    print("─"*80)

    docs = [d.strip() for d in documents[:5] if d.strip()]
    graphs = []

    for i, doc in enumerate(docs):
        print(f"\n  Generating graph for document {i+1}/{len(docs)}...")
        g = kg.generate(
            input_data=doc,
            context=context,
        )
        print(f"    → {len(g.entities)} entities, {len(g.relations)} relations")
        graphs.append(g)

    # Aggregate all per-document graphs
    print("\n  Aggregating graphs...")
    combined = kg.aggregate(graphs)
    print(f"  ✓ Combined: {len(combined.entities)} entities, {len(combined.relations)} relations")

    # Cluster to merge duplicate entities/edges
    print("  Clustering combined graph...")
    clustered = kg.cluster(combined, context=context)
    print(f"  ✓ Clustered: {len(clustered.entities)} entities, {len(clustered.relations)} relations")

    print(f"\n  Final entities: {clustered.entities}")
    print(f"  Final relations:")
    for subj, rel, obj in sorted(clustered.relations):
        print(f"    • {subj} → [{rel}] → {obj}")

    if hasattr(clustered, "entity_clusters") and clustered.entity_clusters:
        print(f"\n  Entity clusters:")
        for canonical, aliases in clustered.entity_clusters.items():
            if len(aliases) > 1:
                print(f"    • {canonical}: {aliases}")

    return clustered


def save_graph_data(graph, path):
    """Save kg-gen graph data to JSON for inspection."""
    data = {
        "entities": sorted(graph.entities),
        "edges": sorted(graph.edges),
        "relations": sorted(
            [{"subject": s, "relation": r, "object": o} for s, r, o in graph.relations],
            key=lambda x: (x["subject"], x["relation"], x["object"])
        ),
    }
    if hasattr(graph, "entity_clusters") and graph.entity_clusters:
        data["entity_clusters"] = {
            k: sorted(v) for k, v in graph.entity_clusters.items()
        }
    if hasattr(graph, "edge_clusters") and graph.edge_clusters:
        data["edge_clusters"] = {
            k: sorted(v) for k, v in graph.edge_clusters.items()
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Graph data saved to {path}")


def main():
    print("\n" + "="*80)
    print("POC: kg-gen LIBRARY FOR KNOWLEDGE GRAPH EXTRACTION")
    print("="*80)

    print(f"\nUsing model: {MODEL}")
    print(f"Provider: {LLM_PROVIDER}")

    # Initialize KGGen
    kg = KGGen(
        model=MODEL,
        temperature=0.0,
        api_key=API_KEY,
    )

    # Load Multi-News data
    documents, reference_summary = load_single_cluster(cluster_idx=0)

    # Preview
    print("First document preview:")
    print(documents[0][:500] + "...\n")

    # ── Test 1: Basic extraction on a single document ────────────────────
    graph_basic = test_basic_extraction(
        kg,
        text=documents[0].strip(),
        context="News article about current events",
    )
    save_graph_data(graph_basic, "data/kggen_basic.json")

    # ── Test 2: Chunked extraction on all documents concatenated ─────────
    all_text = "\n\n".join(d.strip() for d in documents[:5] if d.strip())
    graph_chunked = test_chunked_extraction(
        kg,
        text=all_text,
        context="Cluster of news articles about the same topic",
    )
    save_graph_data(graph_chunked, "data/kggen_chunked.json")

    # ── Test 3: Per-document extraction → aggregate → cluster ────────────
    graph_aggregated = test_multi_doc_aggregation(
        kg,
        documents=documents,
        context="Cluster of news articles about the same topic",
    )
    save_graph_data(graph_aggregated, "data/kggen_aggregated.json")

    # ── Visualize the best graph ─────────────────────────────────────────
    print("\n" + "─"*80)
    print("VISUALIZATION")
    print("─"*80)
    try:
        viz_path = "data/kggen_graph.html"
        KGGen.visualize(graph_aggregated, output_path=viz_path, open_in_browser=False)
        print(f"✓ Interactive visualization saved to {viz_path}")
    except Exception as e:
        print(f"⚠ Visualization failed (pyvis may not be installed): {e}")
        print("  Install with: pip install 'kg-gen[dev]'")

    # ── Summary comparison ───────────────────────────────────────────────
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    print(f"\n  {'Method':<30} {'Entities':>10} {'Relations':>10}")
    print(f"  {'─'*30} {'─'*10} {'─'*10}")
    print(f"  {'Basic (single doc)':<30} {len(graph_basic.entities):>10} {len(graph_basic.relations):>10}")
    print(f"  {'Chunked (all docs)':<30} {len(graph_chunked.entities):>10} {len(graph_chunked.relations):>10}")
    print(f"  {'Aggregated + clustered':<30} {len(graph_aggregated.entities):>10} {len(graph_aggregated.relations):>10}")

    print(f"\n{'='*80}")
    print("✓ kg-gen POC COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
