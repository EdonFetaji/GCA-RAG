"""
LEGACY / REFERENCE ONLY — not the canonical extraction path.

This was POC Step 1, the original proof-of-concept that KG extraction from
a Multi-News cluster works at all: single inline prompt, one LLM call, no
evidence tracing, no grading, no ontology customization. It's been
superseded by extraction/service.py's five methods (ontology-constrained,
evidence-traced, two-agent finder/grader, evidence accumulation, and the
full pipeline combining all of them), which is the production extraction
path per docs/adr/0001-canonical-extraction-path.md.

Kept around because it's small, self-contained, and useful for a quick
sanity check that an API key + model + graph pipeline works end-to-end
without spinning up the FastAPI service — but new work should go through
extraction/service.py (or extraction/router.py's endpoints), not here.

---

POC Step 1: Extract Knowledge Graph from a single Multi-News cluster

This script proves the core extraction logic works:
1. Load one document cluster from Multi-News
2. Send documents to LLM with extraction prompt
3. Parse JSON response into NetworkX graph
4. Visualize the result

Run: python poc_extraction.py
"""

import os
import json
import pickle
from dotenv import load_dotenv
from datasets import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from cerebras.cloud.sdk import Cerebras


# Load environment variables
load_dotenv()

# Configuration
# The repo previously supported Anthropic/OpenAI/Gemini/Groq here, but
# utils/llm_utils.py (used by extraction/service.py) and extractor_agent/
# only ever implemented Cerebras. Standardizing every entrypoint on
# Cerebras so one CEREBRAS_API_KEY is enough to run the whole repo — see
# Track 0.2 in the work plan for the alternative (restore multi-provider).
# Cerebras retires/rotates model IDs on its public endpoints periodically
# (llama-3.1-8b/70b, used here previously, are gone as of mid-2026), so this
# is read from the environment with a current production-tier default
# rather than hardcoded.
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")


def load_single_cluster(cluster_idx=0):
    """Load one document cluster from Multi-News test set."""
    print("Loading Multi-News dataset...")
    dataset = load_dataset("multi_news", split="test", trust_remote_code=True)

    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")  # Multi-News separates docs with |||||
    reference_summary = cluster["summary"]

    print(f"\n{'='*80}")
    print(f"Loaded cluster {cluster_idx}")
    print(f"Number of documents: {len(documents)}")
    print(f"Reference summary length: {len(reference_summary)} chars")
    print(f"{'='*80}\n")

    return documents, reference_summary


def build_extraction_prompt(documents):
    """
    Build prompt that instructs LLM to extract entities and relations as JSON.

    The prompt is designed to output valid graph structure:
    - Every relation references entities that exist in the entity list
    - Entities have types and metadata
    - Relations have types and confidence scores
    """
    docs_text = "\n\n---DOCUMENT---\n\n".join(documents[:5])  # Limit to first 5 docs to stay under token limits

    prompt = f"""You are extracting a knowledge graph from a cluster of news articles about the same topic.

Extract:
1. **Entities**: Key people, organizations, locations, events, concepts mentioned across documents
2. **Relations**: How these entities relate to each other

Output ONLY valid JSON in this exact format (no markdown, no code fences, no commentary):
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "PERSON|ORGANIZATION|LOCATION|EVENT|CONCEPT",
      "document_frequency": 3,
      "confidence": 0.95
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "verb phrase describing relation",
      "support_count": 2,
      "source_documents": [0, 1],
      "confidence": 0.85
    }}
  ]
}}

CRITICAL RULES:
- Every relation's source and target MUST reference an entity id from the entities list
- Use past tense verbs for relations (e.g., "announced", "acquired", "launched")
- document_frequency = how many documents mention this entity
- support_count = how many documents support this relation
- confidence = how certain you are this entity/relation is correct (0.0 to 1.0)

Documents:
{docs_text}

Extract the knowledge graph as JSON:"""

    return prompt


def call_llm(prompt):
    """Call the Cerebras API (see the CEREBRAS_MODEL note above)."""
    client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))

    response = client.chat.completions.create(
        model=CEREBRAS_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=4000,
    )

    return response.choices[0].message.content


def parse_json_to_graph(json_text):
    """
    Parse LLM JSON output into NetworkX directed graph.

    Validates:
    - All relations reference existing entities
    - No duplicate entity IDs
    - All required fields present
    """
    # Extract JSON from markdown code blocks if present
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]

    data = json.loads(json_text.strip())

    # Validate structure
    assert "entities" in data, "Missing 'entities' key in JSON"
    assert "relations" in data, "Missing 'relations' key in JSON"

    # Build graph
    G = nx.DiGraph()

    # Add nodes with attributes
    entity_ids = set()
    for entity in data["entities"]:
        entity_id = entity["id"]
        entity_ids.add(entity_id)

        G.add_node(
            entity_id,
            name=entity["name"],
            type=entity["type"],
            document_frequency=entity.get("document_frequency", 1),
            confidence=entity.get("confidence", 1.0)
        )

    # Add edges with attributes
    for relation in data["relations"]:
        source = relation["source"]
        target = relation["target"]

        # Validate relation references existing entities
        assert source in entity_ids, f"Relation source '{source}' not in entities"
        assert target in entity_ids, f"Relation target '{target}' not in entities"

        G.add_edge(
            source,
            target,
            relation_type=relation["relation_type"],
            support_count=relation.get("support_count", 1),
            source_documents=relation.get("source_documents", []),
            confidence=relation.get("confidence", 1.0)
        )

    print(f"\n✓ Valid graph created:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    return G


def repair_json_with_llm(bad_json_text):
    """Ask the LLM to repair invalid JSON output (one-shot fix)."""
    repair_prompt = f"""The following is supposed to be valid JSON but is malformed or truncated.
Fix it and return ONLY valid JSON with the same schema as originally requested.
No markdown. No commentary.

Malformed JSON:
{bad_json_text}
"""

    return call_llm(repair_prompt)


def visualize_graph(G, save_path="data/kg_visualization.png"):
    """Visualize the knowledge graph with node colors by type."""
    plt.figure(figsize=(14, 10))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Color nodes by type
    type_colors = {
        "PERSON": "#ff6b6b",
        "ORGANIZATION": "#4ecdc4",
        "LOCATION": "#45b7d1",
        "EVENT": "#ffa07a",
        "CONCEPT": "#98d8c8"
    }

    node_colors = [type_colors.get(G.nodes[node].get("type", "CONCEPT"), "#cccccc") for node in G.nodes()]

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=800,
        alpha=0.9
    )

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#666666",
        arrows=True,
        arrowsize=20,
        arrowstyle="->",
        width=1.5,
        alpha=0.6
    )

    # Draw labels (entity names)
    labels = {node: G.nodes[node]["name"] for node in G.nodes()}
    nx.draw_networkx_labels(
        G, pos,
        labels,
        font_size=8,
        font_weight="bold"
    )

    plt.title("Extracted Knowledge Graph", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    os.makedirs("data", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Graph visualization saved to {save_path}")

    plt.show()


def print_graph_details(G):
    """Print detailed information about the extracted graph."""
    print(f"\n{'='*80}")
    print("KNOWLEDGE GRAPH DETAILS")
    print(f"{'='*80}\n")

    print("ENTITIES:")
    for node in G.nodes():
        attrs = G.nodes[node]
        print(f"  • {attrs['name']} ({attrs['type']})")
        print(f"    - Document frequency: {attrs['document_frequency']}")
        print(f"    - Confidence: {attrs['confidence']:.2f}")
        print()

    print("\nRELATIONS:")
    for source, target in G.edges():
        attrs = G.edges[source, target]
        source_name = G.nodes[source]["name"]
        target_name = G.nodes[target]["name"]
        print(f"  • {source_name} → {target_name}")
        print(f"    - Type: {attrs['relation_type']}")
        print(f"    - Support: {attrs['support_count']} documents")
        print(f"    - Confidence: {attrs['confidence']:.2f}")
        print()


def main():
    """Run the complete extraction POC."""
    print("\n" + "="*80)
    print("POC STEP 1: KNOWLEDGE GRAPH EXTRACTION")
    print("="*80 + "\n")

    # Step 1: Load data
    documents, reference_summary = load_single_cluster(cluster_idx=0)

    # Show first document preview
    print("First document preview:")
    print(documents[0][:500] + "...\n")

    # Step 2: Build prompt
    print("Building extraction prompt...")
    prompt = build_extraction_prompt(documents)
    print(f"✓ Prompt built ({len(prompt)} chars)\n")

    # Step 3: Call LLM
    print(f"Calling CEREBRAS API ({CEREBRAS_MODEL})...")
    response = call_llm(prompt)
    print(f"✓ Received response ({len(response)} chars)\n")

    # Save raw response for inspection
    os.makedirs("data", exist_ok=True)
    with open("data/raw_extraction_response.json", "w") as f:
        f.write(response)
    print("✓ Raw response saved to data/raw_extraction_response.json")

    # Step 4: Parse to graph
    print("\nParsing JSON to NetworkX graph...")
    try:
        G = parse_json_to_graph(response)
    except Exception as e:
        print(f"\n✗ ERROR parsing response: {e}")
        print("\nRaw response:")
        print(response)

        print("\nAttempting to repair JSON with a second LLM call...")
        try:
            repaired = repair_json_with_llm(response)
            G = parse_json_to_graph(repaired)
            response = repaired
            print("✓ JSON repaired successfully")
        except Exception as repair_error:
            print(f"\n✗ ERROR repairing response: {repair_error}")
            print("\nRepaired response:")
            print(repaired if "repaired" in locals() else "<no repaired response>")
            return

    # Step 5: Print details
    print_graph_details(G)

    # Step 6: Visualize
    print("\nVisualizing graph...")
    visualize_graph(G)

    # Save graph for later use.
    # nx.write_gpickle()/read_gpickle() were removed in networkx 3.x, so we
    # pickle the graph object directly instead.
    with open("data/extracted_graph.pkl", "wb") as f:
        pickle.dump(G, f)
    print("\n✓ Graph saved to data/extracted_graph.pkl")

    print(f"\n{'='*80}")
    print("✓ POC STEP 1 COMPLETE")
    print(f"{'='*80}\n")

    return G


if __name__ == "__main__":
    main()
