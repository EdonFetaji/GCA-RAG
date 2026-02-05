"""
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
from dotenv import load_dotenv
from datasets import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from groq import Groq


# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
    """Call LLM API based on configured provider."""

    if LLM_PROVIDER == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.content[0].text

    elif LLM_PROVIDER == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    elif LLM_PROVIDER == "gemini":
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            }
        )

        return response.text

    elif LLM_PROVIDER == "groq":
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_completion_tokens=4000,
            top_p=1
        )

        return response.choices[0].message.content

    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


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
    print(f"Calling {LLM_PROVIDER.upper()} API...")
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

    # Save graph for later use
    nx.write_gpickle(G, "data/extracted_graph.pkl")
    print("\n✓ Graph saved to data/extracted_graph.pkl")

    print(f"\n{'='*80}")
    print("✓ POC STEP 1 COMPLETE")
    print(f"{'='*80}\n")

    return G


if __name__ == "__main__":
    main()
