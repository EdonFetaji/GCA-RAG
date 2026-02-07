import os
import json
import time
from dotenv import load_dotenv
from datasets import load_dataset
import networkx as nx
import matplotlib.pyplot as plt
from anthropic import Anthropic
from openai import OpenAI
from google import genai
from groq import Groq
from cerebras.cloud.sdk import Cerebras
import pickle
import requests

# Load environment variables
load_dotenv()

# Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "llama-3.3-70b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://627b-136-110-32-230.ngrok-free.app")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:9b")

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries


def load_single_cluster(cluster_idx=0):
    """Load one document cluster from Multi-News test set."""
    print("Loading Multi-News dataset...")
    dataset = load_dataset("multi_news", split="test", trust_remote_code=True)

    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")
    reference_summary = cluster["summary"]

    print(f"\n{'=' * 80}")
    print(f"Loaded cluster {cluster_idx}")
    print(f"Number of documents: {len(documents)}")
    print(f"Reference summary length: {len(reference_summary)} chars")
    print(f"{'=' * 80}\n")

    return documents, reference_summary


def build_extraction_prompt(documents):
    """Build prompt that instructs LLM to extract entities and relations as JSON."""
    docs_text = "\n\n---DOCUMENT---\n\n".join(documents[:5])

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
- COMPLETE THE ENTIRE JSON - do not truncate

Documents:
{docs_text}

Extract the knowledge graph as JSON:"""

    return prompt


def build_continuation_prompt(incomplete_json):
    """Build prompt to continue generating incomplete JSON."""
    prompt = f"""The previous response was incomplete. Here is the partial JSON output:

{incomplete_json}

CONTINUE generating the JSON from where it was cut off. Output ONLY the continuation needed to complete the valid JSON structure. Make sure to:
1. Close all open arrays and objects
2. Ensure all entity IDs in relations exist in the entities list
3. Return valid, parseable JSON

Continue the JSON:"""
    return prompt


def build_repair_prompt(bad_json_text, error_message):
    """Build prompt to repair invalid JSON with specific error context."""
    prompt = f"""The following JSON is invalid and produced this error:
ERROR: {error_message}

INVALID JSON:
{bad_json_text}

Fix this JSON and return ONLY valid, complete JSON with the same structure. Ensure:
1. All brackets and braces are properly closed
2. All strings are properly quoted
3. No trailing commas
4. All relation sources and targets reference valid entity IDs
5. The JSON is complete and parseable

Return the corrected JSON:"""
    return prompt


def call_llm(prompt, timeout=120):
    """Call LLM API based on configured provider."""

    if LLM_PROVIDER == "anthropic":
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text

    elif LLM_PROVIDER == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
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
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=4000
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "cerebras":
        client = Cerebras(api_key=os.getenv("CEREBRAS_API_KEY"))
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_completion_tokens=4000
        )
        return response.choices[0].message.content

    elif LLM_PROVIDER == "ollama":
        print(f"Calling Ollama API at {OLLAMA_BASE_URL}...")
        print(f"Using model: {OLLAMA_MODEL}")

        url = f"{OLLAMA_BASE_URL}/api/generate"

        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "temperature": 0.3,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json"
        }


        response = requests.post(
             url,
             json=payload,
             headers=headers,
             timeout=timeout
        )
        response.raise_for_status()

        result = response.json()

        if "response" in result:
            return result["response"]
        else:
            raise ValueError(f"Unexpected response format: {result}")



    else:
        raise ValueError(f"Unknown LLM provider: {LLM_PROVIDER}")


def is_json_truncated(json_text):
    """Check if JSON appears to be truncated/incomplete."""
    json_text = json_text.strip()

    # Check for common truncation indicators
    if not json_text:
        return True

    # Should end with closing brace
    if not json_text.endswith('}'):
        return True

    # Count braces
    open_braces = json_text.count('{')
    close_braces = json_text.count('}')
    open_brackets = json_text.count('[')
    close_brackets = json_text.count(']')

    if open_braces != close_braces or open_brackets != close_brackets:
        return True

    return False


def clean_json_text(json_text):
    """Clean JSON text by removing markdown code fences and extra whitespace."""
    # Remove markdown code fences
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        parts = json_text.split("```")
        if len(parts) >= 2:
            json_text = parts[1]

    return json_text.strip()


def parse_json_to_graph(json_text):
    """Parse LLM JSON output into NetworkX directed graph."""
    json_text = clean_json_text(json_text)

    data = json.loads(json_text)
    assert "entities" in data and "relations" in data, "JSON must contain 'entities' and 'relations' keys"

    G = nx.DiGraph()

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

    for relation in data["relations"]:
        source, target = relation["source"], relation["target"]

        # Validate that source and target exist
        if source not in entity_ids:
            print(f"Warning: Relation source '{source}' not found in entities, skipping relation")
            continue
        if target not in entity_ids:
            print(f"Warning: Relation target '{target}' not found in entities, skipping relation")
            continue

        G.add_edge(
            source, target,
            relation_type=relation["relation_type"],
            support_count=relation.get("support_count", 1),
            source_documents=relation.get("source_documents", []),
            confidence=relation.get("confidence", 1.0)
        )

    return G


def extract_with_retry(prompt, max_retries=MAX_RETRIES):
    """
    Call LLM with retry logic for incomplete or invalid JSON.

    Returns:
        tuple: (graph, success, attempt_count)
    """

    for attempt in range(1, max_retries + 1):
        print(f"\n{'─' * 80}")
        print(f"Attempt {attempt}/{max_retries}")
        print(f"{'─' * 80}")


        # Get LLM response
        if attempt == 1:
            current_prompt = prompt
            print("Sending initial extraction prompt...")
        else:
            print("Retrying with modified prompt...")

        response = call_llm(current_prompt)

        # Save raw response for debugging
        debug_file = f"data/raw_response_attempt_{attempt}.txt"
        os.makedirs("data", exist_ok=True)
        with open(debug_file, "w") as f:
            f.write(response)
        print(f"Raw response saved to {debug_file}")

        # Check if truncated
        if is_json_truncated(response):
            print("⚠ Response appears truncated/incomplete")

            if attempt < max_retries:
                print("Requesting continuation...")
                continuation_prompt = build_continuation_prompt(response)
                continuation = call_llm(continuation_prompt)

                # Try to merge responses
                response = response.rstrip() + continuation

                # Save merged response
                with open(f"data/merged_response_attempt_{attempt}.txt", "w") as f:
                    f.write(response)

        # Try to parse
        print("Parsing JSON...")
        G = parse_json_to_graph(response)

        # Success!
        print(f"✓ Successfully parsed on attempt {attempt}")
        print(f"  Entities: {len(G.nodes())}")
        print(f"  Relations: {len(G.edges())}")

        return G, True, attempt




    return None, False, max_retries


def visualize_graph(G, save_path="data/kg_visualization.png", model_name="Unknown", provider="Unknown"):
    """Visualize the knowledge graph with custom title."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=2, iterations=50)

    type_colors = {
        "PERSON": "#ff6b6b",
        "ORGANIZATION": "#4ecdc4",
        "LOCATION": "#45b7d1",
        "EVENT": "#ffa07a",
        "CONCEPT": "#98d8c8"
    }
    node_colors = [type_colors.get(G.nodes[node].get("type", "CONCEPT"), "#cccccc") for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#666666", arrows=True, width=1.5, alpha=0.6)
    labels = {node: G.nodes[node]["name"] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

    plt.title(f"Extracted Knowledge Graph\nModel: {model_name} ({provider})", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()

    os.makedirs("data", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\n✓ Graph visualization saved to {save_path}")
    plt.show()


def print_graph_details(G):
    """Print detailed information about the extracted graph."""
    print("\n" + "=" * 80)
    print("EXTRACTED KNOWLEDGE GRAPH")
    print("=" * 80)

    print(f"\nEntities: {len(G.nodes())}")
    print(f"Relations: {len(G.edges())}")

    print("\nENTITIES:")
    for node in G.nodes():
        attrs = G.nodes[node]
        doc_freq = attrs.get('document_frequency', 1)
        conf = attrs.get('confidence', 1.0)
        print(f"  • {attrs['name']} ({attrs['type']}) [docs: {doc_freq}, conf: {conf:.2f}]")

    print("\nRELATIONS:")
    for source, target in G.edges():
        attrs = G.edges[source, target]
        support = attrs.get('support_count', 1)
        conf = attrs.get('confidence', 1.0)
        print(f"  • {G.nodes[source]['name']} → {G.nodes[target]['name']}")
        print(f"    ({attrs['relation_type']}) [support: {support}, conf: {conf:.2f}]")

    print("=" * 80)


def main():
    """Run the complete extraction POC with retry logic."""

    # Model mapping for visualization
    model_mapping = {
        "anthropic": ANTHROPIC_MODEL,
        "openai": OPENAI_MODEL,
        "gemini": GEMINI_MODEL,
        "groq": GROQ_MODEL,
        "cerebras": CEREBRAS_MODEL,
        "ollama": OLLAMA_MODEL
    }
    current_model = model_mapping.get(LLM_PROVIDER, "Unknown Model")

    print(f"\n{'=' * 80}")
    print(f"KNOWLEDGE GRAPH EXTRACTION")
    print(f"Provider: {LLM_PROVIDER.upper()}")
    print(f"Model: {current_model}")
    print(f"Max Retries: {MAX_RETRIES}")
    print(f"{'=' * 80}\n")

    # Load data
    documents, _ = load_single_cluster(cluster_idx=0)

    # Build prompt
    prompt = build_extraction_prompt(documents)

    # Extract with retry logic
    try:
        G, success, attempts = extract_with_retry(prompt, max_retries=MAX_RETRIES)

        if success:
            print(f"\n{'=' * 80}")
            print(f"✓ EXTRACTION SUCCESSFUL (took {attempts} attempt(s))")
            print(f"{'=' * 80}")

            # Print details
            print_graph_details(G)

            # Visualize
            visualize_graph(G, model_name=current_model, provider=LLM_PROVIDER.upper())

            # Save
            pickle_path = "data/extracted_graph.pkl"
            with open(pickle_path, "wb") as f:
                pickle.dump(G, f)
            print(f"\n✓ Graph saved to {pickle_path}")

            return G
        else:
            print(f"\n{'=' * 80}")
            print(f"✗ EXTRACTION FAILED after {attempts} attempts")
            print(f"{'=' * 80}")
            return None

    except Exception as e:
        print(f"\n{'=' * 80}")
        print(f"✗ FATAL ERROR: {e}")
        print(f"{'=' * 80}")
        raise


if __name__ == "__main__":
    main()