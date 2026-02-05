"""
POC Step 3: End-to-End Pipeline

This script connects all pieces:
1. Load documents → Extract KG
2. Score KG with validator (random scores for now)
3. Simulate refinement decision (if score < threshold, would refine)
4. Generate summary from KG

This proves the complete data flow works before adding complexity.

Run: python poc_pipeline.py
"""

import os
import json
from dotenv import load_dotenv
from datasets import load_dataset
import networkx as nx
import torch
from anthropic import Anthropic
from google import genai
from groq import Groq
from openai import OpenAI

# Import from other POC scripts
import sys
sys.path.append(os.path.dirname(__file__))

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ============================================================================
# Step 1: Extraction (simplified from poc_extraction.py)
# ============================================================================

def load_cluster(cluster_idx=0):
    """Load one document cluster."""
    dataset = load_dataset("multi_news", split="test")
    cluster = dataset[cluster_idx]
    documents = cluster["document"].split("|||||")
    reference_summary = cluster["summary"]
    return documents, reference_summary


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


def extract_kg(documents):
    """Extract KG from documents via LLM."""
    docs_text = "\n\n---DOCUMENT---\n\n".join(documents[:5])
    
    prompt = f"""Extract a knowledge graph from these news articles.

Output ONLY valid JSON:
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
      "relation_type": "verb phrase",
      "support_count": 2,
      "confidence": 0.85
    }}
  ]
}}

Rules:
- Every relation source/target must reference an entity id
- Use past tense verbs for relations

Documents:
{docs_text}

JSON:"""
    
    # Call LLM
    json_text = call_llm(prompt)
    
    # Parse to graph
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]
    
    data = json.loads(json_text.strip())
    
    G = nx.DiGraph()
    
    for entity in data["entities"]:
        G.add_node(
            entity["id"],
            name=entity["name"],
            type=entity["type"],
            document_frequency=entity.get("document_frequency", 1),
            confidence=entity.get("confidence", 1.0)
        )
    
    for relation in data["relations"]:
        G.add_edge(
            relation["source"],
            relation["target"],
            relation_type=relation["relation_type"],
            support_count=relation.get("support_count", 1),
            confidence=relation.get("confidence", 1.0)
        )
    
    return G


# ============================================================================
# Step 2: Validator (mock for now)
# ============================================================================

def score_kg(G):
    """
    Score KG quality.
    
    For POC, we use simple heuristics instead of trained GNN:
    - consistency: high if well-connected, low if fragmented
    - missing_entities: high if low average degree
    - contradictions: always low (we can't detect this without training)
    - fragmentation: high if multiple components
    
    Returns: dict with 4 scores in [0, 1]
    """
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_weakly_connected_components(G)
    
    if num_nodes == 0:
        return {
            'consistency': 0.0,
            'missing_entities': 1.0,
            'contradictions': 0.0,
            'fragmentation': 1.0
        }
    
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0
    
    # Simple heuristics
    consistency = min(1.0, avg_degree / 3.0) * (1.0 - (num_components - 1) * 0.2)
    consistency = max(0.0, min(1.0, consistency))
    
    missing_entities = 1.0 - min(1.0, avg_degree / 2.0)
    fragmentation = min(1.0, (num_components - 1) * 0.3)
    contradictions = 0.1  # Mock value
    
    return {
        'consistency': consistency,
        'missing_entities': missing_entities,
        'contradictions': contradictions,
        'fragmentation': fragmentation
    }


# ============================================================================
# Step 3: Refinement Loop (simulation)
# ============================================================================

def should_refine(scores, threshold=0.7):
    """Decide if refinement is needed based on scores."""
    return scores['consistency'] < threshold


def simulate_refinement(G, scores, iteration):
    """
    Simulate refinement actions based on auxiliary scores.
    
    For POC, we just print what would happen. Real implementation
    would call LLM for retrieval expansion, contradiction resolution, etc.
    """
    print(f"\n  Refinement iteration {iteration}:")
    
    actions = []
    
    if scores['missing_entities'] > 0.5:
        actions.append("→ Expand retrieval with entity queries")
    
    if scores['contradictions'] > 0.5:
        actions.append("→ Retrieve corroborating sources")
    
    if scores['fragmentation'] > 0.5:
        actions.append("→ Rerank documents for topic cohesion")
    
    if not actions:
        actions.append("→ Generic re-extraction with refined prompt")
    
    for action in actions:
        print(f"    {action}")
    
    # In real implementation, this would return a revised KG
    # For POC, just return the same graph
    return G


# ============================================================================
# Step 4: Summary Generation
# ============================================================================

def generate_summary(G, documents):
    """
    Generate summary from KG + source documents.
    
    Uses hybrid prompt: KG structure + supporting text evidence.
    """
    # Build KG description
    kg_description = "Knowledge Graph:\n\nEntities:\n"
    for node in G.nodes():
        attrs = G.nodes[node]
        kg_description += f"  • {attrs['name']} ({attrs['type']})\n"
    
    kg_description += "\nRelations:\n"
    for u, v in G.edges():
        attrs = G.edges[u, v]
        u_name = G.nodes[u]['name']
        v_name = G.nodes[v]['name']
        kg_description += f"  • {u_name} {attrs['relation_type']} {v_name}\n"
    
    # Build prompt
    docs_sample = "\n\n".join(documents[:3])  # First 3 docs as evidence
    
    prompt = f"""Generate a concise summary of these news articles using the extracted knowledge graph.

{kg_description}

Source documents (for context):
{docs_sample[:2000]}...

Write a 3-4 sentence summary that:
- Covers the main events and entities from the knowledge graph
- Is factually grounded in the source documents
- Uses clear, journalistic style

Summary:"""
    
    # Call LLM
    summary = call_llm(prompt)
    
    return summary.strip()


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    """Run complete end-to-end pipeline."""
    print("\n" + "="*80)
    print("POC STEP 3: END-TO-END PIPELINE")
    print("="*80 + "\n")
    
    # Load data
    print("Loading document cluster...")
    documents, reference_summary = load_cluster(cluster_idx=1)  # Try cluster 1
    print(f"✓ Loaded {len(documents)} documents")
    print(f"  Reference summary length: {len(reference_summary)} chars\n")
    
    # Phase 1: Extraction
    print(f"{'='*80}")
    print("PHASE 1: EXTRACTION")
    print(f"{'='*80}\n")
    
    print("Extracting knowledge graph...")
    try:
        kg = extract_kg(documents)
        print(f"✓ Extracted KG: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges\n")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return
    
    # Phase 2: Scoring
    print(f"{'='*80}")
    print("PHASE 2: VALIDATION")
    print(f"{'='*80}\n")
    
    print("Scoring KG quality...")
    scores = score_kg(kg)
    print("✓ KG scores:")
    for key, value in scores.items():
        print(f"    {key}: {value:.4f}")
    
    # Phase 3: Refinement Decision
    print(f"\n{'='*80}")
    print("PHASE 3: REFINEMENT LOOP")
    print(f"{'='*80}\n")
    
    threshold = 0.7
    max_iterations = 3
    
    print(f"Consistency threshold: {threshold}")
    print(f"Current consistency score: {scores['consistency']:.4f}")
    
    if should_refine(scores, threshold):
        print(f"\n⚠ Score below threshold. Refinement needed.")
        
        for i in range(1, max_iterations + 1):
            kg = simulate_refinement(kg, scores, i)
            scores = score_kg(kg)
            
            print(f"    New consistency score: {scores['consistency']:.4f}")
            
            if not should_refine(scores, threshold):
                print(f"\n✓ Threshold reached after {i} iteration(s)")
                break
        else:
            print(f"\n⚠ Max iterations reached. Proceeding with current KG.")
    else:
        print(f"\n✓ Score above threshold. No refinement needed.")
    
    # Phase 4: Generation
    print(f"\n{'='*80}")
    print("PHASE 4: SUMMARY GENERATION")
    print(f"{'='*80}\n")
    
    print("Generating summary from refined KG...")
    summary = generate_summary(kg, documents)
    
    print("\n" + "-"*80)
    print("GENERATED SUMMARY:")
    print("-"*80)
    print(summary)
    print("-"*80)
    
    print("\n" + "-"*80)
    print("REFERENCE SUMMARY:")
    print("-"*80)
    print(reference_summary[:500] + "..." if len(reference_summary) > 500 else reference_summary)
    print("-"*80)
    
    # Save results
    results = {
        'cluster_idx': 1,
        'num_documents': len(documents),
        'kg_stats': {
            'nodes': kg.number_of_nodes(),
            'edges': kg.number_of_edges(),
            'components': nx.number_weakly_connected_components(kg)
        },
        'scores': scores,
        'generated_summary': summary,
        'reference_summary': reference_summary
    }
    
    os.makedirs("data", exist_ok=True)
    with open("data/pipeline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to data/pipeline_results.json")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ POC STEP 3 COMPLETE")
    print(f"{'='*80}\n")
    
    print("Pipeline summary:")
    print(f"  • Extracted {kg.number_of_nodes()} entities, {kg.number_of_edges()} relations")
    print(f"  • Final consistency score: {scores['consistency']:.4f}")
    print(f"  • Generated {len(summary)} character summary")
    print(f"  • Reference summary: {len(reference_summary)} characters")
    
    print("\nNext steps:")
    print("  1. Generate training data (200-300 clean KGs + corrupted variants)")
    print("  2. Train the GNN validator on real corruption patterns")
    print("  3. Implement actual refinement actions (not just simulation)")
    print("  4. Run full evaluation on 100 test clusters")


if __name__ == "__main__":
    main()
