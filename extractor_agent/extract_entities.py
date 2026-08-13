import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from extractor_agent.constants import VALID_ENTITY_TYPES
from extractor_agent.json_utils import parse_json_with_repair

load_dotenv()

# Cerebras periodically retires model IDs from its public endpoints (e.g. the
# llama-3.1-8b/70b models this used to hardcode are gone as of mid-2026), so
# read it from the environment with a current, production-tier default
# instead of hardcoding a name that can 404 without warning.
CEREBRAS_MODEL = os.getenv("CEREBRAS_MODEL", "gpt-oss-120b")

def extract_entities(document: str) -> dict:
    """
    Extract entities from document using a Cerebras-hosted model.
    
    Args:
        document (str): The document text to extract entities from
        
    Returns:
        dict: Dictionary with entity types as keys and lists of entities as values
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    client = Cerebras(api_key=api_key)

    # Build the schema from VALID_ENTITY_TYPES (imported from
    # extraction.schemas.EntityType) rather than a hardcoded literal, so
    # this prompt can't silently drift out of sync with the ontology again.
    schema_json = "{" + ", ".join(f'"{t}": []' for t in sorted(VALID_ENTITY_TYPES)) + "}"

    prompt = f"""
Extract all entities explicitly mentioned in the document.

Return ONLY this JSON schema:
{schema_json}

Document:
{document}
"""

    def _call(p: str) -> str:
        response = client.chat.completions.create(
            model=CEREBRAS_MODEL,
            temperature=0.0,
            top_p=1.0,
            max_tokens=1024,
            messages=[
                {
                    "role": "system",
                    "content": "You are a deterministic knowledge graph entity extractor. Output strictly valid JSON. No explanations."
                },
                {
                    "role": "user",
                    "content": p
                }
            ]
        )
        return response.choices[0].message.content.strip()

    entities = parse_json_with_repair(_call, prompt)
    
    # Validate schema
    if not isinstance(entities, dict):
        raise ValueError("Output is not a dictionary.")
    if set(entities.keys()) != VALID_ENTITY_TYPES:
        raise ValueError("Unexpected schema keys.")
    for key in VALID_ENTITY_TYPES:
        if not isinstance(entities[key], list):
            raise ValueError(f"{key} must be a list.")

    return entities
