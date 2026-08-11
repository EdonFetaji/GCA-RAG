import os
import json
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from constants import VALID_ENTITY_TYPES

load_dotenv()

def safe_parse_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != -1:
            return json.loads(text[start:end])
        raise

def extract_entities(document: str) -> dict:
    """
    Extract entities from document using Cerebras Llama 3.1 model.
    
    Args:
        document (str): The document text to extract entities from
        
    Returns:
        dict: Dictionary with entity types as keys and lists of entities as values
    """
    api_key = os.getenv("CEREBRAS_API_KEY")
    client = Cerebras(api_key=api_key)
    
    prompt = f"""
Extract all entities explicitly mentioned in the document.

Return ONLY this JSON schema:
{{"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "PRODUCT": [], "OTHER": []}}

Document:
{document}
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b",
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
                "content": prompt
            }
        ]
    )

    response_text = response.choices[0].message.content.strip()
    entities = safe_parse_json(response_text)
    
    # Validate schema
    if not isinstance(entities, dict):
        raise ValueError("Output is not a dictionary.")
    if set(entities.keys()) != VALID_ENTITY_TYPES:
        raise ValueError("Unexpected schema keys.")
    for key in VALID_ENTITY_TYPES:
        if not isinstance(entities[key], list):
            raise ValueError(f"{key} must be a list.")

    return entities
