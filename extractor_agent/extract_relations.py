import os
import json
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras

# Load environment variables from .env file
load_dotenv()

def extract_relations(document: str, entities: dict) -> list:
    """
    Extract relations between entities from a document using Cerebras Llama 3.1 model.
    
    Args:
        document (str): The original document text
        entities (dict): Dictionary of extracted entities with types as keys
        
    Returns:
        list: List of relation objects with format: {"subject": str, "relation": str, "object": str}
    """
    # Initialize Cerebras client
    api_key = os.getenv("CEREBRAS_API_KEY")
    client = Cerebras(api_key=api_key)
    
    # Format entities for the prompt
    entities_text = json.dumps(entities, indent=2)
    
    # Prompt for relation extraction
    prompt = f"""You are a knowledge graph extractor. You must follow the schema strictly.
You must not infer unstated relations.
You must extract ONLY relations that are explicitly mentioned in the document.
If uncertain, omit.

Extract all relations between entities from the following document. Return ONLY a JSON array of relation objects. Each object must have "subject", "relation", and "object" fields. All values must be from the provided entities list.

Document:
{document}

Entities:
{entities_text}

Response format:
[{{"subject": "entity", "relation": "relation_type", "object": "entity"}}]

Return only the JSON array, no additional text."""
    
    # Call Cerebras model
    response = client.chat.completions.create(
        model="llama-3.1-8b",
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    # Extract and parse the response
    response_text = response.choices[0].message.content.strip()
    relations = json.loads(response_text)
    
    return relations
