import unicodedata
import re
from typing import Dict, List
from constants import VALID_ENTITY_TYPES

def normalize_text(text: str) -> str:
    text = text.strip()

    # Remove accents properly
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )

    text = text.lower()

    # Keep hyphens
    text = re.sub(r'[^\w\s\-]', '', text)

    text = re.sub(r'\s+', ' ', text)

    return text

def normalize_entities(entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}

    for entity_type, entity_list in entities.items():
        if entity_type not in VALID_ENTITY_TYPES:
            continue

        seen = {}
        
        for entity in entity_list:
            if not entity or not isinstance(entity, str):
                continue

            normalized_name = normalize_text(entity)

            if not normalized_name:
                continue

            # Keep longest surface form
            if normalized_name not in seen:
                seen[normalized_name] = entity
            else:
                if len(entity) > len(seen[normalized_name]):
                    seen[normalized_name] = entity

        normalized[entity_type] = list(seen.values())

    return normalized
