import json
from typing import Tuple, Dict, List
from constants import VALID_ENTITY_TYPES

def validate_schema(entities: Dict) -> Tuple[bool, Dict, List]:
    """
    Validate entities against the schema.
    
    Args:
        entities (dict): Dictionary with entity types as keys and lists of entities
        
    Returns:
        tuple: (is_valid, cleaned_entities, errors)
            - is_valid (bool): Whether all entities are valid
            - cleaned_entities (dict): Validated entities with invalid ones removed
            - errors (list): List of validation errors
    """
    errors = []
    cleaned = {}
    
    # Check if input is a dictionary
    if not isinstance(entities, dict):
        errors.append("Entities must be a dictionary")
        return False, {}, errors
    
    # Validate each entity type
    for entity_type, entity_list in entities.items():
        # Check if type is valid
        if entity_type not in VALID_ENTITY_TYPES:
            errors.append(f"Invalid entity type: {entity_type}")
            continue
        
        # Check if value is a list
        if not isinstance(entity_list, list):
            errors.append(f"Entity type '{entity_type}' must have a list value, got {type(entity_list).__name__}")
            continue
        
        # Validate individual entities
        valid_entities = []
        for entity in entity_list:
            if not isinstance(entity, str):
                errors.append(f"Entity in '{entity_type}' must be string, got {type(entity).__name__}")
                continue
            
            if not entity or not entity.strip():
                errors.append(f"Empty entity string in '{entity_type}'")
                continue
            
            valid_entities.append(entity)
        
        cleaned[entity_type] = valid_entities
    
    # Ensure all valid types are present
    for entity_type in VALID_ENTITY_TYPES:
        if entity_type not in cleaned:
            cleaned[entity_type] = []
    
    is_valid = len(errors) == 0
    return is_valid, cleaned, errors
