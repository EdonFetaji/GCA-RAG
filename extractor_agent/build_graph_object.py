import json
from typing import Dict, List

def build_graph_object(entities: Dict, relations: List) -> Dict:
    """
    Build a knowledge graph object from entities and relations.
    
    Args:
        entities (Dict): Dictionary with entity types as keys and lists of entities
        relations (List): List of relation objects with format: {"subject": str, "relation": str, "object": str}
        
    Returns:
        Dict: Knowledge graph object with nodes and edges
    """
    # Build nodes from entities
    nodes = []
    entity_to_type = {}
    
    for entity_type, entity_list in entities.items():
        for entity in entity_list:
            if entity:  # Skip empty entities
                node = {
                    "id": entity,
                    "label": entity,
                    "type": entity_type
                }
                nodes.append(node)
                entity_to_type[entity] = entity_type
    
    # Build edges from relations
    edges = []
    for idx, rel in enumerate(relations):
        if not isinstance(rel, dict):
            continue
        
        subject = rel.get("subject")
        relation = rel.get("relation")
        obj = rel.get("object")
        
        # Validate relation has required fields
        if not subject or not relation or not obj:
            continue
        
        # Validate subject and object are in entities
        if subject not in entity_to_type or obj not in entity_to_type:
            continue
        
        edge = {
            "id": f"rel_{idx}",
            "source": subject,
            "target": obj,
            "label": relation,
            "type": relation
        }
        edges.append(edge)
    
    # Build graph object
    graph = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "entity_types": list(entities.keys())
        }
    }
    
    return graph
