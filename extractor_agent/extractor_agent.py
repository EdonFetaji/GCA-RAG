from extractor_agent.extract_entities import extract_entities
from extractor_agent.normalize_entities import normalize_entities
from extractor_agent.extract_relations import extract_relations
from extractor_agent.validate_schema import validate_schema
from extractor_agent.build_graph_object import build_graph_object

def run_extractor_pipeline(document: str) -> dict:
    """
    Run the complete knowledge graph extraction pipeline.
    
    Args:
        document (str): The document text to extract knowledge from
        
    Returns:
        dict: Knowledge graph object with nodes and edges
    """
    # Step 1: Extract entities from document
    print("Step 1: Extracting entities...")
    raw_entities = extract_entities(document)
    
    # Step 2: Validate entities against schema
    print("Step 2: Validating schema...")
    is_valid, validated_entities, errors = validate_schema(raw_entities)
    
    if errors:
        print(f"Validation errors: {errors}")
    
    # Step 3: Normalize entities (deduplicate and clean)
    print("Step 3: Normalizing entities...")
    normalized_entities = normalize_entities(validated_entities)
    
    # Step 4: Extract relations between entities
    print("Step 4: Extracting relations...")
    relations = extract_relations(document, normalized_entities)
    
    # Step 5: Build knowledge graph object
    print("Step 5: Building knowledge graph...")
    knowledge_graph = build_graph_object(normalized_entities, relations)
    
    return knowledge_graph
