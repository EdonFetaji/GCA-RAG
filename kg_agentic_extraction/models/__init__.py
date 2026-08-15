"""
models — the Pydantic contracts every agent speaks.

Nothing here imports an agent, an LLM client, or LangGraph. The dependency
arrow points one way: agents depend on models, never the reverse.
"""

from kg_agentic_extraction.models.grading import (
    GraderIssue,
    GraderReport,
    IssueType,
    Severity,
)
from kg_agentic_extraction.models.grounding import (
    GroundedEntity,
    GroundedKnowledgeGraph,
    GroundedRelation,
    OntologyMapping,
)
from kg_agentic_extraction.models.knowledge_graph import (
    Entity,
    EvidenceSpan,
    KnowledgeGraph,
    Relation,
)
from kg_agentic_extraction.models.ontology import (
    VALID_ENTITY_TYPES,
    VALID_RELATION_TYPES,
    EntityType,
    OntologyConfig,
    RelationType,
)

__all__ = [
    "VALID_ENTITY_TYPES",
    "VALID_RELATION_TYPES",
    "Entity",
    "EntityType",
    "EvidenceSpan",
    "GraderIssue",
    "GraderReport",
    "GroundedEntity",
    "GroundedKnowledgeGraph",
    "GroundedRelation",
    "IssueType",
    "KnowledgeGraph",
    "OntologyConfig",
    "OntologyMapping",
    "Relation",
    "RelationType",
    "Severity",
]
