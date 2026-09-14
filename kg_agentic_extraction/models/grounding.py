"""
The grounder's output contract — the refined graph mapped onto DBpedia.

Grounding is additive: a `GroundedKnowledgeGraph` wraps the original
`KnowledgeGraph` untouched and carries mappings alongside it, rather than
rewriting entity names in place. That way a failed or low-confidence mapping
degrades to "ungrounded" instead of corrupting the graph the grader already
signed off on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph


class OntologyMapping(BaseModel):
    """One resolved link from a local graph element to a DBpedia resource."""

    uri: str = Field(..., description="Full DBpedia URI, e.g. http://dbpedia.org/resource/Seattle")
    label: str = Field("", description="rdfs:label of the resolved resource.")
    ontology_class: str | None = Field(
        None, description="dbo: class, e.g. http://dbpedia.org/ontology/City"
    )
    abstract: str | None = Field(None, description="Short dbo:abstract, when retrieved.")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Mapping confidence.")

    model_config = {"frozen": True}


class GroundedEntity(BaseModel):
    """An entity id paired with its DBpedia mapping (or the reason there isn't one)."""

    entity_id: str
    mapping: OntologyMapping | None = None
    candidates: list[OntologyMapping] = Field(
        default_factory=list,
        description="Runner-up matches, kept so a later disambiguation pass can revisit.",
    )
    unresolved_reason: str | None = Field(
        None, description="Why grounding failed, when mapping is None."
    )

    @property
    def is_grounded(self) -> bool:
        return self.mapping is not None


class GroundedRelation(BaseModel):
    """A relation key paired with the DBpedia property it maps to."""

    relation_key: str = Field(..., description="'source|RELATION_TYPE|target'.")
    property_uri: str | None = Field(None, description="e.g. http://dbpedia.org/ontology/location")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    unresolved_reason: str | None = None

    @property
    def is_grounded(self) -> bool:
        return self.property_uri is not None


class GroundedKnowledgeGraph(BaseModel):
    """The refined graph plus its DBpedia alignment."""

    graph: KnowledgeGraph
    entities: list[GroundedEntity] = Field(default_factory=list)
    relations: list[GroundedRelation] = Field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Fraction of entities successfully mapped — the headline quality number."""
        if not self.entities:
            return 0.0
        return sum(e.is_grounded for e in self.entities) / len(self.entities)
