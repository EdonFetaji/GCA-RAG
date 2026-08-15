"""
The knowledge-graph contract passed between agents.

These models are the interface the extractor produces, the grader consumes,
and the grounder enriches. They carry no behaviour beyond validation and a
couple of derived lookups — deliberately, so that agents can be swapped
without the data contract moving.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, model_validator

from kg_agentic_extraction.models.ontology import EntityType, RelationType


class EvidenceSpan(BaseModel):
    """A verbatim quote from a source document that supports a fact."""

    document_index: int = Field(..., ge=0, description="0-based index of the source document.")
    quote: str = Field(..., min_length=1, description="Exact sentence copied from the document.")

    model_config = {"frozen": True}


class Entity(BaseModel):
    """An entity discovered in the source documents"""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    type: EntityType
    document_frequency: int = Field(1, ge=1, description="How many documents mention it.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class Relation(BaseModel):
    """A directed edge between two entities, referenced by entity id."""

    source: str = Field(..., min_length=1, description="Source entity id.")
    target: str = Field(..., min_length=1, description="Target entity id.")
    relation_type: RelationType
    support_count: int = Field(1, ge=1, description="How many documents support it.")
    source_documents: list[int] = Field(default_factory=list) # not sure if we want this
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identity used when de-duplicating or merging edges."""
        return f"{self.source}|{self.relation_type.value}|{self.target}"


class KnowledgeGraph(BaseModel):
    """
    A complete extracted graph.

    Referential integrity (every relation endpoint resolving to a real entity)
    is enforced here rather than left to the grader, so a structurally broken
    graph can never reach the grader in the first place — the grader's job is
    semantic faithfulness to the documents, not structural repair.
    """

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)

    @classmethod
    def from_json(cls, data: str) -> KnowledgeGraph:
        """Construct a KnowledgeGraph from a JSON-like dict."""
        data_dict = json.loads(data)
        return cls(entities=data_dict.get("entities", []), relations=data_dict.get("relations", []))

    @model_validator(mode="after")
    def _relations_reference_known_entities(self) -> KnowledgeGraph:
        known = {e.id for e in self.entities}
        dangling = [
            f"{r.source}->{r.target}"
            for r in self.relations
            if r.source not in known or r.target not in known
        ]
        if dangling:
            raise ValueError(
                f"{len(dangling)} relation(s) reference unknown entity ids: {dangling[:5]}"
            )
        return self

    @property
    def entity_index(self) -> dict[str, Entity]:
        return {e.id: e for e in self.entities}

    def __len__(self) -> int:
        return len(self.entities)
