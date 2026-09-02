"""
The knowledge-graph contract passed between agents.

These models are the interface the extractor produces, the grader consumes,
and the grounder enriches. They carry no behaviour beyond validation and a
couple of derived lookups — deliberately, so that agents can be swapped
without the data contract moving.
"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, field_validator, model_validator

#: Anything that is not a letter or a digit becomes a separator, so that
#: "located in", "located-in" and "Located In" are one type rather than three.
_SEPARATORS = re.compile(r"[^A-Za-z0-9]+")


def normalize_type(raw: str) -> str:
    """
    Canonical spelling of a free-form type label.

    Types are no longer drawn from a closed enum — the extractor invents them
    from the documents — so the one thing still enforced is *spelling*. Without
    it the same relation arrives as `LOCATED_IN`, `located in` and `Located-In`,
    which fragments `Relation.key`, breaks de-duplication, and hands the
    grounder three lookups for one concept.

    Normalising rather than rejecting is the point: an unknown type is a
    legitimate finding here, an inconsistently spelled one never is.
    """
    return _SEPARATORS.sub("_", raw.strip()).strip("_").upper()


class EvidenceSpan(BaseModel):
    """A verbatim quote from a source document that supports a fact."""

    document_index: int = Field(..., ge=0, description="0-based index of the source document.")
    quote: str = Field(..., min_length=1, description="Exact sentence copied from the document.")

    model_config = {"frozen": True}


class Entity(BaseModel):
    """An entity discovered in the source documents"""

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    type: str = Field(
        ...,
        min_length=1,
        description=(
            "Entity type, in the extractor's own vocabulary. Not checked against an "
            "ontology — that is the grounder's job, if grounding runs at all."
        ),
    )
    document_frequency: int = Field(1, ge=1, description="How many documents mention it.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @field_validator("type", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return normalize_type(value) if isinstance(value, str) else value


class Relation(BaseModel):
    """A directed edge between two entities, referenced by entity id."""

    source: str = Field(..., min_length=1, description="Source entity id.")
    target: str = Field(..., min_length=1, description="Target entity id.")
    relation_type: str = Field(
        ...,
        min_length=1,
        description=(
            "Relation type, in the extractor's own vocabulary. Not checked against an "
            "ontology — that is the grounder's job, if grounding runs at all."
        ),
    )
    support_count: int = Field(1, ge=1, description="How many documents support it.")
    source_documents: list[int] = Field(default_factory=list)  # not sure if we want this
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @field_validator("relation_type", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> object:
        return normalize_type(value) if isinstance(value, str) else value

    @property
    def key(self) -> str:
        """Stable identity used when de-duplicating or merging edges."""
        return f"{self.source}|{self.relation_type}|{self.target}"


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

    @property
    def entity_type_vocabulary(self) -> list[str]:
        """
        The entity types this graph actually uses, sorted.

        With extraction unconstrained, the vocabulary is discovered rather than
        declared — so downstream consumers (the grounder building its hint
        table, an analyst counting type drift across clusters) have to read it
        off the graph instead of off `OntologyConfig`.
        """
        return sorted({e.type for e in self.entities})

    @property
    def relation_type_vocabulary(self) -> list[str]:
        """The relation types this graph actually uses, sorted."""
        return sorted({r.relation_type for r in self.relations})

    def __len__(self) -> int:
        return len(self.entities)
