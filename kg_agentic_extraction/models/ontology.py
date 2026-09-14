"""
The controlled vocabulary the grounder maps onto.

Not a constraint on extraction. Since ADR 0004 the extractor names its own
entity and relation types and `Entity.type` is a plain `str`; these enums are
read by the grounder, which uses them to build its DBpedia hint tables, and by
`kg_dataset/corruption.py` and `extractor_agent/constants.py`.

Adding a member here therefore widens what the grounder knows a first guess for.
It does not widen — and never restricted — what the extractor may emit.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EntityType(StrEnum):
    """Allowed node types. Tuned for Multi-News (general news reporting)."""

    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"


class RelationType(StrEnum):
    """Allowed edge types."""

    LOCATED_IN = "LOCATED_IN"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    ANNOUNCED = "ANNOUNCED"
    ACQUIRED = "ACQUIRED"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    RESULTED_IN = "RESULTED_IN"
    CAUSES = "CAUSES"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    RELATED_TO = "RELATED_TO"


class OntologyConfig(BaseModel):
    """
    The ontology as handed to the grounder for a single run.

    Defaults to the full enums, but callers may narrow the lists to constrain a
    particular run without touching the enums themselves — which is why the
    grounder takes an `OntologyConfig` rather than reading the enums directly.

    `domain_context` is the exception: it is the one field the extractor still
    receives, passed as a plain string, and it steers subject matter rather
    than types.
    """

    entity_types: list[EntityType] = Field(default_factory=lambda: list(EntityType))
    relation_types: list[RelationType] = Field(default_factory=lambda: list(RelationType))
    domain_context: str = Field(
        default="general news articles",
        description="Short domain description used to focus extraction.",
    )

    @property
    def entity_type_names(self) -> list[str]:
        """Plain strings, for rendering into prompt templates."""
        return [t.value for t in self.entity_types]

    @property
    def relation_type_names(self) -> list[str]:
        return [r.value for r in self.relation_types]


VALID_ENTITY_TYPES: set[str] = {t.value for t in EntityType}
VALID_RELATION_TYPES: set[str] = {r.value for r in RelationType}
