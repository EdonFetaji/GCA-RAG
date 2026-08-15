"""
The ontology every agent in the pipeline is constrained by.

This module is the single source of truth for what an entity or a relation is
allowed to be, repo-wide. `validator/corruption.py` and
`extractor_agent/constants.py` import from here.

Extending the ontology means adding a member to one of the enums below — no
prompt edits required, because the templates render `OntologyConfig` at
runtime rather than hardcoding the type lists.
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
    The ontology as handed to an agent for a single run.

    Defaults to the full enums, but callers may narrow the lists to constrain a
    particular run without touching the enums themselves — which is why agents
    take an `OntologyConfig` rather than reading the enums directly.
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
