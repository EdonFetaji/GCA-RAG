"""
The grounding port.

The grounder agent depends on this Protocol, not on MCP. That boundary is what
lets the agent be tested against an in-memory fake, and what would let DBpedia
be swapped for Wikidata by writing one adapter — the agent's disambiguation
logic does not change either way.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from kg_agentic_extraction.models.grounding import OntologyMapping


@runtime_checkable
class GroundingBackend(Protocol):
    """A source of candidate ontology resources for a surface form."""

    def lookup_entity(
        self,
        name: str,
        *,
        entity_type: str | None = None,
        limit: int = 5,
    ) -> list[OntologyMapping]:
        """
        Candidate resources for an entity name, best first.

        `entity_type` is the *local* type (PERSON, ORGANIZATION, …); backends
        may use it to filter but must not require it. Returns `[]` when nothing
        matches — absence of a match is a normal outcome, not an error.
        """
        ...

    def resolve_property(self, relation_type: str) -> str | None:
        """The ontology property URI for a local relation type, or None."""
        ...

    def close(self) -> None:
        """Release any held connection. Safe to call more than once."""
        ...


class GroundingError(RuntimeError):
    """The backend could not be reached or returned something unusable."""


class NullGroundingBackend:
    """
    A backend that resolves nothing.

    Used when `grounding_enabled` is false and as the default in tests, so that
    the grounder node has a valid collaborator instead of a None to guard
    against at every call site.
    """

    def lookup_entity(
        self,
        name: str,
        *,
        entity_type: str | None = None,
        limit: int = 5,
    ) -> list[OntologyMapping]:
        return []

    def resolve_property(self, relation_type: str) -> str | None:
        return None

    def close(self) -> None:
        return None
