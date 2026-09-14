"""
The grounding port.

The grounder agent depends on this Protocol, not on MCP. That boundary is what
lets the agent be tested against an in-memory fake, and what would let DBpedia
be swapped for Wikidata by writing one adapter — the agent's disambiguation
logic does not change either way.

The Protocol mirrors the server's tools one-to-one. Each method returns the
tool's payload as a plain dict rather than a mirrored Pydantic model: the
server already declares those schemas, and re-declaring them here would give
two definitions of one contract, free to drift. The shapes are documented at
`mcp_server/dbpedia/models.py`, and `tool_specs()` carries the machine-readable
version straight from the running server.

Two members are not tools:

- `tool_specs()` — what the agent hands to the model's tool binding.
- `dispatch()`   — routes a tool call the *model* chose, by name, and returns
  JSON text. The agent never has to know the tool list at compile time, which
  is what keeps adding a ninth tool a server-side change only.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from kg_agentic_extraction.llm.base import ToolSpec
from kg_agentic_extraction.models.grounding import OntologyMapping


@runtime_checkable
class GroundingBackend(Protocol):
    """The DBpedia grounding toolkit, as the grounder agent sees it."""

    # ── Entity tools ──────────────────────────────────────────────────

    def spotlight_link(self, mention: str, context: str) -> dict[str, Any]:
        """Context-aware candidates for a mention. `{mention, candidates[], note}`."""
        ...

    def search_resource(
        self, label: str, expected_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Resources matching a name. `{label, expected_types[], results[], note}`."""
        ...

    def get_resource_profile(self, uri: str) -> dict[str, Any]:
        """Abstract, types and redirect/disambiguation flags for one resource."""
        ...

    def search_class(self, label: str) -> dict[str, Any]:
        """Ontology classes matching a name. `{label, results[], note}`."""
        ...

    # ── Relation tools ────────────────────────────────────────────────

    def find_object_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,
        object_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Object properties matching a relation phrase. `{relation_text, results[], note}`."""
        ...

    def find_datatype_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,
        literal_datatype: str | None = None,
    ) -> dict[str, Any]:
        """Datatype properties matching a relation phrase."""
        ...

    def get_property_profile(self, property_uri: str) -> dict[str, Any]:
        """Meaning, domain, range and usage count for one ontology property."""
        ...

    def get_predicates_between(self, subject_uri: str, object_uri: str) -> dict[str, Any]:
        """Predicates DBpedia asserts between two resources. `{predicates[], note}`."""
        ...

    # ── Agent plumbing ────────────────────────────────────────────────

    def tool_specs(self) -> list[ToolSpec]:
        """The tools to bind to the model, as published by the server."""
        ...

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Run a tool the model asked for and return its payload as JSON text.

        Must not raise: whatever goes wrong — an unknown tool name, a dead
        server — comes back as a JSON object with an `error` key. A raise here
        aborts the agent's loop mid-graph, where an error the model can read
        lets it try something else.
        """
        ...

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __enter__(self) -> GroundingBackend:
        """Open the connection for a grounding pass."""
        ...

    def __exit__(self, *exc_info: object) -> None: ...

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

    _EMPTY_NOTE = "grounding is disabled; no DBpedia backend is connected"

    # ── Entity tools ──────────────────────────────────────────────────

    def spotlight_link(self, mention: str, context: str) -> dict[str, Any]:  # noqa: ARG002
        return {"mention": mention, "candidates": [], "note": self._EMPTY_NOTE}

    def search_resource(
        self,
        label: str,
        expected_types: list[str] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {"label": label, "expected_types": [], "results": [], "note": self._EMPTY_NOTE}

    def get_resource_profile(self, uri: str) -> dict[str, Any]:
        return {"uri": uri, "found": False, "note": self._EMPTY_NOTE}

    def search_class(self, label: str) -> dict[str, Any]:
        return {"label": label, "results": [], "note": self._EMPTY_NOTE}

    # ── Relation tools ────────────────────────────────────────────────

    def find_object_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,  # noqa: ARG002
        object_types: list[str] | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {"relation_text": relation_text, "results": [], "note": self._EMPTY_NOTE}

    def find_datatype_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,  # noqa: ARG002
        literal_datatype: str | None = None,  # noqa: ARG002
    ) -> dict[str, Any]:
        return {"relation_text": relation_text, "results": [], "note": self._EMPTY_NOTE}

    def get_property_profile(self, property_uri: str) -> dict[str, Any]:
        return {"uri": property_uri, "found": False, "note": self._EMPTY_NOTE}

    def get_predicates_between(self, subject_uri: str, object_uri: str) -> dict[str, Any]:
        return {
            "subject_uri": subject_uri,
            "object_uri": object_uri,
            "predicates": [],
            "note": self._EMPTY_NOTE,
        }

    # ── Agent plumbing ────────────────────────────────────────────────

    def tool_specs(self) -> list[ToolSpec]:
        """No tools. The agent sees this and falls back to a plain, toolless call."""
        return []

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:  # noqa: ARG002
        return f'{{"error": "{self._EMPTY_NOTE}", "tool": "{name}"}}'

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __enter__(self) -> NullGroundingBackend:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def close(self) -> None:
        return None


def mapping_from_hit(hit: dict[str, Any], *, confidence: float = 0.5) -> OntologyMapping | None:
    """
    Turn one tool result row into an `OntologyMapping`.

    Shared by anything that needs to record a candidate the model saw but did
    not choose. Rows from `search_resource` and `spotlight_link` differ in their
    scoring fields but agree on `uri`, `label`, and `types`, which is all a
    mapping carries. Returns None for a row with no URI, so callers can map
    over a result list without filtering first.
    """
    uri = str(hit.get("uri") or "").strip()
    if not uri:
        return None
    types = hit.get("types") or []
    return OntologyMapping(
        uri=uri,
        label=str(hit.get("label") or ""),
        ontology_class=str(types[0]) if types else None,
        abstract=str(hit.get("comment") or "") or None,
        confidence=confidence,
    )
