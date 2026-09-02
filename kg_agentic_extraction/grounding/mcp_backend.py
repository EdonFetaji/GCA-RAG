"""
`GroundingBackend` implemented over the standalone DBpedia MCP server.

This is the only place in the pipeline that speaks MCP. The server itself lives
in `mcp_server/` and is never imported from here — the two sides are coupled by
the protocol and the tool names below, nothing else, so the server can be
restarted, replaced, or moved to another host independently.

The MCP client API is async; the `GroundingBackend` port is sync because the
agents and LangGraph nodes are. `SyncMCPClient` does that bridging, holding one
session open for a whole grounding pass rather than opening one per call.

Everything here is a thin pass-through. There is no candidate ranking, no
scoring, and no retry policy: those belong to the server (which owns the
endpoints) or to the model (which owns the judgement). Adding logic here would
put a third opinion between the two.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kg_agentic_extraction.grounding.base import GroundingError
from kg_agentic_extraction.llm.base import ToolSpec

logger = logging.getLogger(__name__)

# Tool names as published by mcp_server/tools.py. Kept as constants so a rename
# on the server side fails loudly here rather than silently returning nothing.
TOOL_SPOTLIGHT_LINK = "spotlight_link"
TOOL_SEARCH_RESOURCE = "search_resource"
TOOL_GET_RESOURCE_PROFILE = "get_resource_profile"
TOOL_SEARCH_CLASS = "search_class"
TOOL_FIND_OBJECT_PROPERTIES = "find_object_properties"
TOOL_FIND_DATATYPE_PROPERTIES = "find_datatype_properties"
TOOL_GET_PROPERTY_PROFILE = "get_property_profile"
TOOL_GET_PREDICATES_BETWEEN = "get_predicates_between"

ENTITY_TOOLS = (
    TOOL_SPOTLIGHT_LINK,
    TOOL_SEARCH_RESOURCE,
    TOOL_GET_RESOURCE_PROFILE,
    TOOL_SEARCH_CLASS,
)
RELATION_TOOLS = (
    TOOL_FIND_OBJECT_PROPERTIES,
    TOOL_FIND_DATATYPE_PROPERTIES,
    TOOL_GET_PROPERTY_PROFILE,
    TOOL_GET_PREDICATES_BETWEEN,
)
ALL_TOOLS = ENTITY_TOOLS + RELATION_TOOLS


class MCPGroundingBackend:
    """
    Talks to the DBpedia MCP server.

    `url` is normally the streamable-HTTP endpoint of a running server, but the
    underlying client also accepts a server *instance*, which is how this
    adapter gets tested end-to-end over a real protocol session without binding
    a port.

    Use it as a context manager. The session opens on entry and closes on exit,
    so it spans one grounding pass rather than the whole pipeline run — a
    session held across the extractor↔grader loop, which can take minutes, would
    have to survive the server timing it out.
    """

    def __init__(self, *, url: str | object, timeout_seconds: float = 30.0) -> None:
        from mcp_server.client import SyncMCPClient

        self._client = SyncMCPClient(url, timeout_seconds=timeout_seconds)
        self._tool_specs: list[ToolSpec] | None = None
        self._calls = 0

    # ── Lifecycle ─────────────────────────────────────────────────────

    def __enter__(self) -> MCPGroundingBackend:
        try:
            self._client.open()
        except Exception as exc:
            raise GroundingError(f"could not open an MCP session: {exc}") from exc
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Release the session. Safe to call more than once."""
        self._client.close()

    @property
    def call_count(self) -> int:
        """Tool calls made since construction. Logged by the ground node."""
        return self._calls

    # ── Entity tools ──────────────────────────────────────────────────

    def spotlight_link(self, mention: str, context: str) -> dict[str, Any]:
        return self._call(TOOL_SPOTLIGHT_LINK, {"mention": mention, "context": context})

    def search_resource(
        self, label: str, expected_types: list[str] | None = None
    ) -> dict[str, Any]:
        return self._call(TOOL_SEARCH_RESOURCE, {"label": label, "expected_types": expected_types})

    def get_resource_profile(self, uri: str) -> dict[str, Any]:
        return self._call(TOOL_GET_RESOURCE_PROFILE, {"uri": uri})

    def search_class(self, label: str) -> dict[str, Any]:
        return self._call(TOOL_SEARCH_CLASS, {"label": label})

    # ── Relation tools ────────────────────────────────────────────────

    def find_object_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,
        object_types: list[str] | None = None,
    ) -> dict[str, Any]:
        return self._call(
            TOOL_FIND_OBJECT_PROPERTIES,
            {
                "relation_text": relation_text,
                "subject_types": subject_types,
                "object_types": object_types,
            },
        )

    def find_datatype_properties(
        self,
        relation_text: str,
        subject_types: list[str] | None = None,
        literal_datatype: str | None = None,
    ) -> dict[str, Any]:
        return self._call(
            TOOL_FIND_DATATYPE_PROPERTIES,
            {
                "relation_text": relation_text,
                "subject_types": subject_types,
                "literal_datatype": literal_datatype,
            },
        )

    def get_property_profile(self, property_uri: str) -> dict[str, Any]:
        return self._call(TOOL_GET_PROPERTY_PROFILE, {"property_uri": property_uri})

    def get_predicates_between(self, subject_uri: str, object_uri: str) -> dict[str, Any]:
        return self._call(
            TOOL_GET_PREDICATES_BETWEEN,
            {"subject_uri": subject_uri, "object_uri": object_uri},
        )

    # ── Agent plumbing ────────────────────────────────────────────────

    def tool_specs(self) -> list[ToolSpec]:
        """
        The grounding tools, as the running server describes them.

        Read from the server rather than hardcoded, so the descriptions and
        JSON schemas the model sees are the ones the server will actually
        enforce. Tools the server publishes but this module does not know about
        are filtered out: the scaffold's `echo`/`add` are gone, but a shared
        server could carry tools that have nothing to do with grounding, and
        offering them would only widen the model's search space.

        Cached for the life of the backend — the tool list does not change
        while a session is open, and re-listing it per grounding pass would be
        a round-trip for a constant.
        """
        if self._tool_specs is not None:
            return self._tool_specs

        try:
            published = self._client.list_tools()
        except Exception as exc:
            raise GroundingError(f"could not list MCP tools: {exc}") from exc

        specs = [
            ToolSpec(
                name=tool.name,
                description=(getattr(tool, "description", "") or "").strip(),
                input_schema=getattr(tool, "inputSchema", None) or {},
            )
            for tool in published
            if tool.name in ALL_TOOLS
        ]

        missing = sorted(set(ALL_TOOLS) - {spec.name for spec in specs})
        if missing:
            # Not fatal: the model can ground with a subset. But a silently
            # absent tool looks exactly like a model that chose not to use it,
            # so say so once, loudly.
            logger.warning("MCP server does not publish: %s", ", ".join(missing))

        self._tool_specs = specs
        return specs

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Run a tool the model chose, and return its payload as JSON text.

        Never raises — see the port. An unknown name or a dead server comes
        back as `{"error": …}`, which the model can read and route around.
        """
        if name not in ALL_TOOLS:
            return json.dumps({"error": f"unknown tool {name!r}", "available": list(ALL_TOOLS)})
        try:
            return json.dumps(self._call(name, arguments), default=str)
        except GroundingError as exc:
            return json.dumps({"error": str(exc), "tool": name})

    # ── MCP plumbing ──────────────────────────────────────────────────

    def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        One tool call.

        `None` arguments are stripped rather than sent: the server's optional
        parameters default to None anyway, and omitting them keeps the wire
        payload honest about what the caller actually specified.
        """
        payload = {k: v for k, v in arguments.items() if v is not None}
        self._calls += 1
        try:
            return self._client.call_tool_json(tool, payload)
        except Exception as exc:
            raise GroundingError(f"MCP tool {tool!r} failed: {exc}") from exc
