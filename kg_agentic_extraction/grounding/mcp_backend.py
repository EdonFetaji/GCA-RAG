"""
`GroundingBackend` implemented over the standalone DBpedia MCP server.

This is the only place in the pipeline that speaks MCP. The server itself lives
in `mcp_servers/dbpedia/` and is never imported from here — the two sides are
coupled by the protocol and the tool names below, nothing else, so the server
can be restarted, replaced, or moved to another host independently.

The MCP client API is async; the `GroundingBackend` port is sync because the
agents and LangGraph nodes are. The bridging happens here, in one place, rather
than colouring the whole pipeline async for one I/O boundary.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from kg_agentic_extraction.grounding.base import GroundingError
from kg_agentic_extraction.models.grounding import OntologyMapping

logger = logging.getLogger(__name__)

# Tool names as published by mcp_servers/dbpedia/tools.py. Kept as constants so
# a rename on the server side fails loudly here rather than silently returning
# no candidates.
TOOL_LOOKUP_ENTITY = "lookup_entity"
TOOL_RESOLVE_PROPERTY = "resolve_property"
TOOL_RESOLVE_CLASS = "resolve_ontology_class"


class MCPGroundingBackend:
    """
    Talks to the DBpedia MCP server.

    `server` is normally the streamable-HTTP URL of a running server, but the
    SDK's high-level client also accepts a server *instance*, which is how this
    adapter gets tested end-to-end over a real protocol session without binding
    a port.

    Sessions are opened per call rather than held open for the life of the
    pipeline: grounding is a short burst of lookups at the end of a run, and a
    long-lived session would have to survive the extractor↔grader loop, which
    can take minutes, without the server timing it out.
    """

    def __init__(self, *, url: str | object, timeout_seconds: float = 30.0) -> None:
        self._server = url
        self._timeout = timeout_seconds
        self._property_cache: dict[str, str | None] = {}

    # ── GroundingBackend ──────────────────────────────────────────────

    def lookup_entity(
        self,
        name: str,
        *,
        entity_type: str | None = None,
        limit: int = 5,
    ) -> list[OntologyMapping]:
        payload = self._call_tool(
            TOOL_LOOKUP_ENTITY,
            {"name": name, "entity_type": entity_type, "limit": limit},
        )
        return [OntologyMapping.model_validate(item) for item in payload.get("results", [])]

    def resolve_property(self, relation_type: str) -> str | None:
        """Relation types are a closed enum, so results are worth caching for the run."""
        if relation_type not in self._property_cache:
            payload = self._call_tool(TOOL_RESOLVE_PROPERTY, {"relation_type": relation_type})
            self._property_cache[relation_type] = payload.get("property_uri")
        return self._property_cache[relation_type]

    def resolve_ontology_class(self, entity_type: str) -> str | None:
        """Not part of the port, but exposed for callers that want the class directly."""
        return self._call_tool(TOOL_RESOLVE_CLASS, {"entity_type": entity_type}).get("class_uri")

    def close(self) -> None:
        self._property_cache.clear()

    # ── MCP plumbing ──────────────────────────────────────────────────

    def _call_tool(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call, bridging the async MCP client onto this sync API."""
        try:
            return asyncio.run(self._call_tool_async(tool, arguments))
        except GroundingError:
            raise
        except RuntimeError as exc:
            # asyncio.run() refuses to nest. Reaching this means the pipeline is
            # being driven from an existing loop (a notebook, an async server),
            # which needs a genuinely async backend rather than a thread hack.
            if "cannot be called from a running event loop" in str(exc):
                raise GroundingError(
                    "MCPGroundingBackend is sync-only and was called from inside a running "
                    "event loop; drive the pipeline from a plain thread, or add an async "
                    "backend implementing the same port."
                ) from exc
            raise GroundingError(f"MCP tool {tool!r} failed: {exc}") from exc
        except Exception as exc:
            raise GroundingError(f"MCP tool {tool!r} failed: {exc}") from exc

    async def _call_tool_async(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        # Imported lazily so the pipeline imports (and the graph compiles)
        # without the MCP client installed or the server running.
        from mcp import Client

        logger.debug("MCP → %s(%s)", tool, arguments)
        async with Client(self._server, read_timeout_seconds=self._timeout) as client:
            result = await client.call_tool(tool, arguments)
        return self._unwrap(result)

    @staticmethod
    def _unwrap(result: Any) -> dict[str, Any]:
        """
        Pull the payload out of a `CallToolResult`.

        Whether a tool's return value arrives as structured content or as a JSON
        text block depends on how the server declared it, and that has moved
        between SDK versions. Both are accepted so this adapter is not brittle
        to a server-side detail it does not control.
        """
        if getattr(result, "is_error", None) or getattr(result, "isError", None):
            raise GroundingError(f"MCP server reported an error: {result}")

        for attr in ("structured_content", "structuredContent"):
            structured = getattr(result, attr, None)
            if isinstance(structured, dict):
                return structured

        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)
            if text:
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise GroundingError(f"tool returned non-JSON text: {text[:200]}") from exc
                if isinstance(parsed, dict):
                    return parsed

        return {}
