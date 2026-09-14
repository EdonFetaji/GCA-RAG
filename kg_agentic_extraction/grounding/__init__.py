"""grounding — the ontology-resolution port and its MCP adapter."""

from kg_agentic_extraction.grounding.base import (
    GroundingBackend,
    GroundingError,
    NullGroundingBackend,
    mapping_from_hit,
)
from kg_agentic_extraction.grounding.hints import entity_hints, relation_hints
from kg_agentic_extraction.grounding.mcp_backend import ALL_TOOLS, MCPGroundingBackend

__all__ = [
    "ALL_TOOLS",
    "GroundingBackend",
    "GroundingError",
    "MCPGroundingBackend",
    "NullGroundingBackend",
    "entity_hints",
    "mapping_from_hit",
    "relation_hints",
]
