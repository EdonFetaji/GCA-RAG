"""grounding — the ontology-resolution port and its MCP adapter."""

from kg_agentic_extraction.grounding.base import (
    GroundingBackend,
    GroundingError,
    NullGroundingBackend,
)
from kg_agentic_extraction.grounding.mcp_backend import MCPGroundingBackend

__all__ = [
    "GroundingBackend",
    "GroundingError",
    "MCPGroundingBackend",
    "NullGroundingBackend",
]
