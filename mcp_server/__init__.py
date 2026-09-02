"""mcp_server — the DBpedia grounding MCP server (FastMCP) and its client."""

from mcp_server.client import DEFAULT_SERVER_URL, MCPClient, MCPClientError, SyncMCPClient

__all__ = ["DEFAULT_SERVER_URL", "MCPClient", "MCPClientError", "SyncMCPClient"]
