"""
MCP client — stub.

The bones only. Every method raises `NotImplementedError`; fill them in when
the client is actually needed.

    client = MCPClient("http://127.0.0.1:8931/mcp/")
    await client.connect()
    tools = await client.list_tools()
    result = await client.call_tool("echo", {"message": "hi"})
    await client.close()
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SERVER_URL = "http://127.0.0.1:8931/mcp/"


class MCPClient:
    """
    Client for talking to the MCP server.

    Stub — no transport is opened and no method is implemented yet.
    """

    def __init__(
        self,
        server: str | Any = DEFAULT_SERVER_URL,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        """
        Parameters
        ----------
        server
            The server URL, or a `FastMCP` instance for in-process use.
        timeout_seconds
            Per-request timeout.
        """
        self._server = server
        self._timeout = timeout_seconds
        self._session: Any | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open a session against the server."""
        raise NotImplementedError

    async def close(self) -> None:
        """Close the session. Should be safe to call more than once."""
        raise NotImplementedError

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ── Operations ────────────────────────────────────────────────────

    async def list_tools(self) -> list[Any]:
        """Return the tools the server exposes."""
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """
        Call a tool by name.

        Parameters
        ----------
        name
            Tool name as published by the server, e.g. `"echo"`.
        arguments
            Keyword arguments for the tool.
        """
        raise NotImplementedError

    @property
    def is_connected(self) -> bool:
        return self._session is not None
