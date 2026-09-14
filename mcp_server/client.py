"""
MCP client.

Two classes, one protocol:

`MCPClient` is the async client — a thin, typed wrapper over `fastmcp.Client`
that adds session lifecycle and payload unwrapping.

`SyncMCPClient` is the facade the extraction pipeline uses. LangGraph's nodes,
the agents, and the CLI are all synchronous, while MCP is not. Rather than
colour the whole pipeline async for one I/O boundary — or open a fresh session
per call, which costs a handshake every time and breaks outright when driven
from a notebook's running event loop — it owns a daemon thread running one
event loop, and holds **one** session open for as long as the `with` block
lasts. A grounding pass is dozens of tool calls; they all share that session.

    # async
    async with MCPClient("http://127.0.0.1:8931/mcp/") as client:
        tools = await client.list_tools()
        result = await client.call_tool_json("search_class", {"label": "City"})

    # sync
    with SyncMCPClient() as client:
        result = client.call_tool_json("search_class", {"label": "City"})
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import AsyncExitStack
from typing import Any

logger = logging.getLogger(__name__)

#: Must stay in step with `PipelineSettings.mcp_url`. FastMCP mounts
#: streamable-http at `/mcp/`, trailing slash included.
DEFAULT_SERVER_URL = "http://127.0.0.1:8931/mcp/"


class MCPClientError(RuntimeError):
    """The server could not be reached, or returned something unusable."""


class MCPClient:
    """
    Async client for talking to the MCP server.

    `server` is normally the streamable-HTTP URL of a running server, but
    `fastmcp.Client` also accepts a `FastMCP` *instance*, which is how this
    client gets tested end-to-end over a real protocol session without binding
    a port.
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
        self._stack: AsyncExitStack | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open a session against the server. A no-op if already connected."""
        if self._session is not None:
            return

        # Imported here rather than at module scope so that importing this
        # module — to reference DEFAULT_SERVER_URL, or in a test that never
        # connects — does not require fastmcp to be installed.
        from fastmcp import Client

        stack = AsyncExitStack()
        try:
            session = await stack.enter_async_context(Client(self._server, timeout=self._timeout))
        except Exception as exc:
            await stack.aclose()
            raise MCPClientError(f"could not connect to {self._describe()}: {exc}") from exc

        self._stack, self._session = stack, session
        logger.debug("MCP session open against %s", self._describe())

    async def close(self) -> None:
        """Close the session. Safe to call more than once."""
        stack, self._stack, self._session = self._stack, None, None
        if stack is not None:
            await stack.aclose()
            logger.debug("MCP session closed")

    async def __aenter__(self) -> MCPClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # ── Operations ────────────────────────────────────────────────────

    async def list_tools(self) -> list[Any]:
        """Return the tools the server exposes, each with `.name`/`.description`/`.inputSchema`."""
        session = await self._require_session()
        try:
            return await session.list_tools()
        except Exception as exc:
            raise MCPClientError(f"list_tools failed: {exc}") from exc

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        """
        Call a tool by name and return the raw `CallToolResult`.

        Parameters
        ----------
        name
            Tool name as published by the server, e.g. `"search_class"`.
        arguments
            Keyword arguments for the tool.
        """
        session = await self._require_session()
        logger.debug("MCP → %s(%s)", name, arguments)
        try:
            # raise_on_error=False so a tool-reported error arrives as data and
            # is turned into an MCPClientError here, rather than surfacing as
            # whichever exception type the SDK happens to use this version.
            return await session.call_tool(name, arguments or {}, raise_on_error=False)
        except Exception as exc:
            raise MCPClientError(f"tool {name!r} failed: {exc}") from exc

    async def call_tool_json(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        """Call a tool and return its payload as a plain dict."""
        return self._unwrap(await self.call_tool(name, arguments))

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    # ── Internals ─────────────────────────────────────────────────────

    async def _require_session(self) -> Any:
        if self._session is None:
            await self.connect()
        assert self._session is not None
        return self._session

    def _describe(self) -> str:
        return self._server if isinstance(self._server, str) else type(self._server).__name__

    @staticmethod
    def _unwrap(result: Any) -> dict[str, Any]:
        """
        Pull the payload out of a `CallToolResult`.

        Whether a tool's return value arrives as structured content or as a
        JSON text block depends on how the server declared it, and that has
        moved between SDK versions. Both are accepted so this client is not
        brittle to a server-side detail it does not control.
        """
        if getattr(result, "is_error", None) or getattr(result, "isError", None):
            raise MCPClientError(f"server reported a tool error: {_text_of(result)[:300]}")

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
                    raise MCPClientError(f"tool returned non-JSON text: {text[:200]}") from exc
                if isinstance(parsed, dict):
                    return parsed

        return {}


class SyncMCPClient:
    """
    Synchronous facade over `MCPClient`, holding one session open.

    Owns a daemon thread running its own event loop. Because that loop is never
    the caller's, this works unchanged from a plain script, a notebook, or an
    async web server — the three contexts where `asyncio.run()` per call either
    wastes a handshake or raises outright.

    `with` blocks nest: an inner block does not tear down a session the outer
    one opened. That lets the grounder node hold the session for a whole
    grounding pass while individual helpers still guard their own use of it.
    """

    def __init__(
        self,
        server: str | Any = DEFAULT_SERVER_URL,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._client = MCPClient(server, timeout_seconds=timeout_seconds)
        self._timeout = timeout_seconds
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._depth = 0
        self._lock = threading.RLock()

    # ── Lifecycle ─────────────────────────────────────────────────────

    def open(self) -> None:
        """Start the loop thread and connect. Reference-counted against `close()`."""
        with self._lock:
            self._depth += 1
            if self._depth > 1:
                return
            self._start_loop()
            try:
                self._run(self._client.connect())
            except Exception:
                self._depth = 0
                self._stop_loop()
                raise

    def close(self) -> None:
        """Release one reference; disconnect and stop the thread at zero."""
        with self._lock:
            if self._depth == 0:
                return
            self._depth -= 1
            if self._depth > 0:
                return
            try:
                self._run(self._client.close())
            finally:
                self._stop_loop()

    def __enter__(self) -> SyncMCPClient:
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ── Operations ────────────────────────────────────────────────────

    def list_tools(self) -> list[Any]:
        return self._run(self._client.list_tools())

    def call_tool_json(self, name: str, arguments: dict[str, Any] | None = None) -> dict:
        return self._run(self._client.call_tool_json(name, arguments))

    @property
    def is_connected(self) -> bool:
        return self._client.is_connected

    # ── Internals ─────────────────────────────────────────────────────

    def _start_loop(self) -> None:
        ready = threading.Event()

        def run() -> None:
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()  # type: ignore[union-attr]

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=run, name="mcp-client-loop", daemon=True)
        self._thread.start()
        ready.wait()

    def _stop_loop(self) -> None:
        loop, thread, self._loop, self._thread = self._loop, self._thread, None, None
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5.0)
        loop.close()

    def _run(self, coro: Any) -> Any:
        """Submit a coroutine to the background loop and block for its result."""
        loop = self._loop
        if loop is None:
            coro.close()
            raise MCPClientError("client is not open; use `with SyncMCPClient(...) as client:`")
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        # A little headroom over the per-request timeout, so a request that is
        # merely slow surfaces as the transport's own error rather than as an
        # opaque timeout here.
        return future.result(timeout=self._timeout + 15.0)


def _text_of(result: Any) -> str:
    """Best-effort human-readable rendering of a tool result, for error messages."""
    parts = [
        text
        for block in (getattr(result, "content", None) or [])
        if (text := getattr(block, "text", None))
    ]
    return " ".join(parts) or str(result)
