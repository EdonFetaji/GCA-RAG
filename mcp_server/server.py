"""
The MCP server.

Defines the server, links the tools from `tools.py` onto it, and exposes it.

    uv run python -m mcp_server.server                      # streamable-http
    uv run python -m mcp_server.server --transport stdio    # stdio
"""

from __future__ import annotations

import argparse
import logging
import sys

from fastmcp import FastMCP

from mcp_server.tools import register_tools

logger = logging.getLogger("mcp-server")

SERVER_NAME = "gca-rag"
SERVER_INSTRUCTIONS = """\
Scaffold MCP server. Exposes two example tools:

- `echo` — return a message back with its length.
- `add`  — add two numbers.
"""

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8931


def build_server() -> FastMCP:
    """
    Build the server and link the tools onto it.

    Returns it unstarted, so callers can either run it (see `main`) or hand the
    instance straight to a client for in-process testing.
    """
    mcp = FastMCP(name=SERVER_NAME, instructions=SERVER_INSTRUCTIONS)
    return register_tools(mcp)


#: The module-level server instance, for `fastmcp run mcp_server/server.py`
#: and for importers that just want the object.
mcp = build_server()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mcp_server.server",
        description="Run the MCP server.",
    )
    parser.add_argument(
        "--transport",
        choices=["http", "stdio", "sse"],
        default="http",
        help="http runs it as a networked service (default); "
        "stdio for a client that spawns it as a subprocess.",
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # stdio speaks MCP on stdout, so logs must go to stderr or they corrupt the
    # protocol stream.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stderr,
    )

    if args.transport == "stdio":
        mcp.run(transport="stdio")
        return 0

    logger.info("MCP server on http://%s:%d/mcp/", args.host, args.port)
    mcp.run(transport=args.transport, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
