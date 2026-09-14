"""
The MCP server.

Defines the server, links the tools from `tools.py` onto it, and exposes it.

    uv run python -m mcp_server.server                      # streamable-http
    uv run python -m mcp_server.server --transport stdio    # stdio

Endpoints and limits come from `DBPEDIA_*` environment variables — see
`mcp_server/dbpedia/config.py` and `.env.example`.
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
DBpedia grounding toolkit. Maps entity mentions and relation phrases from an
extracted knowledge graph onto canonical DBpedia URIs.

ENTITY TOOLS — surface form → dbpedia.org/resource/…
  spotlight_link         link a mention using the sentence around it. Start here
                         whenever you have context; it is the only tool that
                         reads it.
  search_resource        search by name, optionally narrowed to a class.
                         The fallback when there is no usable context.
  get_resource_profile   abstract, types, redirect and disambiguation flags for
                         one URI. The confirmation step before committing.
  search_class           find the ontology class name to pass as expected_types.

RELATION TOOLS — relation phrase → dbpedia.org/ontology/…
  find_object_properties    properties linking two resources.
  find_datatype_properties  properties whose object is a date/number/string.
  get_property_profile      meaning, domain, range and real-world usage of one
                            property.
  get_predicates_between    what DBpedia actually asserts between two resources.
                            The strongest evidence for a relation.

A workable order: spotlight_link (or search_resource) → get_resource_profile to
confirm → find_object_properties on the relation → get_predicates_between to
check the edge really exists.

No tool raises. A failure comes back as an empty result with `note` explaining
why, so try a different route rather than retrying the same call. Never emit a
URI a tool did not return: a reconstructed one will look plausible and resolve
to nothing.
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
