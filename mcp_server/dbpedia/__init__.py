"""
dbpedia — the transports and contracts behind the server's DBpedia tools.

Three services are fronted here, each with its own module:

    spotlight.py  context-aware entity linking      (api.dbpedia-spotlight.org)
    lookup.py     keyword search over labels        (lookup.dbpedia.org)
    sparql.py     everything else                   (dbpedia.org/sparql)

`tools.py` is the only importer. Nothing in this package knows about the
extraction pipeline or its ontology — the server is DBpedia-only, so any MCP
client can use it.
"""

from mcp_server.dbpedia.config import DBpediaSettings, get_settings

__all__ = ["DBpediaSettings", "get_settings"]
