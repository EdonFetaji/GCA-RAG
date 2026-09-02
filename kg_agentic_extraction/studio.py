"""
Entry point for `langgraph dev` / LangGraph Studio. See `langgraph.json`.

This module exists only so the graph can be named by a static spec
(`module:variable`) without putting a module-level build into `graph.py` —
`runner.py` and the offline test suite import that module, and building there
would construct a real LLM client on every import.

Settings come from `PipelineSettings`, i.e. from `.env`, so Studio runs exactly
what the CLI runs. Note that grounding is on by default and expects the DBpedia
MCP server at `KG_MCP_URL`; without it a run fails at the `ground` node. Start
it with `uv run python -m mcp_server.server`, or set KG_GROUNDING_ENABLED=false.
"""

from __future__ import annotations

from kg_agentic_extraction.graph import build_graph

#: Compiled with no checkpointer — the dev server attaches its own persistence,
#: which is what gives Studio the per-step state and time travel.
graph = build_graph()
