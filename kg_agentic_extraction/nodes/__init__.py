"""
nodes — the adapter layer between LangGraph state and the pure agents.

Each `make_*_node` is a factory returning a closure over its agent. That
indirection is what keeps `agents/` free of any LangGraph import: the agents
take dataclass payloads, the nodes translate state into those payloads and the
results back into state deltas.
"""

from kg_agentic_extraction.nodes.extract_node import make_extract_node
from kg_agentic_extraction.nodes.grade_node import make_grade_node
from kg_agentic_extraction.nodes.ground_node import make_ground_node
from kg_agentic_extraction.nodes.routing import make_loop_router

__all__ = [
    "make_extract_node",
    "make_grade_node",
    "make_ground_node",
    "make_loop_router",
]
