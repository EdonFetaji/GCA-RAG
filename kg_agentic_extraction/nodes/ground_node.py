"""
Ground node — adapts `GrounderAgent` onto graph state.

Runs once, after the loop settles. Grounding is treated as best-effort: a
failure here leaves `grounded_graph` unset and records the error, but never
discards the refined graph the loop worked to produce.

The node owns the backend's session lifetime. One `with` block spans the whole
pass, so the agent's dozens of tool calls share a single MCP connection —
opening it here rather than in the agent keeps the agent testable without one.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.grounder_agent import GrounderAgent, GroundingTask
from kg_agentic_extraction.grounding.base import GroundingBackend
from kg_agentic_extraction.models.ontology import OntologyConfig
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)


def make_ground_node(
    agent: GrounderAgent,
    *,
    ontology: OntologyConfig,
    backend: GroundingBackend,
) -> NodeFn:
    """Build the node function that aligns the refined graph to DBpedia."""

    def ground(state: PipelineState) -> PipelineState:
        graph = state.get("knowledge_graph")
        if graph is None:
            return PipelineState(errors=["ground: no graph to ground"])

        try:
            with backend:
                grounded = agent.ground(GroundingTask(graph=graph, ontology=ontology))
        except Exception as exc:
            logger.exception("grounding failed")
            return PipelineState(errors=[f"ground: {exc}"])

        logger.info(
            "ground — %d/%d entities and %d/%d relations mapped "
            "(%.0f%% entity coverage, %d tool call(s))",
            sum(e.is_grounded for e in grounded.entities),
            len(grounded.entities),
            sum(r.is_grounded for r in grounded.relations),
            len(grounded.relations),
            grounded.coverage * 100,
            getattr(backend, "call_count", 0),
        )
        return PipelineState(grounded_graph=grounded)

    return ground
