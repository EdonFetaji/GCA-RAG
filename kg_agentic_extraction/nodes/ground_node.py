"""
Ground node — adapts `GrounderAgent` onto graph state.

Runs once, after the loop settles. Grounding is treated as best-effort: a
failure here leaves `grounded_graph` unset and records the error, but never
discards the refined graph the loop worked to produce.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.grounder_agent import GrounderAgent, GroundingTask
from kg_agentic_extraction.models.ontology import OntologyConfig
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)


def make_ground_node(agent: GrounderAgent, *, ontology: OntologyConfig) -> NodeFn:
    """Build the node function that aligns the refined graph to DBpedia."""

    def ground(state: PipelineState) -> PipelineState:
        graph = state.get("knowledge_graph")
        if graph is None:
            return PipelineState(errors=["ground: no graph to ground"])

        try:
            grounded = agent.ground(GroundingTask(graph=graph, ontology=ontology))
        except Exception as exc:
            logger.exception("grounding failed")
            return PipelineState(errors=[f"ground: {exc}"])

        logger.info(
            "ground — %d/%d entities mapped (%.0f%% coverage)",
            sum(e.is_grounded for e in grounded.entities),
            len(grounded.entities),
            grounded.coverage * 100,
        )
        return PipelineState(grounded_graph=grounded)

    return ground
