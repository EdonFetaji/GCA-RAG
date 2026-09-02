"""
Extract node — adapts `ExtractorAgent` onto graph state.

The node is a closure over its agent rather than a class, because a LangGraph
node is exactly one function of state; the agent holds everything stateful.
Nodes return only the keys they changed, which is what lets the reducers in
`state.py` do the merging.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.extractor_agent import ExtractionTask, ExtractorAgent
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)


def make_extract_node(
    agent: ExtractorAgent,
    *,
    domain_context: str,
    max_documents: int,
) -> NodeFn:
    """Build the node function that runs first-pass extraction *and* repairs."""

    def extract(state: PipelineState) -> PipelineState:
        iteration = state.get("iteration", 0) + 1
        previous = state.get("knowledge_graph")
        feedback = state.get("grader_markdown")

        task = ExtractionTask(
            documents=state["documents"],
            domain_context=domain_context,
            max_documents=max_documents,
            # Both must be present for the agent to enter repair mode; on the
            # first pass they are None and it extracts from scratch.
            previous_graph=previous,
            grader_markdown=feedback,
        )
        logger.info(
            "extract — iteration %d (%s)", iteration, "repair" if task.is_repair else "first pass"
        )

        try:
            graph = agent.run(task)
        except Exception as exc:
            logger.exception("extraction failed on iteration %d", iteration)
            # Keep the previous graph so the run can still ground whatever was
            # last known-good rather than ending with nothing.
            return PipelineState(
                iteration=iteration,
                errors=[f"extract[{iteration}]: {exc}"],
            )

        logger.info(
            "extract — %d entities, %d relations", len(graph.entities), len(graph.relations)
        )
        return PipelineState(knowledge_graph=graph, iteration=iteration)

    return extract
