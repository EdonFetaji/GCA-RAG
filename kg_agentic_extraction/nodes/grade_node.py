"""
Grade node — adapts `GraderAgent` onto graph state.

This node used to render the Markdown artifact, turning the agent's typed report
into text with `report_to_markdown`. Since the v4 grader the model writes that
Markdown itself, so the node just threads it through: one artifact, read both by
the human and by the extractor's next repair prompt, as before.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.grader_agent import GraderAgent, GradingTask
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)


def make_grade_node(
    agent: GraderAgent,
    *,
    max_documents: int,
) -> NodeFn:
    """Build the node function that audits the current graph."""

    def grade(state: PipelineState) -> PipelineState:
        graph = state.get("knowledge_graph")
        iteration = state.get("iteration", 1)

        if graph is None:
            # Extraction failed; there is nothing to grade. Mark converged so
            # routing exits the loop instead of spinning on an absent graph.
            logger.warning("grade — no graph to grade, ending loop")
            return PipelineState(
                converged=True,
                errors=[f"grade[{iteration}]: no graph produced by the extractor"],
            )

        task = GradingTask(
            graph=graph,
            documents=state["documents"],
            max_documents=max_documents,
            iteration=iteration,
        )

        try:
            report = agent.run(task)
        except Exception as exc:
            logger.exception("grading failed on iteration %d", iteration)
            # An unusable grader cannot be allowed to block the run forever;
            # treat the graph as final and let the error surface in the result.
            return PipelineState(
                converged=True,
                errors=[f"grade[{iteration}]: {exc}"],
            )

        logger.info(
            "grade — iteration %d: %s",
            iteration,
            "converged" if report.converged else f"{len(report.issues_markdown)} chars of issues",
        )

        return PipelineState(
            grader_report=report,
            grader_reports=[report],
            grader_markdown=report.issues_markdown,
            converged=report.converged,
        )

    return grade
