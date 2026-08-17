"""
Grade node — adapts `GraderAgent` onto graph state.

Also the place where the Markdown artifact is produced: the agent returns a
typed report, and this node renders it once so that both the human-facing
output and the extractor's next repair prompt read the same text.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.grader_agent import GraderAgent, GradingTask
from kg_agentic_extraction.models.ontology import OntologyConfig
from kg_agentic_extraction.prompts.renderers import report_to_markdown
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)


def make_grade_node(
    agent: GraderAgent,
    *,
    ontology: OntologyConfig,
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
            ontology=ontology,
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

        markdown = report_to_markdown(report, iteration=iteration)
        logger.info(
            "grade — iteration %d: %d issue(s)%s",
            iteration,
            len(report.issues),
            " — converged" if report.converged else "",
        )

        return PipelineState(
            grader_report=report,
            grader_reports=[report],
            grader_markdown=markdown,
            converged=report.converged,
        )

    return grade
