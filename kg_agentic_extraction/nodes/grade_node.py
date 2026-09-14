"""
Grade node — adapts `GraderAgent` onto graph state.

The agent answers with `SimpleSchemaGraderReport` — the model's own Markdown
beside an explicit `converged` flag — so this node passes `issues_markdown`
through as the single artifact read both by the human and by the extractor's
next repair prompt. The convergence decision stays on the Pydantic object, where
formatting cannot reach it.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.agents.grader_agent import GraderAgent, GradingTask
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn

logger = logging.getLogger(__name__)

_CONVERGED_BODY = "**No issues.** The graph is faithful to the source documents."


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
            "converged" if report.converged else f"{len(report.issues_markdown)} char report",
        )

        # A converged run leaves the model's report empty, and `--report` still
        # writes this to disk — so say what happened rather than saving a bare
        # heading. The extractor never reads it: the loop has already ended.
        body = report.issues_markdown.strip() or _CONVERGED_BODY

        return PipelineState(
            grader_report=report,
            grader_reports=[report],
            grader_markdown=f"# Grader report — iteration {iteration}\n\n{body}",
            converged=report.converged,
        )

    return grade
