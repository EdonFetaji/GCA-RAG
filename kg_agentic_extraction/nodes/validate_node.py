"""
Validate node — runs the structural validator between extract and grade.

Best-effort: a failure costs the round its hints, not the run. A stale report
is cleared so the grader never sees flags computed on a different graph.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import NodeFn
from kg_agentic_extraction.validation.base import GraphValidator

logger = logging.getLogger(__name__)


def make_validate_node(validator: GraphValidator) -> NodeFn:
    """Build the node function that scores the current graph."""

    def validate(state: PipelineState) -> PipelineState:
        graph = state.get("knowledge_graph")
        iteration = state.get("iteration", 1)
        if graph is None:
            return PipelineState(validation_report=None)

        try:
            report = validator.validate(graph, iteration=iteration)
        except Exception as exc:
            logger.exception("validation failed on iteration %d", iteration)
            return PipelineState(validation_report=None, errors=[f"validate[{iteration}]: {exc}"])

        logger.info(
            "validate — iteration %d: consistency %.2f, %d element(s) flagged",
            iteration,
            report.consistency,
            len(report.flagged),
        )
        return PipelineState(validation_report=report, validation_reports=[report])

    return validate
