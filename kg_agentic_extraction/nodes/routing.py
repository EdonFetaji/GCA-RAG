"""
Loop routing policy.

The one place that decides whether the extractor↔grader loop keeps going.
Isolated from the nodes so the stopping rule can be changed — or swapped for a
different policy — without touching an agent or the graph wiring.
"""

from __future__ import annotations

import logging

from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.types import LoopDecision, RouterFn

logger = logging.getLogger(__name__)


def make_loop_router(*, max_iterations: int, grounding_enabled: bool) -> RouterFn:
    """
    Build the conditional edge out of the grade node.

    Three outcomes:

    - ``refine``  — the grader found issues and there are rounds left.
    - ``ground``  — done looping, and grounding is switched on.
    - ``end``     — done looping, grounding is off.

    Exhausting `max_iterations` is a legitimate exit, not an error: some graphs
    have issues no amount of re-prompting resolves, and looping forever on them
    would burn the token budget for every other cluster in a batch.
    """

    def route(state: PipelineState) -> LoopDecision:
        converged = state.get("converged", False)
        iteration = state.get("iteration", 0)
        finished = "ground" if grounding_enabled else "end"

        if converged:
            logger.info("routing — converged after %d iteration(s)", iteration)
            return finished

        if iteration >= max_iterations:
            # The grader's findings are prose since v4, so there is no issue
            # count to report here any more — just say the report is unresolved.
            logger.warning(
                "routing — iteration cap (%d) hit with the grader still reporting issues; stopping",
                max_iterations,
            )
            return finished

        logger.info("routing — refining (iteration %d of %d)", iteration, max_iterations)
        return "refine"

    return route
