"""
The grader agent — knowledge graph + documents in, Markdown issue report out.

The report is written by the model, not rendered from typed issues by
`prompts/renderers.report_to_markdown`. The earlier design asked for a list of
`GraderIssue` objects precisely so that convergence would not depend on
formatting — it was `not report.issues`, evaluated on a Pydantic object.

Two things moved it. The grader now runs on Mistral, whose `json_schema` mode
decodes strictly and handles a flat two-field object far better than a list of
nested objects carrying enums; and the extractor's repair prompt consumed the
rendered Markdown anyway, so the typed intermediate was being flattened one step
later regardless. `MistralGraderReport` keeps convergence off the formatting by
carrying an explicit `converged` flag beside the prose.

`GraderReport` and its renderer are still there, unchanged. Pointing
`output_schema` back at it, together with `KG_GRADER_PROMPT_VERSION=v3`,
reproduces the old regime — the two have to move together, because a v1–v3
template asks for typed issues and v4 asks for Markdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.models.grading import MistralGraderReport
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.prompts.renderers import format_documents, graph_to_json

logger = logging.getLogger(__name__)


@dataclass
class GradingTask:
    """One graph to be graded against the documents it came from."""

    graph: KnowledgeGraph
    documents: list[str]
    max_documents: int = 10
    iteration: int = 1


class GraderAgent(Agent[GradingTask, MistralGraderReport]):
    """Audits a candidate graph for faithfulness to its source documents."""

    name = "grader"

    @property
    def output_schema(self) -> type[MistralGraderReport]:
        return MistralGraderReport

    def build_context(self, payload: GradingTask) -> dict[str, object]:
        return {
            "graph_json": graph_to_json(payload.graph),
            "documents_text": format_documents(payload.documents, limit=payload.max_documents),
            # The vocabulary the extractor invented, handed back so the grader
            # can audit it for internal consistency. There is no allowed-list to
            # check against any more, so "are these labels used coherently?" is
            # the question that replaces "are these labels permitted?".
            "entity_type_vocabulary": payload.graph.entity_type_vocabulary,
            "relation_type_vocabulary": payload.graph.relation_type_vocabulary,
            "iteration": payload.iteration,
        }

    def post_process(
        self, result: MistralGraderReport, payload: GradingTask
    ) -> MistralGraderReport:
        """
        Treat "not converged, but no issues named" as converged.

        The typed schema could not express that state — an unconverged report
        had issues in it by definition. A free-text one can, and it is the
        expensive failure: the extractor would be sent into a repair round whose
        prompt contains an empty issue list, produce something arbitrary, and be
        graded again, all the way to `max_iterations`. A grader with nothing to
        say is done.

        Note what is *not* here any more: the old filter that dropped issues
        citing element ids the graph does not contain. Prose cannot be filtered
        that way, so `max_iterations` is now the only bound on a grader that
        keeps re-raising something the extractor cannot find.
        """
        result.issues_markdown = result.issues_markdown.strip()
        if not result.converged and not result.issues_markdown:
            logger.warning(
                "[grader] reported unconverged with an empty report on iteration %d; "
                "treating as converged",
                payload.iteration,
            )
            result.converged = True
        return result
