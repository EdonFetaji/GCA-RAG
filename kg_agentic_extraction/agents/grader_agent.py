"""
The grader agent — knowledge graph + documents in, Markdown issue report out.

The model writes the Markdown, but it hands it back inside a schema: the v4
templates ask for `SimpleSchemaGraderReport`, two flat fields, and the provider
is held to decoding exactly that. So `converged` arrives as a bool the model set
deliberately — never derived from a list, never recovered by pattern-matching
prose that an LLM happened to format a particular way. Whether the loop runs
another round is a field lookup, which is what makes it testable.

`issues_markdown` carries the report itself, and `grade_node` passes it through
to the extractor as repair instructions. That the document travels as an escaped
JSON string is the cost of the arrangement; two flat fields is the shape that
survives a strict decoder, which is why the typed `GraderReport` was retired
from this call in the first place.

Two other regimes stay reachable. The unconstrained one lives in code without a
template: `Agent.run_completion` plus `SimpleSchemaGraderReport.from_completion`,
which reads the same two fields out of plain text against a `CONVERGED`
sentinel — worth reaching for if a provider's decoder starts truncating the
report mid-string. v1-v3 remain on disk and ask for a list of typed
`GraderIssue` objects with convergence *derived* from it (`GraderReport`,
rendered by `report_to_markdown`). Reproducing either means changing the call
shape here and `KG_GRADER_PROMPT_VERSION` together — the prompt and the call
shape are one decision, not two.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.models.grading import SimpleSchemaGraderReport
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.validation import ValidationReport
from kg_agentic_extraction.prompts.renderers import format_documents, graph_to_json
from kg_agentic_extraction.validation.base import grader_hints

logger = logging.getLogger(__name__)


@dataclass
class GradingTask:
    """One graph to be graded against the documents it came from."""

    graph: KnowledgeGraph
    documents: list[str]
    max_documents: int = 10
    iteration: int = 1
    #: The structural validator's report on this graph, when it ran (v6 prompt).
    validation: ValidationReport | None = None


class GraderAgent(Agent[GradingTask, SimpleSchemaGraderReport]):
    """Audits a candidate graph for faithfulness to its source documents."""

    name = "grader"

    @property
    def output_schema(self) -> type[SimpleSchemaGraderReport]:
        """
        The schema the provider is constrained to decode into.

        Sent on every grade by the inherited `Agent.run()`, and also the
        contract `grade_node` and `PipelineState` read afterwards.
        """
        return SimpleSchemaGraderReport

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
            # Rendered by v6+ only; empty when nothing is flagged.
            "gnn_flags": grader_hints(payload.validation),
        }

    def post_process(
        self, result: SimpleSchemaGraderReport, payload: GradingTask
    ) -> SimpleSchemaGraderReport:
        """
        Reconcile a report that is not converged but points at nothing.

        Under `GraderReport` the state was impossible, since `converged` was
        derived from the issue list. On the flat model the decoder will hand
        back whatever the model put in the two fields, including `false` beside
        an empty report — which sends the extractor into a repair round with no
        instructions, and would do so again every round to the cap.

        The strip is part of the check, not cosmetics: a decoder that returns a
        field of newlines is claiming issues it did not write.
        """
        result.issues_markdown = result.issues_markdown.strip()

        if not result.converged and not result.issues_markdown:
            logger.warning(
                "[grader] empty issue report on iteration %d — treating as converged",
                payload.iteration,
            )
            result.converged = True
        return result
