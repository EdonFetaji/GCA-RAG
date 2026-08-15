"""
The grader agent — knowledge graph + documents in, typed issue report out.

The Markdown the user sees is rendered from the report by
`prompts/renderers.report_to_markdown`, not produced by the model. Asking an
LLM for Markdown *and* then parsing it back to decide whether the loop is done
would make convergence depend on formatting; here it depends on a list being
empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.models.grading import GraderReport
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.ontology import OntologyConfig
from kg_agentic_extraction.prompts.renderers import format_documents, graph_to_json


@dataclass
class GradingTask:
    """One graph to be graded against the documents it came from."""

    graph: KnowledgeGraph
    documents: list[str]
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    max_documents: int = 10
    iteration: int = 1


class GraderAgent(Agent[GradingTask, GraderReport]):
    """Audits a candidate graph for faithfulness to its source documents."""

    name = "grader"

    @property
    def output_schema(self) -> type[GraderReport]:
        return GraderReport

    def build_context(self, payload: GradingTask) -> dict[str, object]:
        return {
            "graph_json": graph_to_json(payload.graph),
            "documents_text": format_documents(payload.documents, limit=payload.max_documents),
            "entity_types": payload.ontology.entity_type_names,
            "relation_types": payload.ontology.relation_type_names,
            "iteration": payload.iteration,
        }

    def post_process(self, result: GraderReport, payload: GradingTask) -> GraderReport:
        """
        Drop issues aimed at graph elements that do not exist.

        A grader that cites `entity_47` when the graph has 12 entities would
        otherwise block convergence forever: the extractor cannot fix a
        reference it cannot find, so the issue survives every round. Issues with
        no `element_id` are structural or global and always kept.
        """
        known = {e.id for e in payload.graph.entities} | {r.key for r in payload.graph.relations}
        result.issues = [i for i in result.issues if i.element_id is None or i.element_id in known]
        return result
