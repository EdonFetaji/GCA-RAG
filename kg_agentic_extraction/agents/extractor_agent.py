"""
The extractor agent — documents in, knowledge graph out.

Handles both roles in the loop: the first-pass extraction, and the repair pass
that consumes the grader's Markdown. They are one agent rather than two because
the persona, ontology, and output schema are identical — only the user-side
template differs, which `user_role_for()` selects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.ontology import OntologyConfig
from kg_agentic_extraction.prompts.renderers import format_documents, graph_to_json


@dataclass
class ExtractionTask:
    """
    One unit of work for the extractor.

    When `previous_graph` and `grader_markdown` are both set, the agent runs in
    repair mode; otherwise it extracts from scratch.
    """

    documents: list[str]
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    max_documents: int = 10
    previous_graph: KnowledgeGraph | None = None
    grader_markdown: str | None = None

    @property
    def is_repair(self) -> bool:
        return self.previous_graph is not None and bool(self.grader_markdown)


class ExtractorAgent(Agent[ExtractionTask, KnowledgeGraph]):
    """Extracts an evidence-traced knowledge graph from a document cluster."""

    name = "extractor"

    @property
    def output_schema(self) -> type[KnowledgeGraph]:
        return KnowledgeGraph

    def user_role_for(self, payload: ExtractionTask) -> str:
        return "repair" if payload.is_repair else "user"

    def build_context(self, payload: ExtractionTask) -> dict[str, object]:
        documents_text = format_documents(payload.documents, limit=payload.max_documents)
        context: dict[str, object] = {
            "documents_text": documents_text,
            "document_count": min(len(payload.documents), payload.max_documents),
            "entity_types": payload.ontology.entity_type_names,
            "relation_types": payload.ontology.relation_type_names,
            "domain_context": payload.ontology.domain_context,
        }
        if payload.is_repair:
            assert payload.previous_graph is not None  # narrowed by is_repair
            context["previous_graph_json"] = graph_to_json(payload.previous_graph)
            context["grader_markdown"] = payload.grader_markdown
        return context
