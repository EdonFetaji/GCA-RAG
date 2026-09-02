"""
The grounder agent — refined graph in, DBpedia-aligned graph out.

A tool-calling agent. It is handed the graph and the DBpedia toolkit, and it
drives its own retrieval: link a mention in context, search by name, read a
resource profile to confirm, look up the property a relation maps to, check
whether DBpedia actually asserts the edge. When it stops asking for tools, one
constrained call turns the transcript into `GroundingDecisions`.

This replaces an earlier design where retrieval was a fixed pre-fetch and the
model only chose among candidates. ADR 0003 records why: with one lookup per
entity there is no way to tell the city from the band, because the fact that
settles it — the resource's abstract and types — is one hop past the candidate
list. See `docs/adr/0003-tool-calling-grounder.md`.

The agent still knows nothing about MCP. It depends on `GroundingBackend` for
the tools and `ToolCallingLLMClient` for the loop, and can be exercised with a
scripted fake of each.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.grounding.base import GroundingBackend
from kg_agentic_extraction.grounding.hints import entity_hints, relation_hints
from kg_agentic_extraction.llm.base import ToolInvocation
from kg_agentic_extraction.models.grounding import (
    GroundedEntity,
    GroundedKnowledgeGraph,
    GroundedRelation,
)
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.ontology import OntologyConfig

logger = logging.getLogger(__name__)

#: Evidence quotes per entity in the prompt. Two is enough for the model to
#: disambiguate on, and the full set would crowd out the tool transcript.
_EVIDENCE_PER_ENTITY = 2


class GroundingDecisions(BaseModel):
    """
    What the LLM is asked for.

    Narrower than `GroundedKnowledgeGraph`: the model returns only the mapping
    decisions, and the agent re-attaches them to the graph. Sending the whole
    graph back through the model would risk it quietly editing entities the
    grader already approved.
    """

    entities: list[GroundedEntity] = Field(default_factory=list)
    relations: list[GroundedRelation] = Field(default_factory=list)


@dataclass
class GroundingTask:
    """One refined graph to align against the ontology."""

    graph: KnowledgeGraph
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    candidates_per_entity: int = 5


class GrounderAgent(Agent[GroundingTask, GroundingDecisions]):
    """Maps a refined graph's entities and relations onto DBpedia resources."""

    name = "grounder"

    def __init__(
        self,
        *,
        backend: GroundingBackend,
        max_tool_rounds: int = 8,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._backend = backend
        self._max_tool_rounds = max_tool_rounds

    @property
    def output_schema(self) -> type[GroundingDecisions]:
        return GroundingDecisions

    # ── The loop ──────────────────────────────────────────────────────

    def run(self, payload: GroundingTask) -> GroundingDecisions:
        """
        Render the prompts, let the model work the tools, post-process.

        Overrides the base Template Method rather than filling in its hooks:
        the base runs exactly one model call, and this agent's whole point is
        that it runs as many as the disambiguation needs. The `build_context` →
        render → `post_process` shape is preserved so the difference is the
        number of calls, not the structure.
        """
        rendered = self._prompts.render_pair(
            self.name,
            user_role=self.user_role_for(payload),
            version=self._prompt_version,
            **self.build_context(payload),
        )

        tools = self._backend.tool_specs()
        logger.info(
            "[grounder] grounding %d entities / %d relations with %d tool(s)",
            len(payload.graph.entities),
            len(payload.graph.relations),
            len(tools),
        )

        result = self._llm.run_tool_loop(  # type: ignore[attr-defined]
            system=rendered.system,
            user=rendered.user,
            tools=tools,
            execute=self._execute,
            schema=self.output_schema,
            max_rounds=self._max_tool_rounds,
        )
        return self.post_process(result, payload)

    def _execute(self, invocation: ToolInvocation) -> str:
        """Route one model-chosen tool call to the backend. Contracted not to raise."""
        return self._backend.dispatch(invocation.name, invocation.arguments)

    # ── Prompt context ────────────────────────────────────────────────

    def build_context(self, payload: GroundingTask) -> dict[str, object]:
        entities = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type,
                "evidence": [ev.quote for ev in e.evidence[:_EVIDENCE_PER_ENTITY]],
            }
            for e in payload.graph.entities
        ]

        # Relations carry their endpoints' names, not just ids: the model needs
        # a surface form to search on, and an id alone would send it back to
        # the entity list on every relation.
        index = payload.graph.entity_index
        relations = [
            {
                "key": r.key,
                "relation_type": r.relation_type,
                "source_name": index[r.source].name if r.source in index else r.source,
                "target_name": index[r.target].name if r.target in index else r.target,
                "evidence": [ev.quote for ev in r.evidence[:1]],
            }
            for r in payload.graph.relations
        ]

        # Both the type lists and the hint tables come off the graph, not off
        # `payload.ontology`: extraction is open-vocabulary, so the types in
        # front of the model are whatever the extractor invented. The ontology
        # survives only as the source of the hint tables — the controlled
        # vocabulary is now a target to map onto, not a constraint upstream.
        graph_entity_types = payload.graph.entity_type_vocabulary
        graph_relation_types = payload.graph.relation_type_vocabulary

        return {
            "entities_json": json.dumps(entities, indent=2),
            "relations_json": json.dumps(relations, indent=2),
            "entity_types": graph_entity_types,
            "relation_types": graph_relation_types,
            "entity_hints": json.dumps(entity_hints(graph_entity_types), indent=2),
            "relation_hints": json.dumps(relation_hints(graph_relation_types), indent=2),
            "max_tool_rounds": self._max_tool_rounds,
        }

    # ── Post-processing ───────────────────────────────────────────────

    def post_process(
        self, result: GroundingDecisions, payload: GroundingTask
    ) -> GroundingDecisions:
        """
        Backfill anything the model skipped, and drop anything it invented.

        Two jobs, both about the gap between what was asked for and what came
        back. A missing element is indistinguishable from a forgotten one, so
        it is recorded as explicitly unresolved. A decision naming an element
        that is not in the graph is dropped outright — the grader already
        signed the graph off, and grounding is not allowed to add to it.
        """
        entity_ids = {e.id for e in payload.graph.entities}
        relation_keys = {r.key for r in payload.graph.relations}

        result.entities = [d for d in result.entities if _keep(d.entity_id, entity_ids, "entity")]
        result.relations = [
            d for d in result.relations if _keep(d.relation_key, relation_keys, "relation")
        ]

        decided_entities = {d.entity_id for d in result.entities}
        result.entities.extend(
            GroundedEntity(
                entity_id=entity.id,
                unresolved_reason="not addressed by the grounding model",
            )
            for entity in payload.graph.entities
            if entity.id not in decided_entities
        )

        decided_relations = {d.relation_key for d in result.relations}
        result.relations.extend(
            GroundedRelation(
                relation_key=relation.key,
                unresolved_reason="not addressed by the grounding model",
            )
            for relation in payload.graph.relations
            if relation.key not in decided_relations
        )
        return result

    # ── Convenience ───────────────────────────────────────────────────

    def ground(self, task: GroundingTask) -> GroundedKnowledgeGraph:
        """Run the agent and attach the decisions to the original graph."""
        decisions = self.run(task)
        return GroundedKnowledgeGraph(
            graph=task.graph,
            entities=decisions.entities,
            relations=decisions.relations,
        )


def _keep(identifier: str, known: set[str], kind: str) -> bool:
    if identifier in known:
        return True
    logger.warning("[grounder] dropping decision for unknown %s %r", kind, identifier)
    return False
