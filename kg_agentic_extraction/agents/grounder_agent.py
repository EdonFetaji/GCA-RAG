"""
The grounder agent — refined graph in, DBpedia-aligned graph out.

Runs in two phases: a deterministic retrieval phase that asks the
`GroundingBackend` for candidate resources, then a single LLM call that
disambiguates among them. Splitting it this way keeps the expensive,
non-deterministic step to one call and makes the retrieval half independently
cacheable and testable.

Extension point: to give the model live tool access instead of pre-fetched
candidates, replace `_gather_candidates` with a tool-calling loop and bind the
MCP tools directly. The output contract does not change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from kg_agentic_extraction.agents.base_agent import Agent
from kg_agentic_extraction.grounding.base import GroundingBackend, GroundingError
from kg_agentic_extraction.models.grounding import (
    GroundedEntity,
    GroundedKnowledgeGraph,
    GroundedRelation,
    OntologyMapping,
)
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.ontology import OntologyConfig

logger = logging.getLogger(__name__)


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

    def __init__(self, *, backend: GroundingBackend, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._backend = backend
        # Populated by build_context() and consumed by post_process(); the base
        # class hands the same payload to both, so this stays per-run state
        # keyed by nothing more than call order within a single `run()`.
        self._last_candidates: dict[str, list[OntologyMapping]] = {}

    @property
    def output_schema(self) -> type[GroundingDecisions]:
        return GroundingDecisions

    # ── Phase 1: retrieval ────────────────────────────────────────────

    def _gather_candidates(self, task: GroundingTask) -> dict[str, list[OntologyMapping]]:
        """
        Look up candidates for every entity.

        A backend failure on one entity is recorded as "no candidates" rather
        than aborting the run — partial grounding is a useful result, a crashed
        pipeline is not.
        """
        candidates: dict[str, list[OntologyMapping]] = {}
        for entity in task.graph.entities:
            try:
                candidates[entity.id] = self._backend.lookup_entity(
                    entity.name,
                    entity_type=entity.type.value,
                    limit=task.candidates_per_entity,
                )
            except GroundingError:
                logger.warning("lookup failed for %r; treating as unresolved", entity.name)
                candidates[entity.id] = []
        return candidates

    # ── Phase 2: disambiguation ───────────────────────────────────────

    def build_context(self, payload: GroundingTask) -> dict[str, object]:
        self._last_candidates = self._gather_candidates(payload)

        entities = [
            {
                "id": e.id,
                "name": e.name,
                "type": e.type.value,
                "evidence": [ev.quote for ev in e.evidence[:2]],
            }
            for e in payload.graph.entities
        ]
        relations = [
            {"key": r.key, "relation_type": r.relation_type.value} for r in payload.graph.relations
        ]
        hints = {
            entity_id: [m.model_dump(mode="json") for m in mappings]
            for entity_id, mappings in self._last_candidates.items()
            if mappings
        }

        return {
            "entities_json": json.dumps(entities, indent=2),
            "relations_json": json.dumps(relations, indent=2),
            "candidate_hints": json.dumps(hints, indent=2) if hints else "",
            "entity_types": payload.ontology.entity_type_names,
            "relation_types": payload.ontology.relation_type_names,
        }

    def post_process(
        self, result: GroundingDecisions, payload: GroundingTask
    ) -> GroundingDecisions:
        """
        Backfill anything the model skipped and resolve relation properties.

        Relation properties come from the backend rather than the model: the
        local relation types are a closed enum, so the mapping is a lookup, not
        a judgement call.
        """
        decided_entities = {d.entity_id for d in result.entities}
        for entity in payload.graph.entities:
            if entity.id not in decided_entities:
                result.entities.append(
                    GroundedEntity(
                        entity_id=entity.id,
                        candidates=self._last_candidates.get(entity.id, []),
                        unresolved_reason="not addressed by the grounding model",
                    )
                )

        decided_relations = {d.relation_key for d in result.relations}
        for relation in payload.graph.relations:
            if relation.key in decided_relations:
                continue
            try:
                property_uri = self._backend.resolve_property(relation.relation_type.value)
            except GroundingError:
                property_uri = None
            result.relations.append(
                GroundedRelation(
                    relation_key=relation.key,
                    property_uri=property_uri,
                    unresolved_reason=None if property_uri else "no ontology property mapping",
                )
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
