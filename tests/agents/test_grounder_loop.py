"""
The grounder's tool-calling loop.

The LLM is a scripted fake, so these assert the agent's *behaviour* — what it
dispatches, what it does with what comes back — without a network or an API
key. The backend is real, running in-process over MCP against stubbed HTTP.
"""

from __future__ import annotations

import json

import pytest

from kg_agentic_extraction.agents.grounder_agent import (
    GrounderAgent,
    GroundingDecisions,
    GroundingTask,
)
from kg_agentic_extraction.grounding.base import NullGroundingBackend
from kg_agentic_extraction.grounding.mcp_backend import MCPGroundingBackend
from kg_agentic_extraction.llm.base import ToolInvocation, ToolSpec
from kg_agentic_extraction.models.grounding import GroundedEntity, GroundedRelation
from kg_agentic_extraction.models.knowledge_graph import (
    Entity,
    EvidenceSpan,
    KnowledgeGraph,
    Relation,
)
from kg_agentic_extraction.models.ontology import EntityType, RelationType
from kg_agentic_extraction.prompts.registry import PromptRegistry
from mcp_server.dbpedia.classes import DBO, DBR
from mcp_server.server import build_server
from tests.conftest import lookup_docs, sparql_results

DBR_SEATTLE = f"{DBR}Seattle"
DBR_WASHINGTON = f"{DBR}Washington_(state)"


# ── Doubles ───────────────────────────────────────────────────────────


class ScriptedLLM:
    """
    A `ToolCallingLLMClient` that replays a fixed plan.

    `rounds` is a list of tool-call batches; each is dispatched, and the results
    are recorded so a test can assert on what the tools actually returned.
    """

    def __init__(self, rounds: list[list[tuple[str, dict]]], answer: GroundingDecisions) -> None:
        self._rounds = rounds
        self._answer = answer
        self.dispatched: list[tuple[str, dict]] = []
        self.results: list[str] = []
        self.tools_seen: list[ToolSpec] = []
        self.direct_calls = 0

    def structured(self, *, system, user, schema):  # noqa: ARG002
        self.direct_calls += 1
        return self._answer

    def run_tool_loop(self, *, system, user, tools, execute, schema, max_rounds=8):  # noqa: ARG002
        self.tools_seen = tools
        if not tools:
            return self.structured(system=system, user=user, schema=schema)
        for batch in self._rounds[:max_rounds]:
            for index, (name, arguments) in enumerate(batch):
                self.dispatched.append((name, arguments))
                self.results.append(execute(ToolInvocation(f"c{index}", name, arguments)))
        return self._answer


def make_agent(llm, backend, **kwargs):
    return GrounderAgent(
        backend=backend,
        llm=llm,
        prompts=PromptRegistry(default_version="v2"),
        prompt_version="v2",
        **kwargs,
    )


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    return KnowledgeGraph(
        entities=[
            Entity(
                id="e1",
                name="Seattle",
                type=EntityType.LOCATION,
                evidence=[EvidenceSpan(document_index=0, quote="Seattle is a seaport city.")],
            ),
            Entity(id="e2", name="Washington", type=EntityType.LOCATION),
        ],
        relations=[
            Relation(source="e1", target="e2", relation_type=RelationType.LOCATED_IN),
        ],
    )


@pytest.fixture
def backend():
    return MCPGroundingBackend(url=build_server())


@pytest.fixture
def dbpedia(stub_http):
    """Answers the queries this scenario makes, from either service."""

    def router(url, params):
        if "lookup" in url:
            return lookup_docs(
                {
                    "resource": DBR_SEATTLE,
                    "label": "Seattle",
                    "comment": "Seattle is a seaport city on the West Coast.",
                    "type": [f"{DBO}City"],
                    "refCount": "1778",
                    "score": "32542.6",
                }
            )
        query = params.get("query", "")
        if "?direction" in query:
            return sparql_results({"p": f"{DBO}subdivision", "direction": "forward"})
        return sparql_results({"p": f"{DBO}wikiPageWikiLink", "o": DBR_WASHINGTON})

    stub_http(router)


# ── The loop ──────────────────────────────────────────────────────────


def test_the_model_is_offered_every_tool(graph, backend, dbpedia):
    llm = ScriptedLLM([], GroundingDecisions())
    with backend:
        make_agent(llm, backend).run(GroundingTask(graph=graph))

    assert {spec.name for spec in llm.tools_seen} == {
        "spotlight_link",
        "search_resource",
        "get_resource_profile",
        "search_class",
        "find_object_properties",
        "find_datatype_properties",
        "get_property_profile",
        "get_predicates_between",
    }


def test_tool_calls_reach_dbpedia_and_their_results_come_back(graph, backend, dbpedia):
    llm = ScriptedLLM(
        rounds=[
            [("search_resource", {"label": "Seattle", "expected_types": ["dbo:City"]})],
            [
                (
                    "get_predicates_between",
                    {"subject_uri": DBR_SEATTLE, "object_uri": DBR_WASHINGTON},
                )
            ],
        ],
        answer=GroundingDecisions(),
    )
    with backend:
        make_agent(llm, backend).run(GroundingTask(graph=graph))

    assert [name for name, _ in llm.dispatched] == ["search_resource", "get_predicates_between"]
    search, predicates = (json.loads(r) for r in llm.results)
    assert search["results"][0]["uri"] == DBR_SEATTLE
    assert predicates["predicates"][0]["predicate"] == f"{DBO}subdivision"


def test_the_round_budget_is_honoured(graph, backend, dbpedia):
    llm = ScriptedLLM(
        rounds=[[("search_class", {"label": f"C{i}"})] for i in range(10)],
        answer=GroundingDecisions(),
    )
    with backend:
        make_agent(llm, backend, max_tool_rounds=3).run(GroundingTask(graph=graph))

    assert len(llm.dispatched) == 3


def test_a_failing_tool_is_reported_to_the_model_not_raised(graph, backend, dbpedia):
    llm = ScriptedLLM(
        rounds=[[("get_resource_profile", {"uri": "not-a-uri"})]],
        answer=GroundingDecisions(),
    )
    with backend:
        make_agent(llm, backend).run(GroundingTask(graph=graph))

    payload = json.loads(llm.results[0])
    assert payload["found"] is False
    assert "rejected URI" in payload["note"]


def test_no_backend_degrades_to_a_plain_call(graph):
    llm = ScriptedLLM([], GroundingDecisions())
    make_agent(llm, NullGroundingBackend()).run(GroundingTask(graph=graph))

    # Grounding without tools is a worse answer, not an error.
    assert llm.tools_seen == []
    assert llm.direct_calls == 1


# ── Post-processing ───────────────────────────────────────────────────


def test_every_entity_and_relation_appears_in_the_output(graph, backend, dbpedia):
    """A missing element is indistinguishable from a forgotten one, so the agent
    fills the gaps with explicit non-answers."""
    llm = ScriptedLLM([], GroundingDecisions())
    with backend:
        result = make_agent(llm, backend).run(GroundingTask(graph=graph))

    assert {d.entity_id for d in result.entities} == {"e1", "e2"}
    assert {d.relation_key for d in result.relations} == {"e1|LOCATED_IN|e2"}
    assert all(d.unresolved_reason for d in result.entities)


def test_a_decision_for_an_entity_that_is_not_in_the_graph_is_dropped(graph, backend, dbpedia):
    """The grader already signed the graph off; grounding may not add to it."""
    llm = ScriptedLLM(
        [],
        GroundingDecisions(
            entities=[
                GroundedEntity(entity_id="e1"),
                GroundedEntity(entity_id="ghost"),
            ],
            relations=[GroundedRelation(relation_key="e9|CAUSES|e8")],
        ),
    )
    with backend:
        result = make_agent(llm, backend).run(GroundingTask(graph=graph))

    assert "ghost" not in {d.entity_id for d in result.entities}
    assert {d.relation_key for d in result.relations} == {"e1|LOCATED_IN|e2"}


def test_ground_attaches_the_decisions_to_the_untouched_graph(graph, backend, dbpedia):
    from kg_agentic_extraction.models.grounding import OntologyMapping

    llm = ScriptedLLM(
        [],
        GroundingDecisions(
            entities=[
                GroundedEntity(
                    entity_id="e1",
                    mapping=OntologyMapping(uri=DBR_SEATTLE, label="Seattle", confidence=0.9),
                )
            ],
        ),
    )
    with backend:
        grounded = make_agent(llm, backend).ground(GroundingTask(graph=graph))

    assert grounded.graph is graph  # additive, never destructive
    assert grounded.coverage == pytest.approx(0.5)


# ── Prompt context ────────────────────────────────────────────────────


def test_relations_carry_their_endpoint_names(graph, backend):
    """An id alone would send the model back to the entity list for every relation."""
    context = make_agent(ScriptedLLM([], GroundingDecisions()), backend).build_context(
        GroundingTask(graph=graph)
    )
    relations = json.loads(context["relations_json"])
    assert relations[0]["source_name"] == "Seattle"
    assert relations[0]["target_name"] == "Washington"


def test_the_prompt_renders_with_the_hint_tables(graph, backend):
    """`StrictUndefined` means a variable the template wants and the agent forgot
    is a render error, not a silently empty prompt."""
    agent = make_agent(ScriptedLLM([], GroundingDecisions()), backend)
    rendered = PromptRegistry(default_version="v2").render_pair(
        "grounder", version="v2", **agent.build_context(GroundingTask(graph=graph))
    )
    assert "dbo:locatedInArea" in rendered.system
    assert "Seattle" in rendered.user


def test_hints_cover_the_graphs_types_and_nothing_else(graph, backend):
    """
    The hint table is keyed by what the graph contains, not by the ontology enum.

    Rendering the whole enum was affordable when extraction was constrained to
    it; now it would be noise, since a graph uses a handful of types and the
    enum is not where they came from.
    """
    agent = make_agent(ScriptedLLM([], GroundingDecisions()), backend)
    context = agent.build_context(GroundingTask(graph=graph))

    assert context["entity_types"] == ["LOCATION"]
    assert context["relation_types"] == ["LOCATED_IN"]
    assert set(json.loads(context["entity_hints"])) == {"LOCATION"}
    # ORGANIZATION is in the enum but not in this graph, so it must not appear.
    assert "dbo:Organisation" not in context["entity_hints"]


def test_an_invented_type_gets_an_empty_hint_list(backend):
    """
    Open-vocabulary extraction means most types have no enum entry at all.

    They must still reach the model — with an empty hint list, which the prompt
    already defines as "search for it yourself" — rather than being dropped and
    leaving the grounder unsure whether the type was withheld.
    """
    invented = KnowledgeGraph(
        entities=[
            Entity(id="e1", name="Rick Scott", type="POLITICIAN"),
            Entity(id="e2", name="Florida", type="LOCATION"),
        ],
        relations=[Relation(source="e1", target="e2", relation_type="GOVERNOR_OF")],
    )
    agent = make_agent(ScriptedLLM([], GroundingDecisions()), backend)
    context = agent.build_context(GroundingTask(graph=invented))

    entity_table = json.loads(context["entity_hints"])
    relation_table = json.loads(context["relation_hints"])
    assert entity_table["POLITICIAN"] == []
    assert entity_table["LOCATION"] == ["dbo:Place", "dbo:Settlement", "dbo:Country"]
    assert relation_table["GOVERNOR_OF"] == []

    # And it still renders — an unknown type is not a prompt failure.
    rendered = PromptRegistry(default_version="v2").render_pair(
        "grounder", version="v2", **context
    )
    assert "GOVERNOR_OF" in rendered.system
