"""
The v2 prompts must not carry an ontology into extraction or grading.

These are regression tests for a property that is easy to reintroduce by
accident: a template variable added back, or an agent handed an `OntologyConfig`
"just for context". Both would restore the constraint silently — the pipeline
would still run, and only the graphs would get worse.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from kg_agentic_extraction.agents.extractor_agent import ExtractionTask, ExtractorAgent
from kg_agentic_extraction.agents.grader_agent import GraderAgent, GradingTask
from kg_agentic_extraction.models.knowledge_graph import Entity, KnowledgeGraph, Relation
from kg_agentic_extraction.models.ontology import EntityType, RelationType
from kg_agentic_extraction.prompts.registry import PromptRegistry

DOCUMENTS = ["Rick Scott of Florida is a Republican governor."]

#: Named in the v2 extractor prompt on purpose — as the anti-pattern to avoid,
#: not as a permitted type. Excluded so the guard below does not flag guidance.
DELIBERATELY_NAMED = {"RELATED_TO"}

#: Type names that only an injected ontology would put in a prompt. Drawn from
#: the enums rather than written out, so extending an enum extends this guard.
ONTOLOGY_ONLY_TYPES = [
    t.value for t in (*EntityType, *RelationType) if t.value not in DELIBERATELY_NAMED
]

#: The v1 templates' injection markers. Their absence is the structural check;
#: the type-name sweep below is the belt to that pair of braces.
INJECTION_MARKERS = ("ALLOWED ENTITY TYPES", "ALLOWED RELATION TYPES", "ONTOLOGY IN FORCE")


def leaked_types(prompt: str, *, allow: set[str] = frozenset()) -> list[str]:
    """
    Ontology type names appearing in `prompt` as whole words.

    Whole-word matching is not fussiness: `CANDIDATE_FOR` contains `DATE`, and a
    substring check flags a prompt that is doing exactly the right thing.
    """
    return [
        t
        for t in ONTOLOGY_ONLY_TYPES
        if t not in allow and re.search(rf"(?<![A-Z_]){re.escape(t)}(?![A-Z_])", prompt)
    ]


@pytest.fixture
def registry():
    return PromptRegistry(default_version="v2")


@pytest.fixture
def graph():
    return KnowledgeGraph(
        entities=[
            Entity(id="e1", name="Rick Scott", type="POLITICIAN"),
            Entity(id="e2", name="Florida", type="LOCATION"),
        ],
        relations=[Relation(source="e1", target="e2", relation_type="GOVERNOR_OF")],
    )


def test_extraction_task_has_no_ontology_field():
    """The constraint cannot be passed even by a caller who wants to."""
    fields = {f.name for f in dataclasses.fields(ExtractionTask)}
    assert "ontology" not in fields
    with pytest.raises(TypeError):
        ExtractionTask(documents=DOCUMENTS, ontology="anything")  # type: ignore[call-arg]


def test_grading_task_has_no_ontology_field():
    fields = {f.name for f in dataclasses.fields(GradingTask)}
    assert "ontology" not in fields


def test_extractor_prompt_names_no_types(registry):
    agent = ExtractorAgent.__new__(ExtractorAgent)  # no LLM needed to build context
    context = agent.build_context(ExtractionTask(documents=DOCUMENTS))
    rendered = registry.render_pair("extractor", version="v2", **context)
    prompt = rendered.system + rendered.user

    for marker in INJECTION_MARKERS:
        assert marker not in prompt
    assert not (leaked := leaked_types(prompt)), f"ontology types leaked: {leaked}"


def test_extractor_repair_prompt_names_no_types(registry, graph):
    """The repair pass renders a different user template and must be just as clean."""
    agent = ExtractorAgent.__new__(ExtractorAgent)
    task = ExtractionTask(
        documents=DOCUMENTS,
        previous_graph=graph,
        grader_markdown="## Issues\n- retype something",
    )
    assert task.is_repair
    rendered = registry.render_pair(
        "extractor", user_role="repair", version="v2", **agent.build_context(task)
    )
    prompt = rendered.system + rendered.user

    for marker in INJECTION_MARKERS:
        assert marker not in prompt
    # `LOCATION` is echoed back inside `previous_graph_json` because the
    # extractor itself chose it last round — that is the graph, not an ontology.
    leaked = leaked_types(prompt, allow={"LOCATION"})
    assert not leaked, f"ontology types leaked into the repair prompt: {leaked}"


def test_grader_prompt_shows_the_graphs_vocabulary_not_the_enum(registry, graph):
    agent = GraderAgent.__new__(GraderAgent)
    context = agent.build_context(GradingTask(graph=graph, documents=DOCUMENTS))
    rendered = registry.render_pair("grader", version="v2", **context)

    assert context["entity_type_vocabulary"] == ["LOCATION", "POLITICIAN"]
    assert context["relation_type_vocabulary"] == ["GOVERNOR_OF"]
    # The invented vocabulary reaches the grader...
    assert "GOVERNOR_OF" in rendered.system
    assert "POLITICIAN" in rendered.system
    # ...and the enum-only types it never used do not.
    for unused in ("AFFILIATED_WITH", "PARTICIPATED_IN", "ORGANIZATION"):
        assert unused not in rendered.system


def test_v1_prompts_still_render_for_comparison(registry):
    """
    v1 is kept on disk so the constrained regime stays reproducible. It takes
    the ontology variables v2 dropped, so rendering it proves the old context
    shape was not broken by the refactor.
    """
    v1 = registry.render_pair(
        "extractor",
        version="v1",
        documents_text="[DOCUMENT 0]\nx",
        document_count=1,
        domain_context="general news articles",
        entity_types=["PERSON", "LOCATION"],
        relation_types=["LOCATED_IN"],
    )
    assert "ALLOWED ENTITY TYPES: PERSON, LOCATION" in v1.system
