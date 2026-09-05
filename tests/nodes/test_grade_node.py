"""
The grade node and the loop it feeds, under the v4 (Markdown) grader.

Convergence used to be derived — `not report.issues`, on a typed object the node
never had to trust. It is now a flag the model sets for itself, so these tests
pin the two things that replaced that guarantee: the flag reaches the router
intact, and the states the model can now express but could not before are all
handled somewhere rather than looping to the iteration cap.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.agents.grader_agent import GraderAgent, GradingTask
from kg_agentic_extraction.models.grading import MistralGraderReport
from kg_agentic_extraction.models.knowledge_graph import Entity, KnowledgeGraph, Relation
from kg_agentic_extraction.nodes.grade_node import make_grade_node
from kg_agentic_extraction.nodes.routing import make_loop_router
from kg_agentic_extraction.prompts.registry import PromptRegistry
from kg_agentic_extraction.state import PipelineState

DOCUMENTS = ["Rick Scott of Florida is a Republican governor."]
REPORT = "## High severity\n\n- **WRONG_TYPE** `e1|LOCATED_IN|e2` — retype it.\n"


class FakeLLM:
    """Returns a scripted report, or raises. Records what it was asked."""

    def __init__(self, result: object) -> None:
        self._result = result
        self.calls: list[dict] = []

    def structured(self, *, system: str, user: str, schema: type):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


@pytest.fixture
def graph():
    return KnowledgeGraph(
        entities=[
            Entity(id="e1", name="Rick Scott", type="POLITICIAN"),
            Entity(id="e2", name="Florida", type="LOCATION"),
        ],
        relations=[Relation(source="e1", target="e2", relation_type="GOVERNOR_OF")],
    )


def grade_with(result: object, graph: KnowledgeGraph, *, iteration: int = 1):
    """Run the grade node over `graph` with a grader whose model returns `result`."""
    llm = FakeLLM(result)
    agent = GraderAgent(llm=llm, prompts=PromptRegistry(default_version="v4"), prompt_version="v4")
    node = make_grade_node(agent, max_documents=10)
    state = PipelineState(documents=DOCUMENTS, knowledge_graph=graph, iteration=iteration)
    return node(state), llm


def route(state: PipelineState, *, max_iterations: int = 6) -> str:
    return make_loop_router(max_iterations=max_iterations, grounding_enabled=False)(state)


# ── The happy paths ───────────────────────────────────────────────────


def test_a_converged_report_ends_the_loop(graph):
    state, _ = grade_with(MistralGraderReport(converged=True), graph)

    assert state["converged"] is True
    assert route({**state, "iteration": 1}) == "end"


def test_an_unconverged_report_sends_the_graph_back(graph):
    state, _ = grade_with(MistralGraderReport(converged=False, issues_markdown=REPORT), graph)

    assert state["converged"] is False
    assert route({**state, "iteration": 1}) == "refine"


def test_the_models_markdown_reaches_the_extractor_verbatim(graph):
    """The node must not reformat the report — it is the repair prompt's payload."""
    state, _ = grade_with(MistralGraderReport(converged=False, issues_markdown=REPORT), graph)

    assert state["grader_markdown"] == REPORT.strip()


def test_the_report_is_kept_in_history(graph):
    report = MistralGraderReport(converged=False, issues_markdown=REPORT)
    state, _ = grade_with(report, graph)

    assert state["grader_report"] is report
    assert state["grader_reports"] == [report]


def test_the_grader_is_asked_for_the_new_schema(graph):
    _, llm = grade_with(MistralGraderReport(converged=True), graph)

    assert llm.calls[0]["schema"] is MistralGraderReport


# ── States only the free-text schema can express ──────────────────────


def test_unconverged_with_an_empty_report_is_treated_as_converged(graph):
    """
    The typed schema could not express this — an unconverged report had issues in
    it by definition. Left alone it costs a full repair round with no instructions
    in the prompt, repeated to the iteration cap.
    """
    state, _ = grade_with(MistralGraderReport(converged=False, issues_markdown="   \n  "), graph)

    assert state["converged"] is True
    assert route({**state, "iteration": 1}) == "end"


def test_converged_wins_over_a_contradictory_report_body(graph):
    """A grader that says 'done' and then lists issues is not worth another round."""
    state, _ = grade_with(MistralGraderReport(converged=True, issues_markdown=REPORT), graph)

    assert state["converged"] is True
    assert route({**state, "iteration": 1}) == "end"


def test_surrounding_whitespace_is_stripped(graph):
    state, _ = grade_with(
        MistralGraderReport(converged=False, issues_markdown=f"\n\n{REPORT}\n\n"), graph
    )

    assert state["grader_markdown"] == REPORT.strip()


# ── Failure paths ─────────────────────────────────────────────────────


def test_a_grader_that_raises_does_not_wedge_the_loop(graph):
    state, _ = grade_with(RuntimeError("provider exploded"), graph)

    assert state["converged"] is True
    assert "provider exploded" in state["errors"][0]
    assert route({**state, "iteration": 1}) == "end"


def test_no_graph_to_grade_ends_the_loop():
    node = make_grade_node(
        GraderAgent(
            llm=FakeLLM(MistralGraderReport(converged=True)),
            prompts=PromptRegistry(default_version="v4"),
        ),
        max_documents=10,
    )

    state = node(PipelineState(documents=DOCUMENTS, knowledge_graph=None, iteration=1))

    assert state["converged"] is True
    assert "no graph" in state["errors"][0]


def test_the_iteration_cap_still_ends_an_unconverged_run(graph):
    state, _ = grade_with(
        MistralGraderReport(converged=False, issues_markdown=REPORT), graph, iteration=6
    )

    assert state["converged"] is False
    assert route({**state, "iteration": 6}, max_iterations=6) == "end"


# ── The v4 prompt ─────────────────────────────────────────────────────


def test_the_v4_prompt_states_the_convergence_rule(graph):
    """
    `converged` is no longer derived from an empty list, so the prompt is the
    only place the rule exists. If this drifts, the loop stops early or never.
    """
    agent = GraderAgent.__new__(GraderAgent)
    rendered = PromptRegistry(default_version="v4").render_pair(
        "grader", version="v4", **agent.build_context(GradingTask(graph=graph, documents=DOCUMENTS))
    )

    assert "converged: true" in rendered.system
    assert "converged: false" in rendered.system
    assert "issues_markdown" in rendered.system


def test_the_v4_grader_still_sees_the_graphs_own_vocabulary(graph):
    agent = GraderAgent.__new__(GraderAgent)
    rendered = PromptRegistry(default_version="v4").render_pair(
        "grader", version="v4", **agent.build_context(GradingTask(graph=graph, documents=DOCUMENTS))
    )

    assert "GOVERNOR_OF" in rendered.system
    assert "POLITICIAN" in rendered.system
