"""
The structural validator inside the extract -> grade loop, offline with scripted
LLMs and validator. The real GNNValidator test is skipped without a checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.graph import NODE_VALIDATE, build_dependencies, build_graph
from kg_agentic_extraction.grounding.base import NullGroundingBackend
from kg_agentic_extraction.models.grading import SimpleSchemaGraderReport
from kg_agentic_extraction.models.knowledge_graph import Entity, KnowledgeGraph, Relation
from kg_agentic_extraction.models.validation import FlaggedElement, ValidationReport
from kg_agentic_extraction.nodes.routing import make_validator_veto
from kg_agentic_extraction.nodes.validate_node import make_validate_node
from kg_agentic_extraction.runner import run_pipeline

DOCUMENTS = ["Rick Scott is the governor of Florida."]

GRAPH = KnowledgeGraph(
    entities=[
        Entity(id="e1", name="Rick Scott", type="PERSON"),
        Entity(id="e2", name="Florida", type="LOCATION"),
    ],
    relations=[Relation(source="e1", target="e2", relation_type="GOVERNOR_OF")],
)

FLAG = FlaggedElement(
    kind="relation",
    element_id="e1|GOVERNOR_OF|e2",
    description="Rick Scott -[GOVERNOR_OF]-> Florida",
    score=0.9,
)


class ScriptedLLM:
    """Extractor calls get GRAPH; grader calls get the next scripted report."""

    def __init__(self, grader_reports: list[SimpleSchemaGraderReport]) -> None:
        self._reports = list(grader_reports)
        self.grader_prompts: list[str] = []
        self.extractor_prompts: list[str] = []

    def structured(self, *, system: str, user: str, schema: type):
        if schema is KnowledgeGraph:
            self.extractor_prompts.append(user)
            return GRAPH
        self.grader_prompts.append(system + "\n" + user)
        return self._reports.pop(0)


class ScriptedValidator:
    def __init__(self, consistency: float, flagged: list[FlaggedElement]) -> None:
        self.consistency, self.flagged, self.calls = consistency, flagged, 0

    def validate(self, graph, *, iteration):  # noqa: ARG002
        self.calls += 1
        return ValidationReport(
            iteration=iteration, scores={"consistency": self.consistency}, flagged=self.flagged
        )


def _run(mode: str, llm: ScriptedLLM, validator=None, **overrides):
    settings = PipelineSettings(
        gnn_mode=mode, grounding_enabled=False, max_iterations=4, worker_keys=[], **overrides
    )
    deps = build_dependencies(
        settings, llm=llm, backend=NullGroundingBackend(), validator=validator
    )
    app = build_graph(deps)
    return app, run_pipeline(DOCUMENTS, settings=settings, graph=app)


CONVERGED = SimpleSchemaGraderReport(converged=True, issues_markdown="")


def test_off_mode_has_no_validate_node_and_no_flags_in_the_prompt():
    llm = ScriptedLLM([CONVERGED])
    validator = ScriptedValidator(0.1, [FLAG])
    app, result = _run("off", llm, validator)
    assert NODE_VALIDATE not in app.get_graph().nodes
    assert validator.calls == 0
    assert "STRUCTURAL VALIDATOR FLAGS" not in llm.grader_prompts[0]
    assert result.validation_history == []


def test_advise_mode_shows_the_grader_the_flags_but_never_overrules_it():
    llm = ScriptedLLM([CONVERGED])
    validator = ScriptedValidator(0.1, [FLAG])
    app, result = _run("advise", llm, validator)
    assert NODE_VALIDATE in app.get_graph().nodes
    assert "STRUCTURAL VALIDATOR FLAGS" in llm.grader_prompts[0]
    assert "e1|GOVERNOR_OF|e2" in llm.grader_prompts[0]
    assert result.converged and result.iterations == 1
    assert result.validator_vetoes == 0
    assert [r.consistency for r in result.validation_history] == [0.1]


def test_veto_mode_forces_one_repair_round_carrying_the_flags():
    llm = ScriptedLLM([CONVERGED, CONVERGED])
    validator = ScriptedValidator(0.1, [FLAG])
    _, result = _run("veto", llm, validator)
    assert result.iterations == 2  # one veto, then gnn_max_vetoes (1) is spent
    assert result.converged
    assert result.validator_vetoes == 1
    repair_prompt = llm.extractor_prompts[1]
    assert "e1|GOVERNOR_OF|e2" in repair_prompt
    assert len(result.validation_history) == 2


def test_veto_mode_does_not_fire_on_a_graph_scored_consistent():
    llm = ScriptedLLM([CONVERGED])
    _, result = _run("veto", llm, ScriptedValidator(0.9, [FLAG]))
    assert result.iterations == 1 and result.validator_vetoes == 0


def test_a_failing_validator_costs_the_round_its_hints_not_the_run():
    class Broken:
        def validate(self, graph, *, iteration):  # noqa: ARG002
            raise RuntimeError("no checkpoint")

    llm = ScriptedLLM([CONVERGED])
    _, result = _run("veto", llm, Broken())
    assert result.succeeded and result.converged
    assert any("validate[1]" in e for e in result.errors)
    assert "STRUCTURAL VALIDATOR FLAGS" not in llm.grader_prompts[0]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({"validation_report": None}, False),
        (
            {"validation_report": ValidationReport(iteration=1, scores={"consistency": 0.1})},
            False,
        ),  # nothing flagged
        (
            {
                "validation_report": ValidationReport(
                    iteration=1, scores={"consistency": 0.1}, flagged=[FLAG]
                ),
                "iteration": 1,
            },
            True,
        ),
        (
            {
                "validation_report": ValidationReport(
                    iteration=1, scores={"consistency": 0.1}, flagged=[FLAG]
                ),
                "iteration": 1,
                "validator_vetoes": 1,
            },
            False,
        ),
        (
            {
                "validation_report": ValidationReport(
                    iteration=4, scores={"consistency": 0.1}, flagged=[FLAG]
                ),
                "iteration": 4,
            },
            False,
        ),  # cap
    ],
)
def test_veto_rule(state, expected):
    veto = make_validator_veto(threshold=0.3, max_vetoes=1, max_iterations=4)
    assert veto(state) is expected


def test_validate_node_clears_a_stale_report_when_there_is_no_graph():
    node = make_validate_node(ScriptedValidator(0.5, []))
    assert node({"knowledge_graph": None, "iteration": 2}) == {"validation_report": None}


CHECKPOINT = Path("data/gnn_checkpoints_gcs")


@pytest.mark.skipif(
    not (CHECKPOINT / "best_model.pt").is_file(), reason="no trained GNN checkpoint on disk"
)
def test_real_gnn_validator_scores_a_graph():
    from kg_agentic_extraction.validation.gnn import GNNValidator

    report = GNNValidator(CHECKPOINT, top_k=3, min_score=0.0).validate(GRAPH, iteration=1)
    assert 0.0 <= report.consistency <= 1.0
    assert len(report.flagged) == 3  # 2 entities + 1 relation, all above min_score=0
    assert {f.element_id for f in report.flagged} == {"e1", "e2", "e1|GOVERNOR_OF|e2"}
    assert report.flagged == sorted(report.flagged, key=lambda f: -f.score)


def test_a_validator_mode_refuses_a_grader_prompt_that_cannot_show_its_flags():
    settings = PipelineSettings(
        gnn_mode="advise", grader_prompt_version="v5", grounding_enabled=False, worker_keys=[]
    )
    with pytest.raises(ValueError, match="KG_GRADER_PROMPT_VERSION"):
        build_dependencies(
            settings,
            llm=ScriptedLLM([]),
            backend=NullGroundingBackend(),
            validator=ScriptedValidator(0.5, []),
        )
