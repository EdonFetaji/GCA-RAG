"""
The LangGraph channel schema.

One state object flows through every node. Fields fall into three groups:
inputs set once at invocation, working values the loop rewrites each round, and
history channels that accumulate.

`grader_reports` and `errors` use `operator.add` reducers so each node returns
only its own contribution and LangGraph appends — nodes never read-modify-write
a list, which is what keeps them safe to reorder or run concurrently later.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from kg_agentic_extraction.models.grading import SimpleSchemaGraderReport
from kg_agentic_extraction.models.grounding import GroundedKnowledgeGraph
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.validation import ValidationReport


class PipelineState(TypedDict, total=False):
    """State threaded through the extraction graph."""

    # ── Inputs (set at invocation, never rewritten) ───────────────────
    documents: list[str]
    cluster_index: int | None

    # ── Working values (rewritten each loop iteration) ────────────────
    knowledge_graph: KnowledgeGraph | None
    grader_report: SimpleSchemaGraderReport | None
    #: The grader's Markdown, rendered from `grader_report` — the artifact, and
    #: the text fed back into the extractor's repair prompt. Kept as its own
    #: channel because that is the half the extractor's template consumes.
    grader_markdown: str | None
    iteration: int
    converged: bool
    #: The structural validator's verdict on the current graph (None when off).
    validation_report: ValidationReport | None
    #: Times the validator has overruled a converged grader this run.
    validator_vetoes: int

    # ── Output ────────────────────────────────────────────────────────
    grounded_graph: GroundedKnowledgeGraph | None

    # ── History (accumulated) ─────────────────────────────────────────
    grader_reports: Annotated[list[SimpleSchemaGraderReport], operator.add]
    validation_reports: Annotated[list[ValidationReport], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(documents: list[str], *, cluster_index: int | None = None) -> PipelineState:
    """
    Build the starting state.

    Every channel is initialized explicitly, including the ones nodes will fill
    in later, so that a node reading a field before it is written gets `None`
    rather than a `KeyError`.
    """
    return PipelineState(
        documents=documents,
        cluster_index=cluster_index,
        knowledge_graph=None,
        grader_report=None,
        grader_markdown=None,
        iteration=0,
        converged=False,
        validation_report=None,
        validator_vetoes=0,
        grounded_graph=None,
        grader_reports=[],
        validation_reports=[],
        errors=[],
    )
