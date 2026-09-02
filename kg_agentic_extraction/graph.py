"""
The pipeline graph.

    START → extract → grade ─┬─ refine ──→ extract     (loop)
                             ├─ ground ──→ END
                             └─ end ─────→ END

This file does two things and nothing else: assemble the collaborators
(composition root) and declare the topology. There is no extraction logic, no
prompt text, and no LLM call here — adding a stage should mean adding a node
factory in `nodes/` and one `add_node`/`add_edge` pair below.

`build_dependencies` is separated from `build_graph` so the wiring can be
exercised with fakes: hand `build_graph` a `PipelineDependencies` built from
stub clients and the whole topology runs offline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from kg_agentic_extraction.agents.extractor_agent import ExtractorAgent
from kg_agentic_extraction.agents.grader_agent import GraderAgent
from kg_agentic_extraction.agents.grounder_agent import GrounderAgent
from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.grounding.base import GroundingBackend, NullGroundingBackend
from kg_agentic_extraction.llm.base import LLMClient
from kg_agentic_extraction.nodes import (
    make_extract_node,
    make_grade_node,
    make_ground_node,
    make_loop_router,
)
from kg_agentic_extraction.prompts.registry import PromptRegistry
from kg_agentic_extraction.state import PipelineState

logger = logging.getLogger(__name__)

# Node names. Constants rather than string literals so a typo is an import
# error instead of an orphaned node that silently never runs.
NODE_EXTRACT = "extract"
NODE_GRADE = "grade"
NODE_GROUND = "ground"


@dataclass
class PipelineDependencies:
    """Everything the graph needs, already constructed."""

    extractor: ExtractorAgent
    grader: GraderAgent
    grounder: GrounderAgent
    settings: PipelineSettings
    backend: GroundingBackend


def build_dependencies(
    settings: PipelineSettings | None = None,
    *,
    llm: LLMClient | None = None,
    backend: GroundingBackend | None = None,
    prompts: PromptRegistry | None = None,
) -> PipelineDependencies:
    """
    Construct the agents and their collaborators.

    Every collaborator is injectable. Passing `llm` and `backend` explicitly is
    the supported way to run the pipeline against test doubles — nothing below
    this function reads configuration or opens a connection on its own.
    """
    settings = settings or PipelineSettings()
    prompts = prompts or PromptRegistry(default_version=settings.prompt_version)

    if llm is None:
        from kg_agentic_extraction.llm.factory import build_llm

        llm = build_llm(settings)

    if backend is None:
        if settings.grounding_enabled:
            from kg_agentic_extraction.grounding.mcp_backend import MCPGroundingBackend

            backend = MCPGroundingBackend(
                url=settings.mcp_url,
                timeout_seconds=settings.grounding_tool_timeout_seconds,
            )
        else:
            backend = NullGroundingBackend()

    shared = {"llm": llm, "prompts": prompts, "prompt_version": settings.prompt_version}
    return PipelineDependencies(
        extractor=ExtractorAgent(**shared),
        grader=GraderAgent(**shared),
        # The grounder pins its own prompt version and needs a tool-calling
        # client; both are why it does not simply take `**shared`.
        grounder=GrounderAgent(
            backend=backend,
            max_tool_rounds=settings.grounding_max_tool_rounds,
            llm=llm,
            prompts=prompts,
            prompt_version=settings.grounder_prompt_version,
        ),
        settings=settings,
        backend=backend,
    )


def build_graph(
    deps: PipelineDependencies | None = None,
    *,
    settings: PipelineSettings | None = None,
    checkpointer: object | None = None,
):
    """
    Wire and compile the extraction graph.

    Returns a compiled LangGraph app; invoke it with `state.initial_state(...)`.
    Pass `checkpointer` to persist state between steps (durability, resuming a
    long run, human-in-the-loop pauses) — the topology is unaffected.
    """
    deps = deps or build_dependencies(settings)
    cfg = deps.settings

    builder = StateGraph(PipelineState)

    # Only the grounder receives `cfg.ontology`. Extraction and grading run
    # open-vocabulary: the extractor names types from the documents, and the
    # grader audits those names for coherence rather than for membership of a
    # list. See docs/adr/0004-open-vocabulary-extraction.md.
    builder.add_node(
        NODE_EXTRACT,
        make_extract_node(
            deps.extractor,
            domain_context=cfg.ontology.domain_context,
            max_documents=cfg.max_documents,
        ),
    )
    builder.add_node(
        NODE_GRADE,
        make_grade_node(deps.grader, max_documents=cfg.max_documents),
    )
    builder.add_node(
        NODE_GROUND,
        make_ground_node(deps.grounder, ontology=cfg.ontology, backend=deps.backend),
    )

    builder.add_edge(START, NODE_EXTRACT)
    builder.add_edge(NODE_EXTRACT, NODE_GRADE)

    # The loop: grade decides whether to send the graph back for repair.
    builder.add_conditional_edges(
        NODE_GRADE,
        make_loop_router(
            max_iterations=cfg.max_iterations,
            grounding_enabled=cfg.grounding_enabled,
        ),
        {"refine": NODE_EXTRACT, "ground": NODE_GROUND, "end": END},
    )
    builder.add_edge(NODE_GROUND, END)

    logger.debug(
        "graph built — max_iterations=%d grounding=%s",
        cfg.max_iterations,
        cfg.grounding_enabled,
    )
    return builder.compile(checkpointer=checkpointer)
