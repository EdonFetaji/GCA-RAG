"""
The pipeline graph.

    START → extract → [validate] → grade ─┬─ refine ──→ extract     (loop)
                                          ├─ ground ──→ END
                                          └─ end ─────→ END

`validate` (the GNN structural validator) is only added when `gnn_mode` is not "off".

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
    make_validate_node,
    make_validator_veto,
)
from kg_agentic_extraction.prompts.registry import PromptRegistry
from kg_agentic_extraction.state import PipelineState
from kg_agentic_extraction.validation.base import GraphValidator

logger = logging.getLogger(__name__)

# Node names. Constants rather than string literals so a typo is an import
# error instead of an orphaned node that silently never runs.
NODE_EXTRACT = "extract"
NODE_GRADE = "grade"
NODE_GROUND = "ground"
NODE_VALIDATE = "validate"


@dataclass
class PipelineDependencies:
    """Everything the graph needs, already constructed."""

    extractor: ExtractorAgent
    grader: GraderAgent
    grounder: GrounderAgent
    settings: PipelineSettings
    backend: GroundingBackend
    #: None when `gnn_mode` is "off".
    validator: GraphValidator | None = None


def build_dependencies(
    settings: PipelineSettings | None = None,
    *,
    llm: LLMClient | None = None,
    backend: GroundingBackend | None = None,
    prompts: PromptRegistry | None = None,
    validator: GraphValidator | None = None,
) -> PipelineDependencies:
    """
    Construct the agents and their collaborators.

    Every collaborator is injectable. Passing `llm` and `backend` explicitly is
    the supported way to run the pipeline against test doubles — nothing below
    this function reads configuration or opens a connection on its own. An
    explicit `llm` binds *every* agent to that one client, which is what makes a
    fake usable here; leave it None to let each role resolve its own provider.

    Roles are resolved through `settings.for_role()`, which hands the factory a
    settings copy with that role's provider and model swapped in. The extractor
    and the grader therefore run on different vendors — whichever pair
    `KG_EXTRACTOR_PROVIDER` and `KG_GRADER_PROVIDER` name — with no change to
    `llm/factory.py` and no provider-awareness anywhere below here.
    """
    settings = settings or PipelineSettings()
    prompts = prompts or PromptRegistry(default_version=settings.prompt_version)

    if llm is None:
        from kg_agentic_extraction.llm.factory import build_llm

        extractor_llm = build_llm(settings.for_role("extractor"))
        grader_llm = build_llm(settings.for_role("grader"))
        # The grounder has no role binding of its own: it is disabled in batch
        # mode, which is the only place the split matters, and a single-cluster
        # run wants it on whatever `llm_provider` names. It needs tool calling,
        # so it gets the extractor's client — Gemini's tool loop is the tested one.
        grounder_llm: LLMClient = extractor_llm
    else:
        extractor_llm = grader_llm = grounder_llm = llm

    if backend is None:
        if settings.grounding_enabled:
            from kg_agentic_extraction.grounding.mcp_backend import MCPGroundingBackend

            backend = MCPGroundingBackend(
                url=settings.mcp_url,
                timeout_seconds=settings.grounding_tool_timeout_seconds,
            )
        else:
            backend = NullGroundingBackend()

    if settings.gnn_mode != "off" and not _grader_reads_validator_flags(settings):
        # An older template silently drops the flags, turning `advise` into `off`.
        raise ValueError(
            f"KG_GNN_MODE={settings.gnn_mode} needs a grader prompt that shows the validator's "
            f"flags (v6+), but KG_GRADER_PROMPT_VERSION={settings.grader_prompt_version}. "
            "Set KG_GRADER_PROMPT_VERSION=v6 — without flags it renders exactly as v5."
        )

    if validator is None and settings.gnn_mode != "off":
        from kg_agentic_extraction.validation.gnn import GNNValidator

        validator = GNNValidator(
            settings.gnn_checkpoint_dir,
            top_k=settings.gnn_top_k,
            min_score=settings.gnn_min_score,
        )

    return PipelineDependencies(
        extractor=ExtractorAgent(
            llm=extractor_llm,
            prompts=prompts,
            prompt_version=settings.prompt_version,
        ),
        # The grader pins its own prompt version, for the reason the grounder
        # already did: its templates moved to v4 (model-written Markdown) while
        # the extractor stayed on v3, and one shared version would have forced a
        # no-op v4 of the extractor just to move the grader.
        grader=GraderAgent(
            llm=grader_llm,
            prompts=prompts,
            prompt_version=settings.grader_prompt_version,
        ),
        # The grounder pins its own prompt version and needs a tool-calling
        # client; both are why it does not simply mirror the two above.
        grounder=GrounderAgent(
            backend=backend,
            max_tool_rounds=settings.grounding_max_tool_rounds,
            llm=grounder_llm,
            prompts=prompts,
            prompt_version=settings.grounder_prompt_version,
        ),
        settings=settings,
        backend=backend,
        validator=validator if settings.gnn_mode != "off" else None,
    )


def _grader_reads_validator_flags(settings: PipelineSettings) -> bool:
    """Whether the configured grader template renders `gnn_flags` (v6 onwards)."""
    version = settings.grader_prompt_version.lstrip("vV")
    return version.isdigit() and int(version) >= 6


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
    veto = None
    if deps.validator is not None and cfg.gnn_mode == "veto":
        veto = make_validator_veto(
            threshold=cfg.gnn_veto_threshold,
            max_vetoes=cfg.gnn_max_vetoes,
            max_iterations=cfg.max_iterations,
        )
    builder.add_node(
        NODE_GRADE,
        make_grade_node(deps.grader, max_documents=cfg.max_documents, veto=veto),
    )
    builder.add_node(
        NODE_GROUND,
        make_ground_node(deps.grounder, ontology=cfg.ontology, backend=deps.backend),
    )

    builder.add_edge(START, NODE_EXTRACT)
    if deps.validator is not None:
        builder.add_node(NODE_VALIDATE, make_validate_node(deps.validator))
        builder.add_edge(NODE_EXTRACT, NODE_VALIDATE)
        builder.add_edge(NODE_VALIDATE, NODE_GRADE)
    else:
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
        "graph built — max_iterations=%d grounding=%s validator=%s",
        cfg.max_iterations,
        cfg.grounding_enabled,
        cfg.gnn_mode if deps.validator is not None else "off",
    )
    return builder.compile(checkpointer=checkpointer)
