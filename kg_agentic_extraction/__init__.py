"""
kg_agentic_extraction — LangGraph-orchestrated knowledge-graph extraction.

Three agents:

- **extractor** — documents → evidence-traced `KnowledgeGraph`.
- **grader**    — graph + documents → typed issue report, rendered to Markdown.
- **grounder**  — refined graph → entities/relations mapped onto DBpedia via a
  standalone MCP server (`mcp_servers/dbpedia/`).

Extractor and grader loop until the grader reports no issues or the iteration
cap is reached; the grounder then runs once.

Layering, innermost first — each layer may import the ones above it, never below:

    models/      pure Pydantic contracts
    prompts/     versioned Jinja2 templates + model→text renderers
    llm/         the LLMClient port and its provider adapters
    grounding/   the GroundingBackend port and its MCP adapter
    agents/      framework-agnostic agents, one per file
    nodes/       adapters binding agents to graph state
    graph.py     composition root and topology
    runner.py    public entrypoint / CLI

Re-exports are resolved lazily. Importing this package therefore costs nothing
beyond the module itself — LangGraph is not pulled in until you actually ask for
`build_graph` — and `python -m kg_agentic_extraction.runner` does not trip the
double-import warning that an eager `from .runner import ...` would cause.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import-time only for type checkers; never at runtime
    from kg_agentic_extraction.config import PipelineSettings
    from kg_agentic_extraction.graph import (
        PipelineDependencies,
        build_dependencies,
        build_graph,
    )
    from kg_agentic_extraction.runner import PipelineResult, run_pipeline
    from kg_agentic_extraction.state import PipelineState, initial_state

__all__ = [
    "PipelineDependencies",
    "PipelineResult",
    "PipelineSettings",
    "PipelineState",
    "build_dependencies",
    "build_graph",
    "initial_state",
    "run_pipeline",
]

# Public name → the module that defines it.
_EXPORTS = {
    "PipelineSettings": "kg_agentic_extraction.config",
    "PipelineDependencies": "kg_agentic_extraction.graph",
    "build_dependencies": "kg_agentic_extraction.graph",
    "build_graph": "kg_agentic_extraction.graph",
    "PipelineResult": "kg_agentic_extraction.runner",
    "run_pipeline": "kg_agentic_extraction.runner",
    "PipelineState": "kg_agentic_extraction.state",
    "initial_state": "kg_agentic_extraction.state",
}


def __getattr__(name: str) -> object:
    """Resolve a public export on first access (PEP 562)."""
    if (module := _EXPORTS.get(name)) is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # cache, so this runs once per name
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
