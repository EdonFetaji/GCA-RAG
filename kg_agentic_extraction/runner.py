"""
Public entrypoint.

Wraps "build the graph, invoke it, read the state back out" so callers do not
have to know the state schema. Also the CLI: `python -m kg_agentic_extraction.runner`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.graph import build_dependencies, build_graph
from kg_agentic_extraction.models.grading import GraderReport
from kg_agentic_extraction.models.grounding import GroundedKnowledgeGraph
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.state import initial_state
from kg_agentic_extraction.storage import graph_filename, save_knowledge_graph

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """The outcome of one run, flattened out of the graph state."""

    knowledge_graph: KnowledgeGraph | None
    grounded_graph: GroundedKnowledgeGraph | None
    grader_report: GraderReport | None
    grader_markdown: str | None
    grader_history: list[GraderReport]
    iterations: int
    converged: bool
    errors: list[str]

    @property
    def succeeded(self) -> bool:
        """A graph was produced. Note this can be true while `converged` is false."""
        return self.knowledge_graph is not None

    def to_json(self, *, indent: int = 2) -> str:
        payload = {
            "converged": self.converged,
            "iterations": self.iterations,
            "errors": self.errors,
            "knowledge_graph": (
                self.knowledge_graph.model_dump(mode="json") if self.knowledge_graph else None
            ),
            "grounded_graph": (
                self.grounded_graph.model_dump(mode="json") if self.grounded_graph else None
            ),
            "grader_history": [r.model_dump(mode="json") for r in self.grader_history],
        }
        return json.dumps(payload, indent=indent)


def run_pipeline(
    documents: list[str],
    *,
    settings: PipelineSettings | None = None,
    cluster_index: int | None = None,
    graph: object | None = None,
) -> PipelineResult:
    """
    Run extraction → grading loop → grounding over one document cluster.

    Pass a pre-built `graph` when running many clusters, so the agents and the
    compiled topology are constructed once rather than per cluster.
    """
    if not documents:
        raise ValueError("run_pipeline() needs at least one document")

    settings = settings or PipelineSettings()
    app = graph or build_graph(settings=settings)

    final = app.invoke(initial_state(documents, cluster_index=cluster_index))  # type: ignore[attr-defined]

    return PipelineResult(
        knowledge_graph=final.get("knowledge_graph"),
        grounded_graph=final.get("grounded_graph"),
        grader_report=final.get("grader_report"),
        grader_markdown=final.get("grader_markdown"),
        grader_history=final.get("grader_reports", []),
        iterations=final.get("iteration", 0),
        converged=final.get("converged", False),
        errors=final.get("errors", []),
    )


# ── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg_agentic_extraction.runner",
        description="Run the agentic KG extraction pipeline over one document cluster.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--file", type=Path, help="Text file; documents split on a ----- line.")
    source.add_argument("--cluster", type=int, help="Multi-News test-split cluster index.")
    parser.add_argument("--out", type=Path, help="Write the result JSON here (default: stdout).")
    parser.add_argument("--report", type=Path, help="Write the grader Markdown report here.")
    parser.add_argument(
        "--h5-dir",
        type=Path,
        help="Directory for the HDF5 graph (default: KG_GRAPH_OUTPUT_DIR, data/graphs).",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip writing the HDF5 graph.")
    parser.add_argument("--max-iterations", type=int, help="Override KG_MAX_ITERATIONS.")
    parser.add_argument("--no-grounding", action="store_true", help="Skip the grounder.")
    parser.add_argument("--draw", action="store_true", help="Print the graph topology and exit.")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _load_documents(args: argparse.Namespace) -> list[str]:
    if args.file:
        return [d.strip() for d in args.file.read_text().split("\n-----\n") if d.strip()]

    # Imported here so the CLI's other modes do not pull in `datasets`.
    from utils.dataset_utils import load_single_cluster

    documents, _ = load_single_cluster(cluster_idx=args.cluster)
    return documents


def _save_graph(
    result: PipelineResult,
    settings: PipelineSettings,
    *,
    cluster_index: int | None,
    out_dir: Path | None = None,
) -> Path | None:
    """
    Persist the extracted graph as HDF5 once the pipeline has finished.

    Best-effort: a write failure is logged and reported but does not change the
    exit code, since the extraction itself already succeeded and its result has
    been printed. Returns the path written, or None on failure.
    """
    assert result.knowledge_graph is not None  # guarded by the caller

    directory = out_dir or settings.graph_output_dir
    path = directory / graph_filename(cluster_index)
    try:
        return save_knowledge_graph(
            result.knowledge_graph,
            path,
            cluster_index=cluster_index,
            metadata={
                "provider": settings.llm_provider,
                "model": settings.model,
                "prompt_version": settings.prompt_version,
                "converged": result.converged,
                "iterations": result.iterations,
                "grounded": result.grounded_graph is not None,
            },
        )
    except Exception as exc:
        logger.error("failed to save graph to %s: %s", path, exc)
        return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # stdout, not basicConfig's stderr default, so progress interleaves with
    # the result JSON in the same stream. force=True because an imported
    # library may have already configured the root logger, in which case
    # basicConfig would otherwise be a no-op.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        stream=sys.stdout,
        force=True,
    )

    settings = PipelineSettings()
    if args.max_iterations:
        settings.max_iterations = args.max_iterations
    if args.no_grounding:
        settings.grounding_enabled = False

    if args.draw:
        print(build_graph(settings=settings).get_graph().draw_ascii())
        return 0

    deps = build_dependencies(settings)
    result = run_pipeline(
        _load_documents(args),
        settings=settings,
        cluster_index=args.cluster,
        graph=build_graph(deps),
    )

    if result.knowledge_graph is not None and not args.no_save:
        _save_graph(result, settings, cluster_index=args.cluster, out_dir=args.h5_dir)

    if args.report and result.grader_markdown:
        args.report.write_text(result.grader_markdown)
        logger.info("grader report → %s", args.report)

    payload = result.to_json()
    if args.out:
        args.out.write_text(payload)
        logger.info("result → %s", args.out)
    else:
        print(payload)

    if result.errors:
        for err in result.errors:
            logger.error("%s", err)
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
