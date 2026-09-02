"""
Batch extraction over a range of Multi-News clusters.

    # range + keys + workers all from .env
    uv run python -m kg_agentic_extraction.batch

    # or spell the range out (overrides KG_CLUSTER_START / KG_CLUSTER_END)
    uv run python -m kg_agentic_extraction.batch --range 0 199
    uv run python -m kg_agentic_extraction.batch --range 0 199 --workers 4 --no-grounding

Two properties make this safe to stop and restart:

- **Resumable.** A cluster whose ``cluster_<i>.h5`` is already done is skipped, so
  re-running after a crash, a `Ctrl-C`, or every Gemini key hitting its daily
  quota picks up where it stopped. "Done" is read from the **GCS bucket** when
  one is configured (`KG_GCS_BUCKET`) — the bucket is the source of truth, not
  the local disk of an ephemeral VM — and from the local output directory
  otherwise. Force either with ``--resume {gcs,local}``.
- **Quota-aware.** The gemini adapter rotates through `GEMINI_API_KEYS` as each
  key hits its per-day free-tier limit. When the *last* key is spent it raises
  `AllGeminiKeysExhausted`; this runner catches it, reports how far it got, and
  exits with status 2 so a wrapper can retry tomorrow.

Concurrency is thread-based and sized for the target box (c4-highcpu-8, 8 vCPUs
/ 16 GB) via `KG_MAX_WORKERS` — see `PipelineSettings.max_workers`. Each worker
holds its own compiled graph and its own DBpedia MCP session; the rotating
Gemini client is shared, so a key exhausted by one worker is skipped by all.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.graph import build_dependencies, build_graph
from kg_agentic_extraction.llm.factory import build_llm
from kg_agentic_extraction.llm.gemini_client import AllGeminiKeysExhausted
from kg_agentic_extraction.runner import (
    _save_graph,
    _upload_graph,
    format_duration,
    run_pipeline,
)
from kg_agentic_extraction.storage import list_uploaded_clusters

_LOCAL_GRAPH_RE = re.compile(r"^cluster_(\d+)\.h5$")

logger = logging.getLogger(__name__)


@dataclass
class BatchSummary:
    """Outcome of one batch invocation."""

    requested: list[int]
    already_present: list[int]
    completed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    #: Set when every Gemini key hit its daily quota mid-run.
    quota_exhausted: bool = False
    elapsed_seconds: float = 0.0

    @property
    def exit_code(self) -> int:
        if self.quota_exhausted:
            return 2
        return 1 if self.failed else 0


def resolve_range(settings: PipelineSettings, override: tuple[int, int] | None) -> tuple[int, int]:
    """The inclusive [start, end] cluster range, from `--range` or the environment."""
    if override is not None:
        start, end = override
    else:
        start, end = settings.cluster_start, settings.cluster_end
    if start is None or end is None:
        raise SystemExit(
            "no cluster range: pass --range START END, or set KG_CLUSTER_START and "
            "KG_CLUSTER_END in .env"
        )
    if end < start:
        raise SystemExit(f"empty range: end ({end}) is before start ({start})")
    return start, end


def _local_clusters(out_dir: Path) -> set[int]:
    """Cluster indices already written as `cluster_<i>.h5` under `out_dir`."""
    if not out_dir.is_dir():
        return set()
    return {
        int(match.group(1))
        for path in out_dir.iterdir()
        if (match := _LOCAL_GRAPH_RE.match(path.name))
    }


def completed_clusters(
    settings: PipelineSettings, out_dir: Path, *, resume: str
) -> tuple[set[int], str]:
    """
    The set of clusters already done, and a human label for where that was read.

    `resume` is "gcs", "local", or "auto" (gcs when a bucket is configured,
    else local). A GCS read that fails is fatal — silently falling back to an
    empty set would re-extract everything.
    """
    use_gcs = resume == "gcs" or (resume == "auto" and bool(settings.gcs_bucket))
    if use_gcs:
        if not settings.gcs_bucket:
            raise SystemExit("--resume gcs needs KG_GCS_BUCKET set")
        done = list_uploaded_clusters(settings.gcs_bucket, prefix=settings.gcs_prefix)
        label = f"gs://{settings.gcs_bucket}/{settings.gcs_prefix}".rstrip("/")
        return done, label
    return _local_clusters(out_dir), str(out_dir)


def select_targets(
    start: int, end: int, *, done: set[int], force: bool
) -> tuple[list[int], list[int]]:
    """Split the inclusive range into (to-run, already-present)."""
    requested = list(range(start, end + 1))
    if force:
        return requested, []
    present = [i for i in requested if i in done]
    todo = [i for i in requested if i not in done]
    return todo, present


def run_batch(
    settings: PipelineSettings,
    *,
    start: int,
    end: int,
    workers: int,
    out_dir: Path,
    resume: str = "auto",
    force: bool = False,
    no_upload: bool = False,
) -> BatchSummary:
    """Extract every cluster in [start, end] that is not already done (see `completed_clusters`)."""
    # Imported here so the module's --help does not pull in `datasets`.
    from utils.dataset_utils import load_multi_news_split, load_single_cluster

    done, done_label = completed_clusters(settings, out_dir, resume=resume)
    todo, present = select_targets(start, end, done=done, force=force)
    summary = BatchSummary(requested=list(range(start, end + 1)), already_present=present)

    if present:
        logger.info(
            "resuming — %d/%d cluster(s) already done (%s)",
            len(present),
            len(summary.requested),
            done_label,
        )
    if not todo:
        logger.info("nothing to do: clusters %d–%d are all present", start, end)
        return summary

    logger.info(
        "batch — %d cluster(s) to extract, %d worker(s), grounding %s",
        len(todo),
        workers,
        "on" if settings.grounding_enabled else "off",
    )

    dataset = load_multi_news_split("test")
    shared_llm = build_llm(settings)  # one rotating client, shared across workers

    # Each worker gets its own compiled graph: the DBpedia MCP backend holds a
    # single session and is not safe to enter from two threads at once.
    local = threading.local()

    def app_for_worker():
        app = getattr(local, "app", None)
        if app is None:
            app = build_graph(build_dependencies(settings, llm=shared_llm))
            local.app = app
        return app

    stop = threading.Event()

    def work(cluster_index: int) -> tuple[str, int]:
        if stop.is_set():
            return "skipped", cluster_index
        documents, _ = load_single_cluster(cluster_index, dataset=dataset)
        try:
            result = run_pipeline(
                documents,
                settings=settings,
                cluster_index=cluster_index,
                graph=app_for_worker(),
            )
        except AllGeminiKeysExhausted:
            stop.set()
            raise

        if result.knowledge_graph is None:
            logger.error(
                "cluster %d — no graph produced: %s", cluster_index, "; ".join(result.errors)
            )
            return "failed", cluster_index

        saved = _save_graph(result, settings, cluster_index=cluster_index, out_dir=out_dir)
        if saved is None:
            return "failed", cluster_index
        if not no_upload:
            _upload_graph(saved, settings)
        logger.info(
            "cluster %d — %s in %s (%d entities, %d relations)",
            cluster_index,
            "converged" if result.converged else "unconverged",
            format_duration(result.elapsed_seconds),
            len(result.knowledge_graph.entities),
            len(result.knowledge_graph.relations),
        )
        return "done", cluster_index

    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="cluster") as pool:
            for outcome, index in pool.map(work, todo):
                {"done": summary.completed, "failed": summary.failed, "skipped": summary.skipped}[
                    outcome
                ].append(index)
    except AllGeminiKeysExhausted as exc:
        summary.quota_exhausted = True
        logger.error("%s — stopping. Re-run to resume once the quotas reset.", exc)
    finally:
        summary.elapsed_seconds = time.perf_counter() - started

    return summary


# ── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg_agentic_extraction.batch",
        description="Extract a range of Multi-News clusters, resumably.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive cluster range. Overrides KG_CLUSTER_START / KG_CLUSTER_END.",
    )
    parser.add_argument("--workers", type=int, help="Override KG_MAX_WORKERS.")
    parser.add_argument(
        "--h5-dir",
        type=Path,
        help="Directory for the HDF5 graphs (default: KG_GRAPH_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--resume",
        choices=("auto", "gcs", "local"),
        default="auto",
        help=(
            "Where to read which clusters are already done: gcs (the bucket), local (the "
            "output dir), or auto — gcs when KG_GCS_BUCKET is set, else local (default)."
        ),
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-run every cluster in range, done or not."
    )
    parser.add_argument(
        "--no-upload", action="store_true", help="Skip the GCS upload even if a bucket is set."
    )
    parser.add_argument("--no-grounding", action="store_true", help="Skip the grounder.")
    parser.add_argument("--max-iterations", type=int, help="Override KG_MAX_ITERATIONS.")
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Skip scripts/cleanup.sh (stale .h5.tmp, caches, HF locks) at the end.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _run_cleanup() -> None:
    """Best-effort: run scripts/cleanup.sh once the batch is done. Never fatal."""
    import subprocess

    script = Path(__file__).resolve().parent.parent / "scripts" / "cleanup.sh"
    if not script.is_file():
        return
    try:
        subprocess.run(["bash", str(script)], check=False, timeout=120)
    except Exception as exc:  # noqa: BLE001 — cleanup failing must not fail the run
        logger.warning("cleanup script did not complete: %s", exc)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
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

    start, end = resolve_range(settings, tuple(args.range) if args.range else None)
    workers = args.workers or settings.max_workers
    out_dir = args.h5_dir or settings.graph_output_dir

    summary = run_batch(
        settings,
        start=start,
        end=end,
        workers=workers,
        out_dir=out_dir,
        resume=args.resume,
        force=args.force,
        no_upload=args.no_upload,
    )

    logger.info(
        "batch done in %s — %d extracted, %d already present, %d failed%s%s",
        format_duration(summary.elapsed_seconds),
        len(summary.completed),
        len(summary.already_present),
        len(summary.failed),
        f", {len(summary.skipped)} skipped" if summary.skipped else "",
        " — DAILY QUOTA EXHAUSTED, resume later" if summary.quota_exhausted else "",
    )
    if summary.failed:
        logger.warning("failed clusters: %s", ", ".join(map(str, summary.failed)))

    if not args.no_cleanup:
        _run_cleanup()
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
