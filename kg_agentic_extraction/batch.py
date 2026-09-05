"""
Batch extraction over a range of Multi-News clusters.

    # range + workers from .env
    uv run python -m kg_agentic_extraction.batch

    # or spell the range out (overrides KG_CLUSTER_START / KG_CLUSTER_END)
    uv run python -m kg_agentic_extraction.batch --range 0 199
    uv run python -m kg_agentic_extraction.batch --range 0 199 --workers 4

Parallelism
-----------
`KG_MAX_WORKERS` **processes**, one per core, each running a single-threaded
LangGraph pipeline over its own slice of the range. Not threads — and the reason
is API keys, not CPU.

Every worker owns a `WorkerKeyBundle`: two Gemini keys for the extractor, one
Mistral key for the grader, shared with nobody. A rotating `GeminiClient` tracks
which of its keys are spent in an instance attribute, so putting several workers
in one process would make them share that state: the first worker to exhaust a
key would retire it for all of them, and one worker running out of quota would
end the whole batch. In separate processes each worker's rotation is its own, and
a worker that exhausts its bundle stops alone while the other three keep going.

`spawn`, not `fork`: the Gemini SDK sits on gRPC and gRPC channels do not survive
a fork, and torch and h5py are no better. Spawn also means each child re-reads
`.env` from scratch, which is the seam the per-worker key injection uses.

The parent process builds no LLM client and runs no pipeline. It resolves the
range, reads which clusters are already done, shards the rest, and waits.

Resume
------
A cluster whose ``cluster_<i>.h5`` already exists is skipped, so re-running after
a crash, a `Ctrl-C`, or a quota wall picks up where it stopped. "Done" is read
from the **GCS bucket** when one is configured (`KG_GCS_BUCKET`) — the bucket is
the source of truth, not the local disk of an ephemeral VM — and from the local
output directory otherwise. Force either with ``--resume {gcs,local}``.

Exit status is `2` when any worker hit its daily Gemini quota (resume tomorrow),
`1` when a cluster failed for any other reason, `0` otherwise.

Grounding is off here unconditionally. It is a per-process MCP session against a
single DBpedia server and it is not part of this extraction pass; use
`runner.py` for a single grounded cluster.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.storage import list_uploaded_clusters

_LOCAL_GRAPH_RE = re.compile(r"^cluster_(\d+)\.h5$")

logger = logging.getLogger(__name__)


@dataclass
class WorkerSummary:
    """What one worker process accomplished, sent back over the result queue."""

    worker_id: int
    completed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    #: This worker's Gemini keys are all spent for the day. Its own shard stopped.
    quota_exhausted: bool = False
    #: Set when the worker died before it could report — see `run_batch`.
    crashed: bool = False


@dataclass
class BatchSummary:
    """Outcome of one batch invocation, aggregated across workers."""

    requested: list[int]
    already_present: list[int]
    completed: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    #: True when *any* worker exhausted its bundle. The others may have finished.
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


def shard(todo: list[int], workers: int) -> list[list[int]]:
    """
    Deal `todo` round-robin into `workers` shards.

    Round-robin over the *todo list*, not over the raw cluster index: after a
    resume has removed an arbitrary set of clusters, dealing by index would leave
    one worker with most of the remainder. Dealing by position keeps the shards
    within one cluster of each other however ragged the leftovers are.

    Cluster cost varies a lot — a graph that converges in two rounds against one
    that runs to the iteration cap — so this is not perfect balance. It is chosen
    over a shared work queue because it needs no live channel between parent and
    children: each worker knows its whole list up front and reports once at the
    end, which is one failure mode instead of several.
    """
    return [todo[i::workers] for i in range(workers)]


def validate_bundles(settings: PipelineSettings, workers: int) -> None:
    """
    Fail before spawning anything if the key bundles do not cover `workers`.

    Deliberately fatal rather than degrading to fewer workers: a batch quietly
    running at half the requested parallelism because an env var was misspelled
    is the kind of thing that is only noticed hours later.
    """
    available = {bundle.worker_id for bundle in settings.worker_keys}
    missing_ids = [w for w in range(workers) if w not in available]
    if missing_ids:
        wanted = ", ".join(
            f"KG_WORKER_{w}_GEMINI_KEYS / KG_WORKER_{w}_MISTRAL_KEY" for w in missing_ids
        )
        raise SystemExit(
            f"{workers} worker(s) requested but no key bundle for worker(s) "
            f"{', '.join(map(str, missing_ids))}. Set {wanted} in .env, or lower --workers."
        )

    for worker_id in range(workers):
        gaps = settings.worker_bundle(worker_id).missing()
        if gaps:
            raise SystemExit(f"worker {worker_id} is missing: {', '.join(gaps)}")


# ── Worker process ────────────────────────────────────────────────────


def _worker_main(
    worker_id: int,
    settings: PipelineSettings,
    clusters: list[int],
    out_dir: Path,
    no_upload: bool,
    verbose: bool,
    queue: mp.Queue,
) -> None:
    """
    One worker process: build the pipeline once, then run its shard sequentially.

    Everything here is process-local — LLM clients, compiled graph, key rotation
    state. Nothing is shared with a sibling worker, which is the whole point of
    the process model.

    `settings` arrives already narrowed by `for_worker`, pickled across from the
    parent rather than re-read from `.env` here. That is deliberate: the parent
    has applied the CLI overrides (`--max-iterations`) by then, and a child that
    re-read the environment would silently ignore them. The narrowing also means
    the only keys in this process are this worker's own.

    Imports are inside the function because this runs under `spawn`: the child
    re-imports this module from scratch, and pulling `datasets` and torch in at
    module scope would cost that on `--help` too.
    """
    # One thread per process. Without this each of the four workers asks torch
    # and the BLAS underneath it for all the cores, and they spend their time
    # descheduling each other rather than waiting on the provider.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=f"%(asctime)s [%(levelname)s] [w{worker_id}] %(name)s — %(message)s",
        stream=sys.stdout,
        force=True,
    )
    log = logging.getLogger(f"{__name__}.w{worker_id}")

    from kg_agentic_extraction.graph import build_dependencies, build_graph
    from kg_agentic_extraction.llm.gemini_client import AllGeminiKeysExhausted
    from kg_agentic_extraction.runner import (
        _save_graph,
        _upload_graph,
        format_duration,
        run_pipeline,
    )
    from utils.dataset_utils import load_multi_news_split, load_single_cluster

    summary = WorkerSummary(worker_id=worker_id)
    try:
        settings.grounding_enabled = False
        bundle = settings.worker_bundle(worker_id)
        log.info(
            "starting — %d cluster(s), %d Gemini key(s) + 1 Mistral key",
            len(clusters),
            len(bundle.gemini_keys),
        )

        dataset = load_multi_news_split("test")
        app = build_graph(build_dependencies(settings))

        for cluster_index in clusters:
            documents, _ = load_single_cluster(cluster_index, dataset=dataset)
            result = run_pipeline(
                documents,
                settings=settings,
                cluster_index=cluster_index,
                graph=app,
            )

            if result.knowledge_graph is None:
                log.error(
                    "cluster %d — no graph produced: %s",
                    cluster_index,
                    "; ".join(result.errors),
                )
                summary.failed.append(cluster_index)
                continue

            saved = _save_graph(result, settings, cluster_index=cluster_index, out_dir=out_dir)
            if saved is None:
                summary.failed.append(cluster_index)
                continue
            if not no_upload:
                _upload_graph(saved, settings)

            log.info(
                "cluster %d — %s in %s (%d entities, %d relations)",
                cluster_index,
                "converged" if result.converged else "unconverged",
                format_duration(result.elapsed_seconds),
                len(result.knowledge_graph.entities),
                len(result.knowledge_graph.relations),
            )
            summary.completed.append(cluster_index)

    except AllGeminiKeysExhausted as exc:
        # This worker's bundle is spent for the day. Its siblings have their own
        # keys and are unaffected, so this ends one shard, not the batch.
        summary.quota_exhausted = True
        log.error("%s — this worker is stopping; siblings continue.", exc)
    except BaseException:
        log.exception("worker %d died", worker_id)
        summary.crashed = True
    finally:
        queue.put(summary)

    done = len(summary.completed)
    log.info("finished — %d/%d cluster(s) extracted", done, len(clusters))


# ── Parent process ────────────────────────────────────────────────────


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
    verbose: bool = False,
) -> BatchSummary:
    """Extract every cluster in [start, end] not already done, across `workers` processes."""
    # `--force` re-runs everything, so the done-set would be discarded anyway.
    # Skipping the read is not just an optimisation: with a bucket configured it
    # is a network call that needs credentials, and failing it is a confusing way
    # for a `--force` run to die.
    if force:
        done, done_label = set(), "(forced)"
    else:
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

    workers = min(workers, len(todo))
    validate_bundles(settings, workers)
    shards = shard(todo, workers)

    logger.info(
        "batch — %d cluster(s) across %d worker process(es): %s",
        len(todo),
        workers,
        ", ".join(f"w{i}={len(s)}" for i, s in enumerate(shards)),
    )

    # spawn: gRPC channels (Gemini) do not survive fork, and neither do torch's
    # thread pools. It is also what makes the child re-read .env cleanly.
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_worker_main,
            args=(
                worker_id,
                # Narrowed here, in the parent: only this worker's keys are
                # pickled into the child, and the CLI overrides already applied
                # to `settings` travel with them.
                settings.for_worker(worker_id),
                shards[worker_id],
                out_dir,
                no_upload,
                verbose,
                queue,
            ),
            name=f"kg-worker-{worker_id}",
        )
        for worker_id in range(workers)
    ]

    started = time.perf_counter()
    try:
        for process in processes:
            process.start()

        # Drain before joining. A queue big enough to fill its pipe blocks the
        # child in `put()` until someone reads, and a join-first order would then
        # deadlock — the classic multiprocessing footgun.
        reported: dict[int, WorkerSummary] = {}
        for _ in processes:
            result: WorkerSummary = queue.get()
            reported[result.worker_id] = result

        for process in processes:
            process.join()
    except KeyboardInterrupt:
        logger.warning("interrupted — terminating workers; re-run to resume")
        for process in processes:
            process.terminate()
        for process in processes:
            process.join(timeout=10)
        raise
    finally:
        summary.elapsed_seconds = time.perf_counter() - started

    for worker_id in range(workers):
        result = reported.get(worker_id)
        if result is None:
            # Killed before it could report (OOM, SIGKILL). Its whole shard is
            # unaccounted for; count it failed so the exit code is non-zero and
            # resume picks it up next run.
            logger.error("worker %d never reported — counting its shard as failed", worker_id)
            summary.failed.extend(shards[worker_id])
            continue

        summary.completed.extend(result.completed)
        summary.failed.extend(result.failed)
        summary.quota_exhausted |= result.quota_exhausted

        accounted = set(result.completed) | set(result.failed)
        unreached = [c for c in shards[worker_id] if c not in accounted]
        if unreached:
            # Stopped early — quota wall or a crash. Not failures: nothing was
            # attempted, and resume will find them missing and retry them.
            summary.skipped.extend(unreached)
        if result.crashed:
            logger.error(
                "worker %d crashed; %d cluster(s) left unattempted", worker_id, len(unreached)
            )

    summary.completed.sort()
    summary.failed.sort()
    summary.skipped.sort()
    return summary


# ── CLI ───────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg_agentic_extraction.batch",
        description="Extract a range of Multi-News clusters across worker processes, resumably.",
    )
    parser.add_argument(
        "--range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive cluster range. Overrides KG_CLUSTER_START / KG_CLUSTER_END.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override KG_MAX_WORKERS. Capped by the number of KG_WORKER_<n>_* key bundles.",
    )
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
    # Deferred for the same reason the worker's imports are: `runner` pulls in
    # the model and storage stack, and `--help` should not pay for it.
    from kg_agentic_extraction.runner import format_duration

    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] [parent] %(name)s — %(message)s",
        stream=sys.stdout,
        force=True,
    )

    settings = PipelineSettings()
    if args.max_iterations:
        settings.max_iterations = args.max_iterations

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
        verbose=args.verbose,
    )

    logger.info(
        "batch done in %s — %d extracted, %d already present, %d failed%s%s",
        format_duration(summary.elapsed_seconds),
        len(summary.completed),
        len(summary.already_present),
        len(summary.failed),
        f", {len(summary.skipped)} not attempted" if summary.skipped else "",
        " — DAILY QUOTA EXHAUSTED, resume later" if summary.quota_exhausted else "",
    )
    if summary.failed:
        logger.warning("failed clusters: %s", ", ".join(map(str, summary.failed)))

    if not args.no_cleanup:
        _run_cleanup()
    return summary.exit_code


if __name__ == "__main__":
    sys.exit(main())
