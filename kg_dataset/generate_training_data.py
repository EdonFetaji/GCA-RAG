"""
Track 2.1 — Batch clean-KG extraction.

Loops over N Multi-News clusters, runs the legacy extractor
(poc.extraction.service.full_pipeline) on each, and saves one JSON file per
cluster to data/training/clean/.

NOTE: this is deliberately still bound to the pre-LangGraph extraction path
(see poc/README.md). The clean KGs already sitting in data/training/clean/
were produced by it, so repointing this at kg_agentic_extraction/ mid-dataset
would make later clusters inconsistent with earlier ones. Rewriting Track 2
against the new pipeline is a separate task that regenerates from scratch.

Design notes
------------
Resume/checkpointing: rather than tracking "last successful index" as a
separate counter (which can go stale if a run is interrupted mid-write or
processes clusters out of order), this skips a cluster if
data/training/clean/{idx}.json already exists. Combined with the atomic
write below (write to a .tmp file, then os.replace), a file only exists
if it was fully written — so re-running this script after a crash at
cluster 150 just skips 0-149 and picks back up, without needing separate
state to go out of sync with what's actually on disk.

Failure handling: a cluster can fail for two different reasons —
(a) the Cerebras API itself (rate limits, timeouts, 5xx) — retried with
    exponential backoff, since these are usually transient;
(b) extraction/parsing failure that survives call_llm_json's own repair
    retries (bad JSON, a KG that fails Pydantic validation, etc.) — not
    worth retrying with backoff since the *content* is what's wrong.
Either way, once retries are exhausted the cluster is logged to
data/training/clean/_failures.jsonl and the batch continues — a handful
of bad clusters should never abort a 250-cluster run.

Run:
    python generate_training_data.py --num-clusters 250
    python generate_training_data.py --num-clusters 5   # smoke test first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from cerebras.cloud.sdk import CerebrasError

from poc.extraction.schemas import ExtractionRequest
from poc.extraction.service import full_pipeline
from utils.dataset_utils import load_multi_news_split, load_single_cluster

logger = logging.getLogger("generate_training_data")


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def call_with_backoff(fn, *args, max_retries: int = 4, base_delay: float = 2.0, **kwargs):
    """
    Retry an API-backed call with exponential backoff on Cerebras API errors
    (rate limits, timeouts, connection errors, 5xx). Does NOT retry on
    parsing/validation errors — those go through call_llm_json's own
    repair-retry loop already, and retrying the same bad content with a
    delay won't fix it.
    """
    delay = base_delay
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except CerebrasError as exc:
            last_exc = exc
            if attempt == max_retries:
                break
            logger.warning(
                "Cerebras API error (attempt %d/%d): %s — backing off %.1fs",
                attempt + 1, max_retries, exc, delay,
            )
            time.sleep(delay)
            delay *= 2

    raise last_exc  # type: ignore[misc]


def _log_failure(failures_path: Path, cluster_idx: int, error: Exception) -> None:
    failures_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "cluster_idx": cluster_idx,
        "error_type": type(error).__name__,
        "error": str(error),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(failures_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write via a temp file + os.replace so a file only ever exists fully-written."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)


def process_cluster(
    cluster_idx: int,
    dataset,
    output_dir: Path,
    failures_path: Path,
    req_kwargs: dict,
    max_api_retries: int,
    backoff_base: float,
) -> bool:
    """Extract one cluster and save it. Returns True on success, False on failure."""
    output_path = output_dir / f"{cluster_idx}.json"
    if output_path.exists():
        logger.info("Cluster %d — already done, skipping.", cluster_idx)
        return True

    try:
        documents, reference_summary = load_single_cluster(cluster_idx, dataset=dataset)
    except Exception as exc:
        logger.error("Cluster %d — failed to load from dataset: %s", cluster_idx, exc)
        _log_failure(failures_path, cluster_idx, exc)
        return False

    non_empty_docs = [d for d in documents if d.strip()]
    if not non_empty_docs:
        logger.warning("Cluster %d — no non-empty documents, skipping.", cluster_idx)
        _log_failure(failures_path, cluster_idx, ValueError("no non-empty documents"))
        return False

    req = ExtractionRequest(cluster_num=cluster_idx, **req_kwargs)

    try:
        response = call_with_backoff(
            full_pipeline, documents, req,
            max_retries=max_api_retries, base_delay=backoff_base,
        )
    except Exception as exc:
        logger.error("Cluster %d — extraction failed: %s: %s", cluster_idx, type(exc).__name__, exc)
        _log_failure(failures_path, cluster_idx, exc)
        return False

    output = {
        "cluster_idx": cluster_idx,
        "num_documents": len(documents),
        "reference_summary": reference_summary,
        "extraction": response.model_dump(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _write_json_atomic(output_path, output)
    except Exception as exc:
        logger.error("Cluster %d — failed to write output: %s", cluster_idx, exc)
        _log_failure(failures_path, cluster_idx, exc)
        return False

    kg = response.knowledge_graph
    logger.info(
        "Cluster %d — done. %d entities, %d relations, score=%s",
        cluster_idx, len(kg.entities), len(kg.relations),
        f"{response.final_score:.2f}" if response.final_score is not None else "n/a",
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Batch clean-KG extraction (Track 2.1)")
    parser.add_argument("--num-clusters", type=int, default=250, help="How many clusters to process (roadmap target: 200-300).")
    parser.add_argument("--start-idx", type=int, default=0, help="First cluster index to process.")
    parser.add_argument("--output-dir", type=str, default="data/training/clean", help="Where to write {cluster_idx}.json files.")
    parser.add_argument("--dataset-split", type=str, default="test", help="Multi-News split to pull clusters from.")
    parser.add_argument("--max-grader-iterations", type=int, default=2, help="Finder/grader loop cap per cluster (passed to ExtractionRequest).")
    parser.add_argument("--grader-threshold", type=float, default=0.7, help="Grader score to accept a KG (passed to ExtractionRequest).")
    parser.add_argument("--max-api-retries", type=int, default=4, help="Backoff retries on Cerebras API errors.")
    parser.add_argument("--backoff-base", type=float, default=2.0, help="Initial backoff delay in seconds (doubles each retry).")
    parser.add_argument("--sleep-between", type=float, default=1.0, help="Seconds to sleep between clusters (basic rate limiting).")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failures_path = output_dir / "_failures.jsonl"
    _configure_logging(output_dir / "generate_training_data.log")

    if not os.getenv("CEREBRAS_API_KEY"):
        logger.error("CEREBRAS_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    dataset = load_multi_news_split(args.dataset_split)
    total_available = len(dataset)
    end_idx = min(args.start_idx + args.num_clusters, total_available)
    cluster_indices = list(range(args.start_idx, end_idx))

    if end_idx < args.start_idx + args.num_clusters:
        logger.warning(
            "Requested %d clusters starting at %d, but the %s split only has %d — "
            "processing %d clusters instead.",
            args.num_clusters, args.start_idx, args.dataset_split, total_available, len(cluster_indices),
        )

    req_kwargs = {
        "dataset": "multi_news",
        "max_grader_iterations": args.max_grader_iterations,
        "grader_threshold": args.grader_threshold,
    }

    logger.info("Processing %d clusters (%d-%d) into %s", len(cluster_indices), args.start_idx, end_idx - 1, output_dir)

    succeeded, failed, skipped = 0, 0, 0
    start_time = time.time()

    for i, cluster_idx in enumerate(cluster_indices):
        already_done = (output_dir / f"{cluster_idx}.json").exists()
        ok = process_cluster(
            cluster_idx=cluster_idx,
            dataset=dataset,
            output_dir=output_dir,
            failures_path=failures_path,
            req_kwargs=req_kwargs,
            max_api_retries=args.max_api_retries,
            backoff_base=args.backoff_base,
        )
        if already_done:
            skipped += 1
        elif ok:
            succeeded += 1
            # Rate-limit only after clusters that actually hit the API.
            if i < len(cluster_indices) - 1:
                time.sleep(args.sleep_between)
        else:
            failed += 1

    elapsed = time.time() - start_time
    logger.info(
        "\n%s\nDone in %.1fs — %d newly extracted, %d already done (skipped), %d failed (see %s)\n%s",
        "=" * 80, elapsed, succeeded, skipped, failed, failures_path, "=" * 80,
    )


if __name__ == "__main__":
    main()
