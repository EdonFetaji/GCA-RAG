"""
sync_from_gcs.py — pull extracted clusters out of gs://<bucket>/<prefix>/cluster_<i>.h5
(the format kg_agentic_extraction.storage.hdf5 writes and the batch runner
uploads via kg_agentic_extraction.storage.gcs.upload_graph) and cache them
locally as {cluster_idx}.json in the exact shape
kg_dataset/generate_corruptions.py and gnn_validator/data.py already expect:

    {"cluster_idx": <int>, "extraction": {"knowledge_graph": {"entities": [...], "relations": [...]}}}

This is a translation/caching step, not a new data path through the rest of
the pipeline — once this has run, `generate_corruptions.py --scheme extended
--clean-dir <output>` and `gnn_validator.train --clean-dir <output>
--corrupted-dir <corrupted output>` work completely unchanged.

Resumable like generate_training_data.py: a cluster already present locally
(by cluster_idx) is skipped unless --overwrite is given, so a run interrupted
partway through 700+ downloads can just be re-run.

Run:
    python -m gnn_validator.sync_from_gcs --bucket my-bucket --prefix graphs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

from kg_agentic_extraction.storage.gcs import GCSUploadError, download_graph, list_uploaded_clusters
from kg_agentic_extraction.storage.hdf5 import load_knowledge_graph

logger = logging.getLogger("sync_from_gcs")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def _json_safe(value):
    """HDF5 attrs come back as numpy scalar types (int64, float64, bool_),
    which json.dumps doesn't accept — coerce to native Python types."""
    if hasattr(value, "item"):
        return value.item()
    return value


def sync_cluster(bucket: str, cluster_idx: int, prefix: str, output_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        local_h5 = download_graph(bucket, cluster_idx, tmp, prefix=prefix)
        graph, attrs = load_knowledge_graph(local_h5)

    record = {
        "cluster_idx": cluster_idx,
        "source": "gcs",
        "hdf5_attrs": {k: _json_safe(v) for k, v in attrs.items()},
        "extraction": {
            "knowledge_graph": {
                "entities": [e.model_dump(mode="json") for e in graph.entities],
                "relations": [r.model_dump(mode="json") for r in graph.relations],
            }
        },
    }

    dest = output_dir / f"{cluster_idx}.json"
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    tmp_path.write_text(json.dumps(record, indent=2))
    tmp_path.replace(dest)


def main():
    parser = argparse.ArgumentParser(description="Sync extracted clusters from GCS (HDF5) into the local clean-KG JSON cache.")
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="", help="Object prefix under the bucket, e.g. 'graphs'.")
    parser.add_argument("--output-dir", type=str, default="data/training/clean_gcs")
    parser.add_argument("--limit", type=int, default=None, help="Only sync the first N cluster indices found (sorted), for a quick test run.")
    parser.add_argument("--overwrite", action="store_true", help="Re-download and overwrite clusters that already exist locally.")
    args = parser.parse_args()

    _configure_logging()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        cluster_ids = sorted(list_uploaded_clusters(args.bucket, prefix=args.prefix))
    except GCSUploadError as exc:
        logger.error("Could not list gs://%s/%s: %s", args.bucket, args.prefix, exc)
        sys.exit(1)

    if args.limit is not None:
        cluster_ids = cluster_ids[: args.limit]

    logger.info("Found %d cluster(s) in gs://%s/%s.", len(cluster_ids), args.bucket, args.prefix)

    synced, skipped, failed = 0, 0, 0
    for cluster_idx in cluster_ids:
        dest = output_dir / f"{cluster_idx}.json"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            sync_cluster(args.bucket, cluster_idx, args.prefix, output_dir)
            synced += 1
        except GCSUploadError as exc:
            logger.warning("Cluster %d failed to sync: %s", cluster_idx, exc)
            failed += 1
        except Exception:
            logger.exception("Cluster %d failed to sync (unexpected error)", cluster_idx)
            failed += 1

    logger.info("Done. %d synced, %d skipped (already local), %d failed.", synced, skipped, failed)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
