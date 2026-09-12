"""
Google Cloud Storage upload for saved graphs.

The local `.h5` stays the source of truth: the file is written to disk first and
uploaded afterwards, so a bucket that is misconfigured, unreachable, or simply
not set up costs the run nothing. Nothing here is on the pipeline's critical
path.

Credentials come from Application Default Credentials — `gcloud auth
application-default login` locally, `GOOGLE_APPLICATION_CREDENTIALS` pointing at
a service-account key file, or the attached service account on GCE/Cloud Run.
The library resolves them itself; this module never reads them.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: `cluster_<i>.h5`, as written by `graph_filename`.
_GRAPH_OBJECT_RE = re.compile(r"(?:^|/)cluster_(\d+)\.h5$")

#: Correct for `.h5`. HDF5 has no registered IANA type, and the de-facto
#: `application/x-hdf5` makes a browser download rather than try to render it.
HDF5_CONTENT_TYPE = "application/x-hdf5"


def download_graph(
    bucket: str,
    cluster_index: int,
    dest_dir: str | Path,
    *,
    prefix: str = "",
    stem: str = "cluster",
) -> Path:
    """
    Download `cluster_<cluster_index>.h5` from `gs://<bucket>/<prefix>/` to `dest_dir`.

    Mirrors upload_graph()'s error handling (GCSUploadError on missing bucket,
    missing dependency, or a failed call) and its "write to a tmp path then
    replace" atomicity — a reader (Track 3's dataset sync) should never see a
    partially-downloaded .h5 as if it were complete.

    Returns
    -------
    Path
        The local path the file was written to.
    """
    if not bucket:
        raise GCSUploadError("no bucket configured (set KG_GCS_BUCKET)")

    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GCSUploadError(
            "google-cloud-storage is not installed; `uv sync` or `uv add google-cloud-storage`"
        ) from exc

    from kg_agentic_extraction.storage.hdf5 import graph_filename

    filename = graph_filename(cluster_index, stem=stem)
    name = object_name(filename, prefix)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / filename
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    try:
        blob = storage.Client().bucket(bucket).blob(name)
        blob.download_to_filename(str(tmp_path))
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise GCSUploadError(f"download of gs://{bucket}/{name} failed: {exc}") from exc

    tmp_path.replace(dest_path)
    return dest_path


class GCSUploadError(RuntimeError):
    """An upload did not complete. Raised in place of the vendor exception."""


def object_name(filename: str, prefix: str = "") -> str:
    """
    Full object name for `filename` under `prefix`.

    GCS has no directories — a prefix is just a literal part of the name — so
    the separator is normalised here rather than trusting the caller to have
    written `graphs/` and not `/graphs` or `graphs`.
    """
    prefix = prefix.strip("/")
    return f"{prefix}/{filename}" if prefix else filename


def list_uploaded_clusters(bucket: str, *, prefix: str = "") -> set[int]:
    """
    The cluster indices already present as `cluster_<i>.h5` in `gs://<bucket>/<prefix>/`.

    This is the batch runner's resume signal when a bucket is configured: the
    bucket is the source of truth, not whatever happens to be on the local disk
    of an ephemeral VM.

    Raises
    ------
    GCSUploadError
        The bucket name is empty, google-cloud-storage is missing, or the list
        call failed (auth, network, no such bucket). The caller must not silently
        treat that as "nothing done" — that would re-run everything.
    """
    if not bucket:
        raise GCSUploadError("no bucket configured (set KG_GCS_BUCKET)")

    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GCSUploadError(
            "google-cloud-storage is not installed; `uv sync` or `uv add google-cloud-storage`"
        ) from exc

    list_prefix = prefix.strip("/")
    list_prefix = f"{list_prefix}/" if list_prefix else ""
    try:
        blobs = storage.Client().list_blobs(bucket, prefix=list_prefix)
        found = {
            int(match.group(1)) for blob in blobs if (match := _GRAPH_OBJECT_RE.search(blob.name))
        }
    except Exception as exc:
        raise GCSUploadError(f"could not list gs://{bucket}/{list_prefix}: {exc}") from exc

    logger.info(
        "gs://%s/%s — %d cluster graph(s) already uploaded", bucket, list_prefix, len(found)
    )
    return found


def upload_graph(
    path: str | Path,
    bucket: str,
    *,
    prefix: str = "",
    content_type: str = HDF5_CONTENT_TYPE,
) -> str:
    """
    Upload one file to `gs://<bucket>/<prefix>/<filename>`.

    Overwrites an object of the same name, matching the local behaviour of
    `save_knowledge_graph` — re-running a cluster replaces its graph in both
    places, so the bucket cannot drift into holding a stale copy of a file the
    disk has already updated.

    Returns
    -------
    str
        The `gs://` URI of the uploaded object.

    Raises
    ------
    GCSUploadError
        The bucket name is empty, the file is missing, google-cloud-storage is
        not installed, or the upload failed.
    """
    if not bucket:
        raise GCSUploadError("no bucket configured (set KG_GCS_BUCKET)")

    path = Path(path)
    if not path.is_file():
        raise GCSUploadError(f"nothing to upload at {path}")

    # Imported lazily so a run without KG_GCS_BUCKET never pays the import, and
    # so an environment that does not need the upload need not have the package.
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise GCSUploadError(
            "google-cloud-storage is not installed; `uv sync` or `uv add google-cloud-storage`"
        ) from exc

    name = object_name(path.name, prefix)
    try:
        blob = storage.Client().bucket(bucket).blob(name)
        blob.upload_from_filename(str(path), content_type=content_type)
    except Exception as exc:
        # Wrapped rather than propagated: the caller decides an upload is
        # non-fatal, and it should not have to catch google.api_core and
        # google.auth exception trees to do so.
        raise GCSUploadError(f"upload to gs://{bucket}/{name} failed: {exc}") from exc

    uri = f"gs://{bucket}/{name}"
    logger.info("uploaded %s → %s (%d bytes)", path.name, uri, path.stat().st_size)
    return uri
