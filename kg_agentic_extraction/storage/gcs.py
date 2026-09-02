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
from pathlib import Path

logger = logging.getLogger(__name__)

#: Correct for `.h5`. HDF5 has no registered IANA type, and the de-facto
#: `application/x-hdf5` makes a browser download rather than try to render it.
HDF5_CONTENT_TYPE = "application/x-hdf5"


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
