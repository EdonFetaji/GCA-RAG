"""
GCS upload — object naming, and that a failure stays non-fatal.

The client is stubbed by injecting a fake `google.cloud.storage` into
`sys.modules`, since `upload_graph` imports it lazily inside the call. That
keeps the test offline while still exercising the real bucket/blob call chain,
which is where a typo would otherwise hide until a run against a live bucket.
"""

from __future__ import annotations

import sys
import types

import pytest

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.runner import _upload_graph
from kg_agentic_extraction.storage import GCSUploadError, object_name, upload_graph


class _FakeBlob:
    def __init__(self, name: str, uploaded: list, bucket: str) -> None:
        self.name = name
        self._uploaded = uploaded
        self._bucket = bucket

    def upload_from_filename(self, filename: str, content_type: str | None = None) -> None:
        self._uploaded.append((self._bucket, self.name, filename, content_type))


class _FakeBucket:
    def __init__(self, name: str, uploaded: list) -> None:
        self.name = name
        self._uploaded = uploaded

    def blob(self, name: str) -> _FakeBlob:
        return _FakeBlob(name, self._uploaded, self.name)


class _FakeClient:
    def __init__(self, uploaded: list, boom: Exception | None = None) -> None:
        self._uploaded = uploaded
        self._boom = boom

    def bucket(self, name: str) -> _FakeBucket:
        if self._boom:
            raise self._boom
        return _FakeBucket(name, self._uploaded)


@pytest.fixture
def uploaded(monkeypatch):
    """Install a fake `google.cloud.storage`; yields the list of upload calls."""
    calls: list = []
    module = types.ModuleType("google.cloud.storage")
    module.Client = lambda: _FakeClient(calls)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.cloud.storage", module)
    monkeypatch.setattr("google.cloud.storage", module, raising=False)
    return calls


@pytest.fixture
def h5(tmp_path):
    path = tmp_path / "cluster_0.h5"
    path.write_bytes(b"\x89HDF\r\n\x1a\n")
    return path


@pytest.mark.parametrize(
    ("prefix", "expected"),
    [
        ("", "cluster_0.h5"),
        ("graphs", "graphs/cluster_0.h5"),
        ("/graphs/", "graphs/cluster_0.h5"),
        ("runs/v3", "runs/v3/cluster_0.h5"),
    ],
)
def test_object_name_normalises_the_prefix(prefix, expected):
    assert object_name("cluster_0.h5", prefix) == expected


def test_upload_returns_the_gs_uri(uploaded, h5):
    uri = upload_graph(h5, "kg-graphs", prefix="graphs")
    assert uri == "gs://kg-graphs/graphs/cluster_0.h5"
    assert uploaded == [("kg-graphs", "graphs/cluster_0.h5", str(h5), "application/x-hdf5")]


def test_upload_without_a_bucket_is_an_error(h5):
    with pytest.raises(GCSUploadError, match="no bucket"):
        upload_graph(h5, "")


def test_upload_of_a_missing_file_is_an_error(tmp_path):
    with pytest.raises(GCSUploadError, match="nothing to upload"):
        upload_graph(tmp_path / "absent.h5", "kg-graphs")


def test_client_failure_is_wrapped(monkeypatch, h5):
    module = types.ModuleType("google.cloud.storage")
    module.Client = lambda: _FakeClient([], boom=RuntimeError("403 forbidden"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.cloud.storage", module)
    with pytest.raises(GCSUploadError, match="403 forbidden"):
        upload_graph(h5, "kg-graphs")


def test_runner_skips_the_upload_when_no_bucket_is_configured(uploaded, h5):
    settings = PipelineSettings(gcs_bucket="", _env_file=None)
    assert _upload_graph(h5, settings) is None
    assert uploaded == []


def test_runner_uploads_when_a_bucket_is_configured(uploaded, h5):
    settings = PipelineSettings(gcs_bucket="kg-graphs", gcs_prefix="graphs", _env_file=None)
    assert _upload_graph(h5, settings) == "gs://kg-graphs/graphs/cluster_0.h5"
    assert len(uploaded) == 1


def test_a_failed_upload_does_not_raise(monkeypatch, h5, caplog):
    """The .h5 is already on disk and the run already succeeded — log and move on."""
    module = types.ModuleType("google.cloud.storage")
    module.Client = lambda: _FakeClient([], boom=RuntimeError("network unreachable"))  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "google.cloud.storage", module)
    settings = PipelineSettings(gcs_bucket="kg-graphs", _env_file=None)
    with caplog.at_level("ERROR"):
        assert _upload_graph(h5, settings) is None
    assert any("network unreachable" in r.getMessage() for r in caplog.records)
