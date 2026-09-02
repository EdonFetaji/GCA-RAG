"""
Batch runner — range resolution and resume.

The dataset and the pipeline are never touched here: `select_targets` is pure,
and `resolve_range` only reads settings and args.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.batch import (
    _local_clusters,
    completed_clusters,
    resolve_range,
    select_targets,
)
from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.storage import graph_filename


def _settings(**kw) -> PipelineSettings:
    return PipelineSettings.model_construct(**kw)


def test_range_comes_from_the_environment_by_default():
    assert resolve_range(_settings(cluster_start=0, cluster_end=9), None) == (0, 9)


def test_explicit_range_overrides_the_environment():
    assert resolve_range(_settings(cluster_start=0, cluster_end=9), (20, 25)) == (20, 25)


def test_a_missing_range_is_a_clean_error():
    with pytest.raises(SystemExit):
        resolve_range(_settings(cluster_start=None, cluster_end=None), None)


def test_reversed_range_is_rejected():
    with pytest.raises(SystemExit):
        resolve_range(_settings(), (9, 0))


def test_select_targets_splits_on_the_done_set():
    todo, present = select_targets(0, 5, done={0, 1, 3}, force=False)

    assert todo == [2, 4, 5]
    assert present == [0, 1, 3]


def test_force_reruns_everything():
    todo, present = select_targets(0, 2, done={0, 1, 2}, force=True)

    assert todo == [0, 1, 2]
    assert present == []


def test_local_clusters_reads_the_output_dir(tmp_path):
    for done in (0, 2, 17):
        (tmp_path / graph_filename(done)).write_bytes(b"stub")
    (tmp_path / "notes.txt").write_bytes(b"ignore me")

    assert _local_clusters(tmp_path) == {0, 2, 17}


def test_completed_clusters_uses_gcs_when_a_bucket_is_set(monkeypatch, tmp_path):
    seen = {}

    def fake_list(bucket, *, prefix=""):
        seen["bucket"], seen["prefix"] = bucket, prefix
        return {4, 5, 6}

    monkeypatch.setattr("kg_agentic_extraction.batch.list_uploaded_clusters", fake_list)
    settings = _settings(gcs_bucket="multi_news_kg", gcs_prefix="graphs")

    done, label = completed_clusters(settings, tmp_path, resume="auto")

    assert done == {4, 5, 6}
    assert seen == {"bucket": "multi_news_kg", "prefix": "graphs"}
    assert label == "gs://multi_news_kg/graphs"


def test_completed_clusters_falls_back_to_local_without_a_bucket(tmp_path):
    (tmp_path / graph_filename(9)).write_bytes(b"stub")

    done, label = completed_clusters(_settings(gcs_bucket=""), tmp_path, resume="auto")

    assert done == {9}
    assert label == str(tmp_path)


def test_resume_gcs_without_a_bucket_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        completed_clusters(_settings(gcs_bucket=""), tmp_path, resume="gcs")
