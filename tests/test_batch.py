"""
Batch runner — range resolution, resume, sharding, and bundle validation.

Nothing here starts a process or touches the dataset: `select_targets` and
`shard` are pure, and `resolve_range` / `validate_bundles` only read settings.
The parallel machinery itself is exercised end-to-end by running a small real
batch, not from here.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.batch import (
    _local_clusters,
    completed_clusters,
    resolve_range,
    select_targets,
    shard,
    validate_bundles,
)
from kg_agentic_extraction.config import PipelineSettings, WorkerKeyBundle
from kg_agentic_extraction.storage import graph_filename


def _settings(**kw) -> PipelineSettings:
    return PipelineSettings.model_construct(**kw)


def _bundles(count: int, **overrides) -> list[WorkerKeyBundle]:
    """`count` complete bundles, worker 0..count-1, with distinct keys."""
    made = [
        WorkerKeyBundle(
            worker_id=i,
            gemini_keys=[f"gem-{i}a", f"gem-{i}b"],
            grader_key=f"grd-{i}",
        )
        for i in range(count)
    ]
    for worker_id, changes in overrides.items():
        made[int(worker_id.removeprefix("w"))] = made[int(worker_id.removeprefix("w"))].model_copy(
            update=changes
        )
    return made


# ── Range ─────────────────────────────────────────────────────────────


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


# ── Resume ────────────────────────────────────────────────────────────


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


# ── Sharding ──────────────────────────────────────────────────────────


def test_shard_deals_round_robin_and_stays_balanced():
    shards = shard(list(range(10)), 4)

    assert [len(s) for s in shards] == [3, 3, 2, 2]
    assert shards[0] == [0, 4, 8]


def test_shards_partition_the_work_exactly():
    todo = [3, 7, 11, 12, 40, 41, 42]
    shards = shard(todo, 4)

    flattened = [cluster for s in shards for cluster in s]
    assert sorted(flattened) == sorted(todo)
    assert len(flattened) == len(set(flattened)), "a cluster was dealt to two workers"


def test_sharding_survives_a_ragged_resume():
    """Dealing by position, not by cluster index — otherwise gaps skew the shards."""
    # Every surviving cluster is ≡ 0 mod 4, which would land entirely on worker 0
    # if the deal used the index rather than the position in the todo list.
    shards = shard([0, 4, 8, 12], 4)

    assert [len(s) for s in shards] == [1, 1, 1, 1]


def test_more_workers_than_clusters_leaves_empty_shards():
    shards = shard([5, 6], 4)

    assert [len(s) for s in shards] == [1, 1, 0, 0]


# ── Key bundles ───────────────────────────────────────────────────────


def test_validate_bundles_accepts_a_full_set():
    validate_bundles(_settings(worker_keys=_bundles(4)), 4)


def test_validate_bundles_rejects_a_missing_worker():
    with pytest.raises(SystemExit, match="worker"):
        validate_bundles(_settings(worker_keys=_bundles(2)), 4)


def test_validate_bundles_names_the_missing_variable():
    settings = _settings(worker_keys=_bundles(3, w2={"grader_key": ""}))

    with pytest.raises(SystemExit, match="KG_WORKER_2_GRADER_KEY"):
        validate_bundles(settings, 3)


def test_validate_bundles_names_a_missing_gemini_list():
    settings = _settings(worker_keys=_bundles(2, w1={"gemini_keys": []}))

    with pytest.raises(SystemExit, match="KG_WORKER_1_GEMINI_KEYS"):
        validate_bundles(settings, 2)


def test_fewer_workers_than_bundles_is_fine():
    """Running 2 workers off a 4-bundle .env must not complain about the unused two."""
    validate_bundles(_settings(worker_keys=_bundles(4, w3={"grader_key": ""})), 2)
