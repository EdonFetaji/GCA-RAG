"""
`PipelineSettings.for_role` / `.for_worker` — the two scoping copies.

These are what let a single `PipelineSettings` serve an extractor on Gemini and
a grader on Mistral, and what gives each batch worker keys nobody else holds,
without `llm/factory.py` knowing that either concept exists. So the properties
worth pinning are: the right fields change, and *nothing else does*.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.config import (
    PipelineSettings,
    WorkerKeyBundle,
    _collect_worker_keys,
    split_keys,
)


def _settings(**kw) -> PipelineSettings:
    return PipelineSettings.model_construct(worker_keys=[], **kw)


# ── for_role ──────────────────────────────────────────────────────────


def test_roles_resolve_to_their_own_providers():
    settings = _settings(
        llm_provider="groq",
        model="llama-3.3-70b-versatile",
        extractor_provider="gemini",
        extractor_model="gemini-2.5-flash",
        grader_provider="mistral",
        grader_model="mistral-large-latest",
    )

    extractor = settings.for_role("extractor")
    grader = settings.for_role("grader")

    assert (extractor.llm_provider, extractor.model) == ("gemini", "gemini-2.5-flash")
    assert (grader.llm_provider, grader.model) == ("mistral", "mistral-large-latest")


def test_an_unbound_role_falls_back_to_the_default_provider():
    """A .env with no role bindings must behave exactly as it did before roles existed."""
    settings = _settings(llm_provider="groq", model="llama-3.3-70b-versatile")

    for role in ("extractor", "grader"):
        scoped = settings.for_role(role)
        assert (scoped.llm_provider, scoped.model) == ("groq", "llama-3.3-70b-versatile")


def test_a_role_can_override_only_the_model():
    settings = _settings(
        llm_provider="gemini", model="gemini-2.5-flash", grader_model="gemini-2.5-pro"
    )

    grader = settings.for_role("grader")

    assert grader.llm_provider == "gemini"
    assert grader.model == "gemini-2.5-pro"


def test_for_role_does_not_mutate_the_original():
    settings = _settings(llm_provider="groq", model="m", grader_provider="mistral")

    settings.for_role("grader")

    assert settings.llm_provider == "groq"


def test_for_role_leaves_every_other_setting_alone():
    settings = _settings(
        llm_provider="groq",
        model="m",
        extractor_provider="gemini",
        max_iterations=9,
        gcs_bucket="a-bucket",
    )

    scoped = settings.for_role("extractor")

    assert scoped.max_iterations == 9
    assert scoped.gcs_bucket == "a-bucket"


# ── for_worker ────────────────────────────────────────────────────────


def _four_bundles() -> list[WorkerKeyBundle]:
    return [
        WorkerKeyBundle(worker_id=i, gemini_keys=[f"gem-{i}a", f"gem-{i}b"], mistral_key=f"mis-{i}")
        for i in range(4)
    ]


def test_a_worker_gets_exactly_its_own_bundle():
    settings = PipelineSettings.model_construct(worker_keys=_four_bundles())

    scoped = settings.for_worker(1)

    assert scoped.gemini_key_list() == ["gem-1a", "gem-1b"]
    assert scoped.mistral_api_key == "mis-1"


def test_a_worker_cannot_see_another_workers_keys():
    """
    The isolation the process model buys is worthless if a bundle leaks sideways.

    Checked against the *whole* serialized copy, not just the provider fields:
    this object is pickled into the child process, so a sibling key surviving
    anywhere on it — `worker_keys` included — has crossed the boundary.
    """
    settings = PipelineSettings.model_construct(worker_keys=_four_bundles())

    scoped = settings.for_worker(2)
    serialized = scoped.model_dump_json()

    assert scoped.gemini_key_list() == ["gem-2a", "gem-2b"]
    for other in (0, 1, 3):
        for key in (f"gem-{other}a", f"gem-{other}b", f"mis-{other}"):
            assert key not in serialized, f"worker 2 can see worker {other}'s {key}"


def test_the_narrowed_copy_still_knows_its_own_bundle():
    """The child logs its key count from this, so narrowing must not orphan it."""
    scoped = PipelineSettings.model_construct(worker_keys=_four_bundles()).for_worker(3)

    assert scoped.worker_bundle(3).gemini_keys == ["gem-3a", "gem-3b"]


def test_cli_overrides_survive_the_narrowing():
    """`--max-iterations` is applied in the parent; the child must inherit it."""
    settings = PipelineSettings.model_construct(max_iterations=2, worker_keys=_four_bundles())

    assert settings.for_worker(1).max_iterations == 2


def test_the_bundle_replaces_the_process_wide_keys_rather_than_adding_to_them():
    """A worker must not be able to fall back onto the shared GEMINI_API_KEY."""
    settings = PipelineSettings.model_construct(
        gemini_api_key="shared-key",
        gemini_api_keys="shared-2,shared-3",
        mistral_api_key="shared-mistral",
        worker_keys=_four_bundles(),
    )

    scoped = settings.for_worker(0)

    assert scoped.gemini_key_list() == ["gem-0a", "gem-0b"]
    assert scoped.mistral_api_key == "mis-0"


def test_an_unconfigured_worker_is_an_error():
    settings = PipelineSettings.model_construct(worker_keys=_four_bundles())

    with pytest.raises(KeyError, match="KG_WORKER_9"):
        settings.for_worker(9)


def test_bundles_report_what_they_are_missing():
    assert WorkerKeyBundle(worker_id=2, gemini_keys=["g"]).missing() == ["KG_WORKER_2_MISTRAL_KEY"]
    assert WorkerKeyBundle(worker_id=0, mistral_key="m").missing() == ["KG_WORKER_0_GEMINI_KEYS"]
    assert WorkerKeyBundle(worker_id=1, gemini_keys=["g"], mistral_key="m").is_complete


# ── Environment scanning ──────────────────────────────────────────────


def test_worker_bundles_are_scanned_from_the_environment(monkeypatch):
    monkeypatch.setenv("KG_WORKER_0_GEMINI_KEYS", "a, b")
    monkeypatch.setenv("KG_WORKER_0_MISTRAL_KEY", "m0")
    monkeypatch.setenv("KG_WORKER_1_GEMINI_KEYS", "c\nd")
    monkeypatch.setenv("KG_WORKER_1_MISTRAL_KEY", "m1")

    bundles = _collect_worker_keys()
    by_id = {b.worker_id: b for b in bundles}

    assert by_id[0].gemini_keys == ["a", "b"]
    assert by_id[1].gemini_keys == ["c", "d"]
    assert by_id[1].mistral_key == "m1"


def test_bundles_come_back_ordered_by_worker_id(monkeypatch):
    for worker_id in (3, 0, 2, 1):
        monkeypatch.setenv(f"KG_WORKER_{worker_id}_MISTRAL_KEY", f"m{worker_id}")

    assert [b.worker_id for b in _collect_worker_keys()] == [0, 1, 2, 3]


def test_a_fifth_worker_needs_no_code_change(monkeypatch):
    monkeypatch.setenv("KG_WORKER_4_GEMINI_KEYS", "e,f")
    monkeypatch.setenv("KG_WORKER_4_MISTRAL_KEY", "m4")

    assert any(b.worker_id == 4 and b.is_complete for b in _collect_worker_keys())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a,b", ["a", "b"]),
        ("a b", ["a", "b"]),
        ("a\nb", ["a", "b"]),
        ("a, a, b", ["a", "b"]),
        ("  ", []),
        ("", []),
    ],
)
def test_split_keys_handles_the_separators_and_duplicates(raw, expected):
    assert split_keys(raw) == expected
