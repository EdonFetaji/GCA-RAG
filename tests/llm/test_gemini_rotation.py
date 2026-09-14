"""
Gemini API-key rotation.

The adapter holds several keys and moves to the next one when Google returns a
429. A per-day quota marks the key spent for the process; a rate-limit 429 just
rotates. When every key is spent, `AllGeminiKeysExhausted` is raised — and it is
a `BaseException`, so it escapes the pipeline's per-node ``except Exception``.

The real `ChatGoogleGenerativeAI` is never constructed: `_make_llm` is swapped
for a factory of fakes whose `.invoke` raises whatever the test scripts.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError
from kg_agentic_extraction.llm.gemini_client import (
    AllGeminiKeysExhausted,
    GeminiClient,
    _is_daily_quota_error,
    _is_quota_error,
    _retry_after_seconds,
)


class _Schema(BaseModel):
    ok: bool = True


_DAILY_429 = (
    "Error calling model 'gemini-2.5-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "{'error': {'code': 429, 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': "
    "'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}, {'@type': "
    "'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '41s'}]}}"
)
_RATE_429 = (
    "429 RESOURCE_EXHAUSTED. quota exceeded: GenerateRequestsPerMinutePerProjectPerModel, "
    "retryDelay: '7s'"
)


class _FakeRunnable:
    def __init__(self, exc: Exception | None) -> None:
        self._exc = exc

    def invoke(self, _messages):
        if self._exc is not None:
            raise self._exc
        return _Schema()


class _FakeLLM:
    """Stands in for a ChatGoogleGenerativeAI bound to one key."""

    def __init__(self, key: str, script: dict[str, Exception | None]) -> None:
        self.key = key
        self._script = script

    def with_structured_output(self, _schema, **_kw):
        return _FakeRunnable(self._script.get(self.key, None))


def _client(keys: list[str], script: dict[str, Exception | None]) -> GeminiClient:
    client = GeminiClient(model="gemini-2.5-flash", api_keys=keys, max_rotation_wait_seconds=5.0)
    client._make_llm = lambda key: _FakeLLM(key, script)  # type: ignore[assignment]
    client._llm = client._make_llm(keys[0])
    return client


# ── Classifiers ──────────────────────────────────────────────────────


def test_classifiers_read_the_quota_body():
    assert _is_quota_error(RuntimeError(_DAILY_429))
    assert _is_daily_quota_error(RuntimeError(_DAILY_429))
    assert _retry_after_seconds(RuntimeError(_DAILY_429)) == 41.0

    assert _is_quota_error(RuntimeError(_RATE_429))
    assert not _is_daily_quota_error(RuntimeError(_RATE_429))


def test_a_non_quota_error_is_not_swallowed():
    assert not _is_quota_error(ValueError("bad schema"))


def test_classifier_unwraps_the_structured_output_error():
    wrapped = LLMStructuredOutputError(_Schema, RuntimeError(_DAILY_429))
    assert _is_quota_error(wrapped)
    assert _is_daily_quota_error(wrapped)


# ── Rotation ─────────────────────────────────────────────────────────


def test_daily_quota_rotates_to_the_next_key():
    script = {"k1": RuntimeError(_DAILY_429), "k2": None}
    client = _client(["k1", "k2"], script)

    result = client.structured(system="s", user="u", schema=_Schema)

    assert result.ok
    assert client._idx == 1
    assert 0 in client._daily_exhausted


def test_every_key_spent_raises_base_exception():
    script = {k: RuntimeError(_DAILY_429) for k in ("k1", "k2", "k3")}
    client = _client(["k1", "k2", "k3"], script)

    with pytest.raises(AllGeminiKeysExhausted):
        client.structured(system="s", user="u", schema=_Schema)
    # BaseException, not Exception — this is what lets it abort a batch.
    assert not isinstance(AllGeminiKeysExhausted("x"), Exception)


def test_all_keys_rate_limited_sleeps_then_retries(monkeypatch):
    slept: list[float] = []

    # Both keys rate-limited on the first pass, both fine on the retry.
    state = {"pass": 0}

    class _Runnable:
        def __init__(self, key):
            self.key = key

        def invoke(self, _m):
            if state["pass"] == 0:
                raise RuntimeError(_RATE_429)
            return _Schema()

    client = GeminiClient(model="m", api_keys=["k1", "k2"], max_rotation_wait_seconds=60.0)

    class _LLM:
        def __init__(self, key):
            self.key = key

        def with_structured_output(self, _s, **_k):
            return _Runnable(self.key)

    client._make_llm = _LLM  # type: ignore[assignment]
    client._llm = _LLM("k1")

    def _advance_and_flip():
        # after the sleep, the next cycle should succeed
        state["pass"] = 1

    monkeypatch.setattr(
        "kg_agentic_extraction.llm.gemini_client.time.sleep",
        lambda s: (slept.append(s), _advance_and_flip()),
    )

    result = client.structured(system="s", user="u", schema=_Schema)
    assert result.ok
    assert slept  # it did sleep rather than spin
    assert client._daily_exhausted == set()  # rate limits never mark a key spent
