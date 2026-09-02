"""
The `meta` provider — registration, key handling, and parameter passthrough.

No network: `ChatOpenAI` resolves its config at construction time and only
opens a connection on `invoke`, so building the client with a dummy key is
enough to assert that the right values reached the right constructor keywords.
What is being tested is the wiring, not Meta's endpoint.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.base import LLMClient, LLMError, ToolCallingLLMClient
from kg_agentic_extraction.llm.factory import available_providers, build_llm
from kg_agentic_extraction.llm.meta_client import MetaClient


def _settings(**overrides) -> PipelineSettings:
    """Settings for a `meta` run, with the environment deliberately not consulted."""
    base = {
        "llm_provider": "meta",
        "model": "muse-spark-1.2-contributor",
        "meta_api_key": "sk-test",
    }
    return PipelineSettings(**{**base, **overrides})


def test_meta_is_registered():
    assert "meta" in available_providers()


def test_build_llm_returns_a_meta_client():
    client = build_llm(_settings())
    assert isinstance(client, MetaClient)
    assert client.model_name == "muse-spark-1.2-contributor"


def test_satisfies_both_ports():
    """The grounder needs the tool-calling port; the extractor and grader need the narrow one."""
    client = build_llm(_settings())
    assert isinstance(client, LLMClient)
    assert isinstance(client, ToolCallingLLMClient)


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="META_API_KEY"):
        build_llm(_settings(meta_api_key=""))


def test_settings_reach_the_underlying_model():
    client = build_llm(_settings(max_tokens=4096, temperature=0.3))
    llm = client._llm
    assert llm.openai_api_base == "https://api.meta.ai/v1"
    assert llm.max_tokens == 4096
    assert llm.temperature == 0.3
    # A first-class ChatOpenAI field, not a model_kwargs passenger — so it is
    # sent as a real request parameter rather than smuggled through.
    assert llm.reasoning_effort == "low"


def test_base_url_is_overridable():
    client = build_llm(_settings(meta_base_url="http://localhost:9000/v1"))
    assert client._llm.openai_api_base == "http://localhost:9000/v1"


def test_reasoning_effort_is_omitted_when_unset():
    """
    An endpoint that does not know the parameter rejects the whole request, so
    `None` must send nothing rather than a null.
    """
    client = build_llm(_settings(meta_reasoning_effort=None))
    assert client._llm.reasoning_effort is None
    assert "reasoning_effort" not in client._llm.model_kwargs


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_structured_method_is_configurable(method):
    """
    Both `structured()` and the tool loop's closing call go through
    `_structured_runnable`, so asserting on it covers both paths.

    The chat model is swapped wholesale rather than monkeypatched: `ChatOpenAI`
    is a Pydantic model and rejects attribute assignment for names that are not
    fields.
    """
    client = build_llm(_settings(meta_structured_method=method))
    seen: list[str] = []

    class Recorder:
        def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
            seen.append(kwargs["method"])
            return object()

    client._llm = Recorder()
    client._structured_runnable(PipelineSettings)
    assert seen == [method]


def test_meta_key_aliases(monkeypatch):
    """META_API_KEY is preferred; Meta's own generic MODEL_API_KEY still works."""
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_API_KEY", "from-meta-docs")
    assert PipelineSettings(_env_file=None).meta_api_key == "from-meta-docs"

    monkeypatch.setenv("META_API_KEY", "unambiguous")
    assert PipelineSettings(_env_file=None).meta_api_key == "unambiguous"
