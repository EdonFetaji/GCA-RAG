"""
The `mistral` provider — registration, key handling, and parameter passthrough.

No network: `ChatMistralAI` resolves its config at construction time and only
opens a connection on `invoke`, so building the client with a dummy key is
enough to assert that the right values reached the right constructor keywords.
What is being tested is the wiring, not La Plateforme's endpoint.

The exception is `test_pipeline_schemas_survive_strict_conversion`, which tests
something real about Mistral rather than about us: `json_schema` decodes under
OpenAI-style strict rules, and a schema that violates them is rejected by the
endpoint at request time. Converting locally catches that in the suite instead
of on the first live call.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.base import LLMClient, LLMError, ToolCallingLLMClient
from kg_agentic_extraction.llm.factory import available_providers, build_llm
from kg_agentic_extraction.llm.mistral_client import MistralClient
from kg_agentic_extraction.models.grading import GraderReport
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph


def _settings(**overrides) -> PipelineSettings:
    """Settings for a `mistral` run, with the environment deliberately not consulted."""
    base = {
        "llm_provider": "mistral",
        "model": "mistral-large-latest",
        "mistral_api_key": "sk-test",
    }
    return PipelineSettings(**{**base, **overrides})


def test_mistral_is_registered():
    assert "mistral" in available_providers()


def test_build_llm_returns_a_mistral_client():
    client = build_llm(_settings())
    assert isinstance(client, MistralClient)
    assert client.model_name == "mistral-large-latest"


def test_satisfies_both_ports():
    """The grounder needs the tool-calling port; the extractor and grader need the narrow one."""
    client = build_llm(_settings())
    assert isinstance(client, LLMClient)
    assert isinstance(client, ToolCallingLLMClient)


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="MISTRAL_API_KEY"):
        build_llm(_settings(mistral_api_key=""))


def test_settings_reach_the_underlying_model():
    """
    `api_key` and `base_url` are populate-by-name aliases, so this also pins the
    thing most likely to break on a package upgrade: that they still land on
    `mistral_api_key` and `endpoint`.
    """
    client = build_llm(_settings(max_tokens=4096, temperature=0.3))
    llm = client._llm
    assert llm.endpoint == "https://api.mistral.ai/v1"
    assert llm.mistral_api_key.get_secret_value() == "sk-test"
    assert llm.max_tokens == 4096
    assert llm.temperature == 0.3


def test_base_url_is_overridable():
    client = build_llm(_settings(mistral_base_url="http://localhost:9000/v1"))
    assert client._llm.endpoint == "http://localhost:9000/v1"


def test_temperature_default_is_not_the_packages():
    """ChatMistralAI defaults to 0.7; an extraction pipeline must not inherit that."""
    assert build_llm(_settings())._llm.temperature == 0.0


def test_timeout_default_is_not_the_packages():
    """
    Regression: at the package's 120s default, a real extraction (16k completion
    tokens, strict decoding, free-tier queue) times out mid-generation and the
    ReadTimeout surfaces as an LLMStructuredOutputError that reads like a schema
    rejection. Anything at or below 120 here means that bug is back.
    """
    llm = build_llm(_settings())._llm
    assert llm.timeout == 600
    assert llm.timeout > 120


def test_timeout_is_overridable():
    assert build_llm(_settings(mistral_timeout_seconds=90))._llm.timeout == 90


@pytest.mark.parametrize("method", ["json_schema", "function_calling"])
def test_structured_method_is_configurable(method):
    """
    Both `structured()` and the tool loop's closing call go through
    `_structured_runnable`, so asserting on it covers both paths.

    The chat model is swapped wholesale rather than monkeypatched: `ChatMistralAI`
    is a Pydantic model and rejects attribute assignment for names that are not
    fields.
    """
    client = build_llm(_settings(mistral_structured_method=method))
    seen: list[str] = []

    class Recorder:
        def with_structured_output(self, schema, **kwargs):  # noqa: ARG002
            seen.append(kwargs["method"])
            return object()

    client._llm = Recorder()
    client._structured_runnable(PipelineSettings)
    assert seen == [method]


def test_default_structured_method_is_json_schema():
    """Not the package default (`function_calling`) — see MistralClient's docstring."""
    assert _settings().mistral_structured_method == "json_schema"


@pytest.mark.parametrize("schema", [KnowledgeGraph, GraderReport])
def test_pipeline_schemas_survive_strict_conversion(schema):
    """
    `method="json_schema"` converts with `strict=True`. A schema that cannot be
    converted that way is a 400 from Mistral, not a fallback — so the two models
    the pipeline actually sends are checked here.
    """
    from langchain_mistralai.chat_models import _convert_to_openai_response_format

    response_format = _convert_to_openai_response_format(schema, strict=True)
    assert response_format["json_schema"]["schema"]["type"] == "object"
