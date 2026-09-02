"""
The `nvidia` provider — registration, key handling, and parameter passthrough.

No network: `ChatNVIDIA` resolves its config at construction time and only
opens a connection on `invoke`, so building the client with a dummy key is
enough to assert that the right values reached the right constructor keywords.
What is being tested is the wiring, not NVIDIA's endpoint.

One exception is deliberate — `with_structured_output` *does* reach out (it
fetches the model catalog to decide whether to warn), so nothing here calls it.
The `None`-handling tests go through `_validate` instead, which is the seam
both `structured()` and the tool loop's closing call share.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.base import (
    LLMClient,
    LLMError,
    LLMStructuredOutputError,
    ToolCallingLLMClient,
)
from kg_agentic_extraction.llm.factory import available_providers, build_llm
from kg_agentic_extraction.llm.nvidia_client import NvidiaClient


class _Schema(BaseModel):
    value: str = "x"


def _settings(**overrides) -> PipelineSettings:
    """Settings for an `nvidia` run, with the environment deliberately not consulted."""
    base = {
        "llm_provider": "nvidia",
        "model": "google/gemma-4-31b-it",
        "nvidia_api_key": "nvapi-test",
    }
    return PipelineSettings(**{**base, **overrides})


def test_nvidia_is_registered():
    assert "nvidia" in available_providers()


def test_build_llm_returns_an_nvidia_client():
    client = build_llm(_settings())
    assert isinstance(client, NvidiaClient)
    assert client.model_name == "google/gemma-4-31b-it"


def test_satisfies_both_ports():
    """The grounder needs the tool-calling port; the extractor and grader need the narrow one."""
    client = build_llm(_settings())
    assert isinstance(client, LLMClient)
    assert isinstance(client, ToolCallingLLMClient)


def test_missing_key_is_a_clear_error():
    with pytest.raises(LLMError, match="NVIDIA_API_KEY"):
        build_llm(_settings(nvidia_api_key=""))


def test_settings_reach_the_underlying_model():
    """
    Note the keyword/field mismatch this covers: the constructor takes
    `max_completion_tokens`, the field it populates is `max_tokens`.
    """
    client = build_llm(_settings(max_tokens=4096, temperature=0.3))
    llm = client._llm
    assert llm.base_url == "https://integrate.api.nvidia.com/v1"
    assert llm.max_tokens == 4096
    assert llm.temperature == 0.3


def test_base_url_is_overridable():
    """A self-hosted NIM container, which also flips ChatNVIDIA off its hosted path."""
    client = build_llm(_settings(nvidia_base_url="http://localhost:8000/v1"))
    assert client._llm.base_url == "http://localhost:8000/v1"


def test_optional_parameters_are_omitted_by_default():
    """
    An endpoint that does not know a parameter rejects the whole request, so
    the defaults must send nothing rather than a null or a false.
    """
    llm = build_llm(_settings())._llm
    assert llm.top_p is None
    assert "chat_template_kwargs" not in llm.model_kwargs


def test_top_p_is_forwarded_when_set():
    assert build_llm(_settings(nvidia_top_p=0.95))._llm.top_p == 0.95


def test_thinking_rides_in_chat_template_kwargs():
    """Not a first-class field — the toggle lives in the model's chat template."""
    llm = build_llm(_settings(nvidia_enable_thinking=True))._llm
    assert llm.model_kwargs["chat_template_kwargs"] == {"enable_thinking": True}


def test_none_result_names_the_token_budget():
    """
    NVIDIA's parser is forgiving: a completion cut off before the schema can be
    built yields `None` from every fallback shape rather than an exception. It
    must not surface as a bare `NoneType`.
    """
    client = build_llm(_settings())
    with pytest.raises(LLMStructuredOutputError, match="KG_MAX_TOKENS"):
        client._validate(None, _Schema)


def test_wrong_type_still_reports_the_type():
    client = build_llm(_settings())
    with pytest.raises(LLMStructuredOutputError, match="returned dict"):
        client._validate({"value": "x"}, _Schema)


def test_valid_result_passes_through():
    client = build_llm(_settings())
    instance = _Schema(value="ok")
    assert client._validate(instance, _Schema) is instance
