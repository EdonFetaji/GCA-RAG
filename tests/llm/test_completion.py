"""
`complete()` — the plain-text call every adapter inherits.

No network: the mixin is exercised over a stub standing in for the LangChain
chat model at `self._llm`, which is the only thing it touches. What is being
tested is the normalization, not any vendor's endpoint.

The interesting part is what comes back. `.content` is a string for most
providers and a list of content blocks for the ones with a reasoning or
multi-part response shape, and the grader's parse rule reads a blank body as
"nothing to fix" — so an empty completion has to fail loudly here rather than
silently converging a run on a model that never spoke.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.llm.base import LLMCompletionError, TextLLMClient
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin


class FakeChatModel:
    """The LangChain seam: records the messages, returns a scripted reply."""

    def __init__(self, content: object) -> None:
        self._content = content
        self.calls: list[list] = []

    def invoke(self, messages, **_kwargs):
        self.calls.append(messages)
        if isinstance(self._content, Exception):
            raise self._content
        return type("AIMessage", (), {"content": self._content})()


class FakeClient(LangChainToolLoopMixin):
    """An adapter that is nothing but the mixin and a chat model."""

    def __init__(self, content: object) -> None:
        self._llm = FakeChatModel(content)
        self._model_name = "fake/model-1"


def complete(content: object) -> str:
    return FakeClient(content).complete(system="be terse", user="ping")


# ── The contract ──────────────────────────────────────────────────────


def test_an_adapter_with_the_mixin_satisfies_the_port():
    assert isinstance(FakeClient("ok"), TextLLMClient)


def test_the_system_and_user_prompts_are_sent_in_order():
    client = FakeClient("ok")
    client.complete(system="be terse", user="ping")

    system, user = client._llm.calls[0]
    assert system.content == "be terse"
    assert user.content == "ping"


# ── Content normalization ─────────────────────────────────────────────


def test_string_content_comes_back_as_is():
    assert complete("## High severity\n\n- **WRONG_TYPE** `e1`") == (
        "## High severity\n\n- **WRONG_TYPE** `e1`"
    )


def test_block_content_is_flattened():
    blocks = [{"type": "text", "text": "## High"}, {"type": "text", "text": " severity"}]

    assert complete(blocks) == "## High severity"


def test_a_thinking_block_is_not_part_of_the_answer():
    """
    Concatenating it would put the model's scratch work into the report that is
    handed to the extractor as repair instructions.
    """
    blocks = [
        {"type": "thinking", "thinking": "let me check the quotes"},
        {"type": "text", "text": "CONVERGED"},
    ]

    assert complete(blocks) == "CONVERGED"


def test_plain_string_blocks_are_accepted():
    assert complete(["CON", "VERGED"]) == "CONVERGED"


# ── Failure paths ─────────────────────────────────────────────────────


def test_a_provider_failure_is_wrapped():
    with pytest.raises(LLMCompletionError) as exc:
        complete(RuntimeError("provider exploded"))

    assert "fake/model-1" in str(exc.value)
    assert "provider exploded" in str(exc.value)


@pytest.mark.parametrize("content", ["", "   \n ", [], [{"type": "thinking", "thinking": "hm"}]])
def test_an_empty_completion_is_an_error_not_an_answer(content):
    """Otherwise the grader's parse rule reads silence as a converged graph."""
    with pytest.raises(LLMCompletionError, match="empty completion"):
        complete(content)
