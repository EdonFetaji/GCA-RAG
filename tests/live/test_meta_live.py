"""
Meta Model API, for real. Deselected by default — run with `pytest -m live`.

This exists to settle one question a stub cannot answer: whether Meta's
OpenAI-compatible endpoint actually implements `response_format: json_schema`,
which Meta documents tool calling for but not structured output. If
`test_structured_output_works` fails here with a 400 from the endpoint, set
`KG_META_STRUCTURED_METHOD=function_calling` and re-run — a pass in that mode
is the answer, and `MetaClient._structured_runnable` documents what it costs.

Skipped, not failed, when no key is present: not everyone running the suite has
Meta access.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.base import ToolInvocation, ToolSpec
from kg_agentic_extraction.llm.factory import build_llm

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")),
        reason="META_API_KEY / MODEL_API_KEY not set",
    ),
]

MODEL = os.environ.get("KG_MODEL_META", "muse-spark-1.2-contributor")


class Capital(BaseModel):
    """Deliberately shaped like the pipeline's schemas: nested, typed, required."""

    city: str = Field(description="The capital city.")
    country: str = Field(description="The country it is the capital of.")
    population_millions: float = Field(description="Approximate population, in millions.")


@pytest.fixture
def client():
    return build_llm(PipelineSettings(llm_provider="meta", model=MODEL))


def test_structured_output_works(client):
    """The load-bearing assertion: `with_structured_output` against a real Muse Spark."""
    result = client.structured(
        system="You answer with facts only.",
        user="What is the capital of France?",
        schema=Capital,
    )
    assert isinstance(result, Capital)
    assert "paris" in result.city.lower()
    assert result.population_millions > 0


def test_tool_loop_works(client):
    """The grounder's path: bind a tool, let the model call it, answer under a schema."""
    calls: list[str] = []

    def execute(invocation: ToolInvocation) -> str:
        calls.append(invocation.name)
        return '{"city": "Ulaanbaatar", "country": "Mongolia", "population_millions": 1.6}'

    tool = ToolSpec(
        name="lookup_capital",
        description="Look up the capital city of a country.",
        input_schema={
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
        },
    )
    result = client.run_tool_loop(
        system="Use the tool to answer. Do not answer from memory.",
        user="What is the capital of Mongolia?",
        tools=[tool],
        execute=execute,
        schema=Capital,
        max_rounds=3,
    )
    assert calls == ["lookup_capital"]
    assert "ulaanbaatar" in result.city.lower()


def test_reasoning_effort_is_accepted(client):
    """
    A reasoning model that rejects the parameter would 400 the whole request,
    so this guards the default in `PipelineSettings.meta_reasoning_effort`.
    """
    assert client._llm.reasoning_effort == "low"
    result = client.structured(system="Be brief.", user="Capital of Japan?", schema=Capital)
    assert "tokyo" in result.city.lower()
