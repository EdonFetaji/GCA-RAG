"""
Mistral La Plateforme, for real. Deselected by default — run with `pytest -m live`.

This exists to settle two questions a stub cannot answer:

- Whether Mistral accepts the *strict* JSON schema that `method="json_schema"`
  sends. The offline suite proves the conversion succeeds locally; only the
  endpoint can say whether it likes the result. On a 400 here, set
  `KG_MISTRAL_STRUCTURED_METHOD=function_calling` and re-run — a pass in that
  mode is the answer, and `MistralClient` documents what it costs.
- Whether the tool loop survives the round trip, which is the grounder's path
  and the one place where Mistral's tool-call ids and message ordering could
  differ from what `LangChainToolLoopMixin` assumes.

Skipped, not failed, when no key is present: not everyone running the suite has
a Mistral account.
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
        not os.environ.get("MISTRAL_API_KEY"),
        reason="MISTRAL_API_KEY not set",
    ),
]

MODEL = os.environ.get("KG_MODEL_MISTRAL", "mistral-large-latest")


class Capital(BaseModel):
    """Deliberately shaped like the pipeline's schemas: nested, typed, required."""

    city: str = Field(description="The capital city.")
    country: str = Field(description="The country it is the capital of.")
    population_millions: float = Field(description="Approximate population, in millions.")


@pytest.fixture
def client():
    return build_llm(PipelineSettings(llm_provider="mistral", model=MODEL))


def test_structured_output_works(client):
    """The load-bearing assertion: strict `json_schema` against a real Mistral model."""
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


def test_the_real_extraction_schema_is_accepted(client):
    """
    `Capital` is a toy. `KnowledgeGraph` is what the extractor actually sends —
    nested lists of models, several levels deep — and it is the shape most
    likely to trip strict decoding. A tiny document keeps the call cheap.
    """
    from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph

    result = client.structured(
        system=(
            "Extract a knowledge graph. Every evidence quote must be an exact "
            "substring of the document, and document_index is 0."
        ),
        user="[DOCUMENT 0]\nGavin Newsom is the governor of California.",
        schema=KnowledgeGraph,
    )
    assert isinstance(result, KnowledgeGraph)
    assert result.entities
