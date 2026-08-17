"""
Groq adapter for the `LLMClient` port, via LangChain's ChatGroq.

Sibling of `cerebras_client.py` — same shape, different vendor. Adding it
required no change to any agent, node, or graph: `llm/factory.py` registers it
and `KG_LLM_PROVIDER=groq` selects it.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class GroqClient:
    """
    `LLMClient` implementation backed by Groq.

    Structured decoding is delegated to LangChain's `with_structured_output`,
    which pins the provider's JSON-schema mode to the Pydantic model — so, as
    with the Cerebras adapter, there is no fence-stripping or repair-retry
    logic here. Malformed JSON is prevented at decode time.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> None:
        # Imported lazily so that merely importing the pipeline (to inspect the
        # graph, run unit tests with a fake client, etc.) does not require the
        # provider SDK to be installed or an API key to be present.
        from langchain_groq import ChatGroq

        self._model_name = model
        self._llm = ChatGroq(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TModel],
    ) -> TModel:
        """Invoke the model and return a validated `schema` instance."""
        logger.debug("Groq call — model=%s schema=%s", self._model_name, schema.__name__)
        # json_schema, not the default function_calling: Groq wraps the schema
        # in a tool call under that method, and any truncated generation then
        # fails tool-call parsing with an opaque 400 tool_use_failed rather
        # than surfacing as a length problem.
        runnable = self._llm.with_structured_output(schema, method="json_schema")
        try:
            result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:  # provider/validation failures alike
            raise LLMStructuredOutputError(schema, exc) from exc

        if not isinstance(result, schema):
            raise LLMStructuredOutputError(
                schema, TypeError(f"provider returned {type(result).__name__}")
            )
        return result
