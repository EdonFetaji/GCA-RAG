"""
Google Gemini adapter for the `LLMClient` port, via langchain-google-genai.

Third sibling of `cerebras_client.py` / `groq_client.py`. Same shape, different
vendor — no agent, node, or graph change was needed to add it.

Note the constructor keyword differences: `ChatGoogleGenerativeAI` exposes the
key as `api_key`, the completion cap as `max_tokens` (aliasing its internal
`max_output_tokens`), and the retry count as `retries`. Those aliases are used
below so this adapter presents the same signature as its siblings.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class GeminiClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by Google Gemini.

    `with_structured_output` already defaults to `method="json_schema"` here, so
    unlike the Groq adapter there is nothing to override — Gemini constrains
    decoding to the Pydantic schema natively.
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
        from langchain_google_genai import ChatGoogleGenerativeAI

        self._model_name = model
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=max_retries,
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
        logger.debug("Gemini call — model=%s schema=%s", self._model_name, schema.__name__)
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

    def _structured_runnable(self, schema: type[TModel]) -> Runnable:
        """`json_schema` for the tool loop's closing call, matching `structured()`."""
        return self._llm.with_structured_output(schema, method="json_schema")
