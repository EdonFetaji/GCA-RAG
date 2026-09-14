"""
Cerebras adapter for the `LLMClient` port, via LangChain's ChatCerebras.

The only file in the pipeline that names a specific provider. Swapping to
another vendor means writing a sibling adapter and registering it in
`factory.py` — no agent changes.
"""

from __future__ import annotations

import logging
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class CerebrasClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by Cerebras.

    Structured decoding is delegated to LangChain's `with_structured_output`,
    which pins the provider's JSON-schema mode to the Pydantic model. That is
    why this class has no fence-stripping or repair-retry logic — malformed
    JSON is prevented at decode time rather than repaired afterwards.
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
        from langchain_cerebras import ChatCerebras

        self._model_name = model
        self._llm = ChatCerebras(
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
        logger.debug("Cerebras call — model=%s schema=%s", self._model_name, schema.__name__)
        runnable = self._llm.with_structured_output(schema)
        try:
            result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:  # provider/validation failures alike
            raise LLMStructuredOutputError(schema, exc) from exc

        if not isinstance(result, schema):
            raise LLMStructuredOutputError(
                schema, TypeError(f"provider returned {type(result).__name__}")
            )
        return result
