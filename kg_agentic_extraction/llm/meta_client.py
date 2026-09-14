"""
Meta Model API adapter for the `LLMClient` port, via LangChain's ChatOpenAI.

Fourth sibling of `cerebras_client.py` / `groq_client.py` / `gemini_client.py`.
The odd one out in exactly one respect: there is no `langchain-meta` package,
because Meta Model API speaks the OpenAI Chat Completions protocol. `ChatOpenAI`
pointed at `https://api.meta.ai/v1` *is* the Meta client — the dependency name
says OpenAI, the traffic goes to Meta.

Two things about Muse Spark that the other adapters do not have to think about:

- **It is a reasoning model.** Reasoning tokens are billed as output and drawn
  from the same completion budget as the answer, which is the failure already
  documented on `PipelineSettings.max_tokens` for `openai/gpt-oss-*`: the cap
  is spent thinking and the structured call returns empty. `reasoning_effort`
  bounds that directly, and the pipeline pins it low — schema-constrained
  extraction wants tokens in the answer, not in the scratchpad.
- **Its structured-output mode is not settled.** Meta documents tool calling
  and parallel tool calls; it does not document `response_format: json_schema`.
  So the method is a constructor argument rather than a hardcoded literal, and
  `KG_META_STRUCTURED_METHOD=function_calling` is the escape hatch if the
  endpoint rejects the schema mode.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: Meta Model API's OpenAI-compatible base. Overridable for a proxy or gateway.
DEFAULT_BASE_URL = "https://api.meta.ai/v1"


class MetaClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by Meta Model API (Muse Spark).

    As with the siblings, structured decoding is delegated to LangChain's
    `with_structured_output` rather than hand-rolled JSON parsing — see
    `structured()` for why the method is parameterised here and not there.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 2,
        reasoning_effort: str | None = None,
        structured_method: str = "json_schema",
    ) -> None:
        # Imported lazily so that merely importing the pipeline (to inspect the
        # graph, run unit tests with a fake client, etc.) does not require the
        # provider SDK to be installed or an API key to be present.
        from langchain_openai import ChatOpenAI

        self._model_name = model
        self._structured_method = structured_method
        # Only forwarded when set: an endpoint that does not know the parameter
        # rejects the whole request, so the default must send nothing at all.
        extra: dict[str, Any] = {}
        if reasoning_effort:
            extra["reasoning_effort"] = reasoning_effort

        self._llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            **extra,
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
        logger.debug("Meta call — model=%s schema=%s", self._model_name, schema.__name__)
        runnable = self._structured_runnable(schema)
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
        """
        The configured structured-output mode, for both `structured()` and the tool loop.

        Note what `function_calling` costs if you fall back to it: it puts
        structured decoding back on the tool-call path, which is precisely what
        `GroqClient` pins `json_schema` to avoid — a truncated generation then
        surfaces as an opaque tool-parse failure instead of a length problem.
        Prefer raising `KG_MAX_TOKENS` over switching modes.
        """
        return self._llm.with_structured_output(schema, method=self._structured_method)
