"""
NVIDIA NIM adapter for the `LLMClient` port, via LangChain's ChatNVIDIA.

Fifth sibling of `cerebras_client.py` / `groq_client.py` / `gemini_client.py` /
`meta_client.py`. It uses the vendor package (`langchain-nvidia-ai-endpoints`)
rather than `ChatOpenAI` pointed at NVIDIA's OpenAI-compatible base, because
`ChatNVIDIA` knows things the generic client does not: NIM's `nvext` guided
decoding, the hosted-vs-self-hosted split, and the `chat_template_kwargs`
channel that toggles thinking on models that have it.

Three things about NIM that the other adapters do not have to think about:

- **Structured output has no `method` to pin.** `ChatNVIDIA.with_structured_output`
  ignores the argument (it warns and drops it) and instead tries three request
  shapes in order — OpenAI `response_format`, then `guided_json`, then
  `nvext` — falling back on each failure. So there is no `_structured_runnable`
  override here; the mixin's default is the only correct call.
- **A truncated generation returns `None`, not an exception.** NVIDIA's parser
  is deliberately forgiving: if the completion stops before the schema can be
  constructed, every fallback yields `None` and the runnable returns it. That
  is the `KG_MAX_TOKENS` failure the other adapters surface as a provider
  error, so `structured()` names it explicitly rather than letting it arrive as
  a bare `NoneType`.
- **Thinking is off unless asked for.** Gemma-4-it and friends gate reasoning
  behind `chat_template_kwargs={"enable_thinking": True}`. Left off by design:
  the trap already documented on `PipelineSettings.max_tokens` applies here
  too — reasoning tokens come out of the same completion budget, and
  schema-constrained extraction wants that budget in the answer.

Expect one warning per structured call for a model NVIDIA has not yet added to
the package's static table ("not known to support structured output"). It is a
staleness report about the table, not about the endpoint.
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMStructuredOutputError
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: NVIDIA's hosted NIM catalog. Override for a self-hosted NIM container.
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

#: What `None` from the structured runnable almost always means, given that
#: every fallback shape has to fail for it to be returned at all.
_EMPTY_RESULT_HINT = (
    "NVIDIA returned no parseable object under the schema (all three request "
    "shapes yielded None). This is usually the completion being cut off — "
    "raise KG_MAX_TOKENS, or turn KG_NVIDIA_ENABLE_THINKING off if it is on."
)


class NvidiaClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by NVIDIA NIM (hosted or self-hosted).

    As with the siblings, structured decoding is delegated to LangChain's
    `with_structured_output` rather than hand-rolled JSON parsing — see the
    module docstring for what NVIDIA does differently underneath it.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        top_p: float | None = None,
        enable_thinking: bool = False,
    ) -> None:
        # Imported lazily so that merely importing the pipeline (to inspect the
        # graph, run unit tests with a fake client, etc.) does not require the
        # provider SDK to be installed or an API key to be present.
        from langchain_nvidia_ai_endpoints import ChatNVIDIA

        self._model_name = model
        # Only forwarded when set: an endpoint that does not know a parameter
        # rejects the whole request, so the defaults must send nothing at all.
        extra: dict[str, Any] = {}
        if top_p is not None:
            extra["top_p"] = top_p
        if enable_thinking:
            # Not a first-class ChatNVIDIA field — it rides in model_kwargs and
            # is forwarded to the model's chat template, which is where the
            # thinking toggle actually lives.
            extra["chat_template_kwargs"] = {"enable_thinking": True}

        self._llm = ChatNVIDIA(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            # The constructor keyword is `max_completion_tokens`; the field it
            # populates is `max_tokens`. Use the keyword NVIDIA documents.
            max_completion_tokens=max_tokens,
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
        logger.debug("NVIDIA call — model=%s schema=%s", self._model_name, schema.__name__)
        runnable = self._structured_runnable(schema)
        try:
            result = runnable.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:  # provider/validation failures alike
            raise LLMStructuredOutputError(schema, exc) from exc

        return self._validate(result, schema)

    def _validate(self, result: Any, schema: type[TModel]) -> TModel:
        """
        The mixin's check, plus a name for NVIDIA's `None`.

        Overridden rather than inlined into `structured()` so the grounder's
        tool loop gets the same message — its closing call goes through
        `_finalize`, which would otherwise report the budget failure below as a
        bare `provider returned NoneType`.
        """
        if result is None:
            raise LLMStructuredOutputError(schema, ValueError(_EMPTY_RESULT_HINT))
        return super()._validate(result, schema)
