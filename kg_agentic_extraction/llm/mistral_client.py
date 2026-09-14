"""
Mistral La Plateforme adapter for the `LLMClient` port, via LangChain's ChatMistralAI.

Sixth sibling of `cerebras_client.py` / `groq_client.py` / `gemini_client.py` /
`meta_client.py` / `nvidia_client.py`. It uses the vendor package
(`langchain-mistralai`) rather than `ChatOpenAI` pointed at Mistral's base:
Mistral's chat API is OpenAI-shaped but not OpenAI-identical — the parameter is
`endpoint`, the safety toggle is `safe_mode`, and the structured-output
converter has its own strictness rules — and `ChatMistralAI` already knows all
three.

Two things about Mistral that the other adapters do not have to think about:

- **The package's default structured mode is the wrong one here.**
  `with_structured_output` defaults to `function_calling`, which is exactly the
  path `GroqClient` pins `json_schema` to avoid: the schema rides inside a tool
  call, so a truncated generation fails tool-call parsing with an opaque error
  instead of surfacing as a length problem. This adapter pins `json_schema` in
  both call paths, and keeps the mode configurable for the reason below.
- **`json_schema` means OpenAI-style *strict* decoding.** LangChain converts the
  Pydantic model with `strict=True`, which requires every property to be listed
  as required and `additionalProperties: false` throughout. `KnowledgeGraph` and
  `GraderReport` both convert cleanly today, but a future field with a default
  could stop doing so — `KG_MISTRAL_STRUCTURED_METHOD=function_calling` is the
  escape hatch if Mistral starts rejecting the schema, at the cost above.

The third thing is the one that actually bites first, so it gets its own
paragraph: **`ChatMistralAI` defaults to a 120-second read timeout**, and this
pipeline's calls do not fit in it. An extraction sends a few thousand prompt
tokens and asks for up to `KG_MAX_TOKENS` (16,384) back under strict decoding;
a grade call carries the graph JSON *and* the documents. On the free tier that
routinely runs past two minutes, and the failure arrives as `httpx.ReadTimeout`
wrapped in `LLMStructuredOutputError` — which reads like a schema problem and is
not one. Hence `DEFAULT_TIMEOUT_SECONDS` below, well above the package default.
Note that it multiplies with `max_retries`: a wholly unresponsive endpoint costs
`timeout × (1 + retries)` before the call gives up.

Free-tier note: La Plateforme's free "Experiment" mode is generous on tokens and
tight on requests per second, which suits this pipeline — the prompts are large
and the call count per cluster is small. `max_retries` matters more here than
elsewhere for that reason: a 429 is a pacing problem, not a failure.
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

#: La Plateforme's chat base. Override for a proxy, a gateway, or a
#: self-deployed Mistral endpoint.
DEFAULT_BASE_URL = "https://api.mistral.ai/v1"

#: Read timeout, in seconds. The package default is 120, which a 16k-token
#: schema-constrained generation on the free tier overruns — see the module
#: docstring. Generous on purpose: waiting is cheaper than re-running the pass.
DEFAULT_TIMEOUT_SECONDS = 600


class MistralClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by Mistral La Plateforme.

    As with the siblings, structured decoding is delegated to LangChain's
    `with_structured_output` rather than hand-rolled JSON parsing — see
    `_structured_runnable` for why the mode is pinned rather than defaulted.
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
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        structured_method: str = "json_schema",
    ) -> None:
        # Imported lazily so that merely importing the pipeline (to inspect the
        # graph, run unit tests with a fake client, etc.) does not require the
        # provider SDK to be installed or an API key to be present.
        from langchain_mistralai import ChatMistralAI

        self._model_name = model
        self._structured_method = structured_method
        self._llm = ChatMistralAI(
            model=model,
            # `api_key` and `base_url` are the populate-by-name aliases of the
            # fields ChatMistralAI actually declares (`mistral_api_key` and
            # `endpoint`). Use the alias: it is what the vendor documents, and
            # it keeps this constructor call the same shape as its siblings'.
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            timeout=timeout,
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
        logger.debug("Mistral call — model=%s schema=%s", self._model_name, schema.__name__)
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

        Pinned rather than left to the package default: `ChatMistralAI` defaults
        to `function_calling`, and the grounder's closing call would then be
        asking for a tool-shaped answer over a transcript in which tools were
        deliberately unbound. See the module docstring for the full reasoning.
        """
        return self._llm.with_structured_output(schema, method=self._structured_method)
