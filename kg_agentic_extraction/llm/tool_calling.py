"""
The behaviour every LangChain-backed adapter shares — the tool-use loop, and the
plain-text completion.

Every provider adapter wraps a LangChain `BaseChatModel`, which normalises tool
binding, tool-call parsing and message content across vendors. So both belong
here, as a mixin, rather than once per adapter — a new adapter gets tool calling
and `complete()` by inheriting it and needs no code of its own.

The loop is deliberately plain: bind, invoke, execute, repeat. What is *not*
plain, and is the reason this file exists rather than a five-line inline loop:

- **The final answer is produced by a separate, unbound call.** Asking for
  structured output while tools are bound puts two competing constraints on one
  generation. Groq's adapter already pins `method="json_schema"` precisely to
  keep structured output off the tool-call path; binding tools would undo that.
- **A tool that raises must not kill the run.** `execute` is contracted not to
  raise, but a backend bug should still degrade to an error the model can read
  rather than lose a transcript that cost several round-trips.
- **Hitting `max_rounds` is not a failure.** The model is told the budget is
  spent and asked to answer with what it has.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import (
    LLMCompletionError,
    LLMStructuredOutputError,
    ToolInvocation,
    ToolSpec,
)

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: Appended when the tool budget runs out, so the model knows why it is being
#: asked to conclude rather than silently losing its remaining plan.
_BUDGET_SPENT = (
    "You have used your entire tool budget. Answer now with what the tools "
    "returned. Leave anything you could not verify unresolved, with a reason — "
    "do not guess a URI to fill the gap."
)

_EMPTY_COMPLETION_HINT = (
    "the model returned an empty completion — usually the response budget being "
    "spent before any visible text, so raise KG_MAX_TOKENS or turn thinking off"
)

_FINALIZE = (
    "Now produce the final answer, using only what the tool results above "
    "established. Every URI you emit must have appeared in a tool result."
)


class LangChainToolLoopMixin:
    """
    `run_tool_loop` and `complete` for any adapter holding a LangChain chat
    model at `self._llm`.

    Mixed into every adapter — Cerebras, Groq, Gemini, Meta, NVIDIA — each of
    which already satisfies the rest of `ToolCallingLLMClient` through
    `structured()`.
    """

    _llm: Any
    _model_name: str

    def run_tool_loop(
        self,
        *,
        system: str,
        user: str,
        tools: list[ToolSpec],
        execute: Callable[[ToolInvocation], str],
        schema: type[TModel],
        max_rounds: int = 8,
    ) -> TModel:
        """Let the model call tools, then answer under `schema`. See the port for the contract."""
        if not tools:
            # No backend, or a backend that publishes nothing. A toolless run is
            # a worse answer, not an error — the model still has the graph text.
            logger.info("[tool loop] no tools available; falling back to a direct call")
            return self.structured(system=system, user=user, schema=schema)  # type: ignore[attr-defined]

        bound = self._llm.bind_tools([_as_langchain_tool(t) for t in tools])
        messages: list[Any] = [SystemMessage(content=system), HumanMessage(content=user)]
        calls_made = 0

        for round_index in range(max_rounds):
            try:
                reply = bound.invoke(messages)
            except Exception as exc:
                # A provider failure mid-loop still leaves a usable transcript.
                # Breaking out to the final call salvages the tool results
                # already gathered instead of losing the whole pass.
                logger.warning("[tool loop] round %d failed: %s", round_index + 1, exc)
                break

            messages.append(reply)
            invocations = _tool_calls(reply)
            if not invocations:
                break

            calls_made += len(invocations)
            logger.info(
                "[tool loop] round %d — %s",
                round_index + 1,
                ", ".join(f"{i.name}({_brief(i.arguments)})" for i in invocations),
            )
            for invocation in invocations:
                messages.append(
                    ToolMessage(
                        content=_execute_safely(execute, invocation),
                        tool_call_id=invocation.id,
                    )
                )
        else:
            messages.append(HumanMessage(content=_BUDGET_SPENT))

        logger.info(
            "[tool loop] finished after %d tool call(s); requesting %s",
            calls_made,
            schema.__name__,
        )
        return self._finalize(messages, schema)

    # ── Plain text ────────────────────────────────────────────────────

    def complete(self, *, system: str, user: str) -> str:
        """Invoke the model with no schema and return its reply; see `TextLLMClient`."""
        logger.debug("completion call — model=%s", getattr(self, "_model_name", "?"))
        try:
            message = self._llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        except Exception as exc:
            raise LLMCompletionError(getattr(self, "_model_name", "?"), exc) from exc

        text = _content_to_text(message.content)
        if not text.strip():
            # An empty completion is a failure, not an answer. The grader's
            # parse rule reads a blank report as "nothing to fix", so letting
            # this through would silently converge a run on a model that never
            # spoke — usually a spent token budget.
            raise LLMCompletionError(
                getattr(self, "_model_name", "?"), ValueError(_EMPTY_COMPLETION_HINT)
            )
        return text

    # ── Internals ─────────────────────────────────────────────────────

    def _finalize(self, messages: list[Any], schema: type[TModel]) -> TModel:
        """One constrained call over the transcript, with tools deliberately unbound."""
        runnable = self._structured_runnable(schema)
        try:
            result = runnable.invoke([*messages, HumanMessage(content=_FINALIZE)])
        except Exception as exc:
            raise LLMStructuredOutputError(schema, exc) from exc

        return self._validate(result, schema)

    def _validate(self, result: Any, schema: type[TModel]) -> TModel:
        """
        Check that the runnable produced a `schema`, and say so usefully if not.

        A seam, not ceremony: most adapters raise on a bad generation, but
        NVIDIA's structured runnable returns `None` for one instead, and only
        that adapter knows what its own `None` means. Overriding here covers
        the tool loop's closing call as well as `structured()`.
        """
        if not isinstance(result, schema):
            raise LLMStructuredOutputError(
                schema, TypeError(f"provider returned {type(result).__name__}")
            )
        return result

    def _structured_runnable(self, schema: type[TModel]) -> Any:
        """
        Structured-output runnable for the final call.

        Overridden by adapters that need a non-default method — `GroqClient`
        does, for the reason documented on its `structured()`.
        """
        return self._llm.with_structured_output(schema)


# ── Helpers ───────────────────────────────────────────────────────────


def _content_to_text(content: Any) -> str:
    """
    Flatten a LangChain message's `content` to a string.

    Most providers put a plain string there, but the ones with a reasoning or
    multi-part response shape put a list of blocks instead. Only the text blocks
    are kept — a thinking block is not part of the answer, and concatenating it
    would put the model's scratch work into the grader's report.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        ]
        return "".join(parts)
    return str(content or "")


def _as_langchain_tool(spec: ToolSpec) -> dict[str, Any]:
    """
    A `ToolSpec` in the shape `bind_tools` accepts.

    LangChain recognises the OpenAI function schema across every provider it
    supports and translates it to each vendor's own format, so emitting that is
    what keeps this mixin provider-agnostic.
    """
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema or {"type": "object", "properties": {}},
        },
    }


def _tool_calls(message: AIMessage) -> list[ToolInvocation]:
    """The tool calls on an assistant turn, normalised."""
    invocations: list[ToolInvocation] = []
    for index, call in enumerate(getattr(message, "tool_calls", None) or []):
        name = call.get("name") or ""
        if not name:
            continue
        invocations.append(
            ToolInvocation(
                # LangChain types the id as optional and some providers leave it
                # unset; ToolMessage needs one to correlate the result, and the
                # index is stable within this turn.
                id=str(call.get("id") or f"call_{index}"),
                name=name,
                arguments=call.get("args") or {},
            )
        )
    return invocations


def _execute_safely(execute: Callable[[ToolInvocation], str], invocation: ToolInvocation) -> str:
    """Run one tool call, turning any escaping exception into a result the model can read."""
    try:
        return execute(invocation)
    except Exception as exc:
        logger.warning("[tool loop] %s raised: %s", invocation.name, exc)
        return json.dumps({"error": str(exc), "tool": invocation.name})


def _brief(arguments: dict[str, Any]) -> str:
    """Compact argument rendering for the log line."""
    rendered = ", ".join(f"{k}={str(v)[:40]!r}" for k, v in arguments.items())
    return rendered[:120]
