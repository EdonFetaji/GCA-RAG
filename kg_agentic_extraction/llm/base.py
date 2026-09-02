"""
The LLM port.

Agents depend on this Protocol, never on a concrete SDK. Structured output is
part of the port rather than something each agent hand-rolls with JSON parsing
and repair retries — the provider knows how to constrain its own decoding
better than a regex ever will.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

TModel = TypeVar("TModel", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """
    Minimal surface an agent needs from a language model.

    Deliberately narrow (ISP): no streaming, no token counting, no tool
    binding. Capabilities beyond this belong on a separate protocol that only
    the agents needing them depend on.
    """

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TModel],
    ) -> TModel:
        """
        Invoke the model and return an instance of `schema`.

        Implementations are responsible for constraining decoding and for
        raising if a valid instance cannot be produced — callers may assume the
        return value is already validated.

        Raises
        ------
        LLMStructuredOutputError
            If the provider could not produce an instance of `schema`.
        """
        ...


@dataclass(frozen=True)
class ToolSpec:
    """One tool as advertised to the model."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolInvocation:
    """One tool call the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ToolCallingLLMClient(Protocol):
    """
    An `LLMClient` that can also run a tool-use loop.

    A separate protocol on purpose. Only the grounder needs tools, so only the
    grounder should have to depend on a client that provides them — the
    extractor and grader keep depending on the narrower `LLMClient`.

    The whole loop sits behind one method rather than exposing message
    bookkeeping, because that bookkeeping is provider-specific: what an
    assistant turn with tool calls looks like, and how a tool result is fed
    back, differs by vendor. The caller supplies only what is genuinely its
    own — how to execute a tool, and what shape the final answer takes.
    """

    def structured(self, *, system: str, user: str, schema: type[TModel]) -> TModel: ...

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
        """
        Let the model call tools, then answer under `schema`.

        `execute` runs one tool call and returns its result as text; it must not
        raise, since an exception mid-loop discards the transcript built so far.
        The loop stops when the model asks for no more tools or `max_rounds` is
        reached, and the final answer is produced by a separate constrained call
        over the accumulated transcript.

        With an empty `tools` list this degrades to a plain `structured()` call
        rather than failing, so a disabled backend is not an error.

        Raises
        ------
        LLMStructuredOutputError
            If the provider could not produce an instance of `schema`.
        """
        ...


class LLMError(RuntimeError):
    """Base for every failure originating in the LLM layer."""


class LLMStructuredOutputError(LLMError):
    """The provider returned something that would not validate against the schema."""

    def __init__(self, schema: type[BaseModel], cause: Exception) -> None:
        super().__init__(f"could not produce a valid {schema.__name__}: {cause}")
        self.schema = schema
        self.cause = cause
