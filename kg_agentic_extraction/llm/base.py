"""
The LLM port.

Agents depend on this Protocol, never on a concrete SDK. Structured output is
part of the port rather than something each agent hand-rolls with JSON parsing
and repair retries — the provider knows how to constrain its own decoding
better than a regex ever will.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

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


class LLMError(RuntimeError):
    """Base for every failure originating in the LLM layer."""


class LLMStructuredOutputError(LLMError):
    """The provider returned something that would not validate against the schema."""

    def __init__(self, schema: type[BaseModel], cause: Exception) -> None:
        super().__init__(f"could not produce a valid {schema.__name__}: {cause}")
        self.schema = schema
        self.cause = cause
