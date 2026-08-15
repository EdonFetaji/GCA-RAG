"""
The agent base class.

Every agent does the same three things in the same order: render its prompts,
call the model under a schema, post-process the result. That skeleton lives
here as a Template Method; subclasses fill in only the parts that differ.

Agents know nothing about LangGraph. They take plain inputs, return plain
models, and can be exercised in a unit test with a stub `LLMClient` and no
graph, no state, and no network.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMClient
from kg_agentic_extraction.prompts.registry import PromptRegistry

logger = logging.getLogger(__name__)


class Agent[TInput, TOutput: BaseModel](ABC):
    """
    Base for every agent in the pipeline.

    Subclasses declare `name` (which also selects their template folder) and
    `output_schema`, then implement `build_context()` to supply template
    variables. Everything else is inherited.
    """

    #: Template folder under `prompts/templates/`, e.g. "extractor".
    name: str

    #: Which user-side template to render. Subclasses may vary it per call by
    #: overriding `user_role_for()`.
    default_user_role: str = "user"

    def __init__(
        self,
        *,
        llm: LLMClient,
        prompts: PromptRegistry,
        prompt_version: str | None = None,
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._prompt_version = prompt_version

    # ── Template Method ───────────────────────────────────────────────

    def run(self, payload: TInput) -> TOutput:
        """
        Execute the agent. Do not override — override the hooks instead.

        render context → render prompts → structured LLM call → post-process
        """
        context = self.build_context(payload)
        rendered = self._prompts.render_pair(
            self.name,
            user_role=self.user_role_for(payload),
            version=self._prompt_version,
            **context,
        )
        logger.info("[%s] calling model (schema=%s)", self.name, self.output_schema.__name__)
        result = self._llm.structured(
            system=rendered.system,
            user=rendered.user,
            schema=self.output_schema,
        )
        return self.post_process(result, payload)

    # ── Hooks ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def output_schema(self) -> type[TOutput]:
        """The Pydantic model the LLM is constrained to produce."""

    @abstractmethod
    def build_context(self, payload: TInput) -> dict[str, object]:
        """Template variables for this agent's prompts."""

    def user_role_for(self, payload: TInput) -> str:  # noqa: ARG002
        """
        Which user-side template to use for this payload.

        Overridden by the extractor to switch between first-pass extraction and
        grader-driven repair without needing two agent classes.
        """
        return self.default_user_role

    def post_process(self, result: TOutput, payload: TInput) -> TOutput:  # noqa: ARG002
        """Last chance to normalize or enrich the model's output. Default: pass through."""
        return result
