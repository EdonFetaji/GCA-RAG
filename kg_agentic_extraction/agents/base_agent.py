"""
The agent base class.

Every agent does the same three things in the same order: render its prompts,
call the model, post-process the result. That skeleton lives here as a Template
Method; subclasses fill in only the parts that differ.

Agents call the model under a schema, which is what `run()` does.
`run_completion()` is the same skeleton without one, for an agent whose answer is
prose rather than an object; both share the rendering step, which is the part
that is genuinely common. Nothing calls it today — the grader did, before its
report went back to being decoded rather than parsed — and it is kept as the
seam that regime needs.

Agents know nothing about LangGraph. They take plain inputs, return plain
models, and can be exercised in a unit test with a stub `LLMClient` and no
graph, no state, and no network.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMClient, LLMError, TextLLMClient
from kg_agentic_extraction.prompts.registry import PromptRegistry, RenderedPrompt

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
        Execute the agent under a schema. Override the hooks, not this.

        render context → render prompts → structured LLM call → post-process
        """
        rendered = self._render(payload)
        logger.info("[%s] calling model (schema=%s)", self.name, self.output_schema.__name__)
        result = self._llm.structured(
            system=rendered.system,
            user=rendered.user,
            schema=self.output_schema,
        )
        return self.post_process(result, payload)

    def run_completion(self, payload: TInput) -> str:
        """
        Render this agent's prompts and answer with unconstrained text.

        The other half of the template method, for an agent whose output is
        prose rather than an object. It stops at the raw string: turning that
        into `TOutput` is the caller's job, because only the caller knows what
        its model's answer is supposed to look like.

        No agent takes this path at present. It is here for the grader, whose
        report is fed verbatim to the extractor and therefore travels as an
        escaped JSON string under a schema — see `TextLLMClient` for when that
        trade stops being worth making.
        """
        if not isinstance(self._llm, TextLLMClient):
            raise LLMError(
                f"[{self.name}] needs a client that can complete plain text, but "
                f"{type(self._llm).__name__} does not implement `complete`"
            )

        rendered = self._render(payload)
        logger.info("[%s] calling model (plain completion)", self.name)
        return self._llm.complete(system=rendered.system, user=rendered.user)

    def _render(self, payload: TInput) -> RenderedPrompt:
        """This agent's system/user pair for one payload."""
        return self._prompts.render_pair(
            self.name,
            user_role=self.user_role_for(payload),
            version=self._prompt_version,
            **self.build_context(payload),
        )

    # ── Hooks ─────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def output_schema(self) -> type[TOutput]:
        """
        The Pydantic model this agent produces.

        For an agent going through `run()` it is also what the provider is
        constrained to decode. For one going through `run_completion()` it is
        only the shape the agent parses its way to — nothing is sent.
        """

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
