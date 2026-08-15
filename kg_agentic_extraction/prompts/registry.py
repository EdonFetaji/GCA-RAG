"""
Prompt management.

Prompts are *data*, not code: they live as Jinja2 templates under
`templates/<agent>/<version>.<role>.j2` and are resolved at runtime. Rewording
a prompt, or A/B-testing a new phrasing, is a file change plus a version bump
in settings — never a Python edit.

Layout
------
    templates/
      extractor/  v1.system.j2  v1.user.j2  v1.repair.j2
      grader/     v1.system.j2  v1.user.j2
      grounder/   v1.system.j2  v1.user.j2

`agent` is the folder, `version` pins the generation, `role` says which slot of
the chat call the text fills. Adding `v2.*.j2` alongside `v1.*.j2` lets both
versions coexist so a run can be reproduced against the prompts it actually used.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

DEFAULT_TEMPLATE_ROOT = Path(__file__).parent / "templates"


class PromptNotFoundError(KeyError):
    """No template on disk for the requested agent/version/role."""

    def __init__(self, agent: str, version: str, role: str, root: Path) -> None:
        super().__init__(f"no template {agent}/{version}.{role}.j2 under {root}")
        self.agent, self.version, self.role = agent, version, role


@dataclass(frozen=True)
class RenderedPrompt:
    """A system/user pair ready to hand to an `LLMClient`."""

    system: str
    user: str


class PromptRegistry:
    """
    Loads and renders versioned prompt templates.

    `StrictUndefined` is deliberate: a template referencing a variable the
    caller forgot to pass raises at render time instead of silently emitting an
    empty string into a prompt, which is the kind of bug that otherwise only
    shows up as mysteriously degraded extraction quality.
    """

    def __init__(
        self,
        *,
        root: Path | None = None,
        default_version: str = "v1",
    ) -> None:
        self._root = root or DEFAULT_TEMPLATE_ROOT
        self._default_version = default_version

    @cached_property
    def _env(self) -> Environment:
        return Environment(
            loader=FileSystemLoader(self._root),
            undefined=StrictUndefined,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    @property
    def root(self) -> Path:
        return self._root

    def render(
        self,
        agent: str,
        role: str,
        *,
        version: str | None = None,
        **context: object,
    ) -> str:
        """Render one template. `role` is 'system', 'user', 'repair', …"""
        resolved = version or self._default_version
        name = f"{agent}/{resolved}.{role}.j2"
        try:
            template = self._env.get_template(name)
        except TemplateNotFound as exc:
            raise PromptNotFoundError(agent, resolved, role, self._root) from exc
        return template.render(**context)

    def render_pair(
        self,
        agent: str,
        *,
        user_role: str = "user",
        version: str | None = None,
        **context: object,
    ) -> RenderedPrompt:
        """
        Render the system+user pair an agent needs for one call.

        `user_role` lets an agent swap which user-side template it uses while
        keeping the same system prompt — that is how the extractor reuses its
        persona for both first-pass extraction ('user') and grader-driven
        repair ('repair').
        """
        return RenderedPrompt(
            system=self.render(agent, "system", version=version, **context),
            user=self.render(agent, user_role, version=version, **context),
        )

    def available(self, agent: str) -> list[str]:
        """Template stems on disk for `agent`, e.g. ['v1.system', 'v1.user']."""
        agent_dir = self._root / agent
        if not agent_dir.is_dir():
            return []
        return sorted(p.name.removesuffix(".j2") for p in agent_dir.glob("*.j2"))
