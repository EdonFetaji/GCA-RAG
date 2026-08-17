"""
Provider registry and factory.

Open for extension: a new provider is one `register_provider` call, with no
edit to the `if/elif` that would otherwise grow here. Closed for modification:
`build_llm` never changes.
"""

from __future__ import annotations

from collections.abc import Callable

from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.base import LLMClient, LLMError

ProviderFactory = Callable[[PipelineSettings], LLMClient]

_PROVIDERS: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider under `name`. Later registrations win, so tests can override."""
    _PROVIDERS[name.lower()] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def build_llm(settings: PipelineSettings) -> LLMClient:
    """Construct the client for `settings.llm_provider`."""
    key = settings.llm_provider.lower()
    try:
        factory = _PROVIDERS[key]
    except KeyError:
        raise LLMError(
            f"unknown LLM provider {settings.llm_provider!r}; "
            f"registered: {', '.join(available_providers()) or '<none>'}"
        ) from None
    return factory(settings)


def _build_cerebras(settings: PipelineSettings) -> LLMClient:
    from kg_agentic_extraction.llm.cerebras_client import CerebrasClient

    if not settings.cerebras_api_key:
        raise LLMError("CEREBRAS_API_KEY is not set (see .env.example)")
    return CerebrasClient(
        model=settings.model,
        api_key=settings.cerebras_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def _build_groq(settings: PipelineSettings) -> LLMClient:
    from kg_agentic_extraction.llm.groq_client import GroqClient

    if not settings.groq_api_key:
        raise LLMError("GROQ_API_KEY is not set (see .env.example)")
    return GroqClient(
        model=settings.model,
        api_key=settings.groq_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


def _build_gemini(settings: PipelineSettings) -> LLMClient:
    from kg_agentic_extraction.llm.gemini_client import GeminiClient

    if not settings.gemini_api_key:
        raise LLMError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set (see .env.example)")
    return GeminiClient(
        model=settings.model,
        api_key=settings.gemini_api_key,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


register_provider("cerebras", _build_cerebras)
register_provider("groq", _build_groq)
register_provider("gemini", _build_gemini)
