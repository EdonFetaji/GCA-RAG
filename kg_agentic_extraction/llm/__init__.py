"""
llm — the model port, its adapters, and the provider factory.

Registered providers: `cerebras`, `groq`, `gemini`. Select with
`KG_LLM_PROVIDER`; each reads its own vendor-named key (`CEREBRAS_API_KEY`,
`GROQ_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`).
"""

from kg_agentic_extraction.llm.base import (
    LLMClient,
    LLMError,
    LLMStructuredOutputError,
)
from kg_agentic_extraction.llm.factory import (
    available_providers,
    build_llm,
    register_provider,
)

__all__ = [
    "LLMClient",
    "LLMError",
    "LLMStructuredOutputError",
    "available_providers",
    "build_llm",
    "register_provider",
]
