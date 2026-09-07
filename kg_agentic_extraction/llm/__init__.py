"""
llm — the model port, its adapters, and the provider factory.

Registered providers: `cerebras`, `groq`, `gemini`, `meta`, `nvidia`,
`mistral`. Select with `KG_LLM_PROVIDER`; each reads its own vendor-named key
(`CEREBRAS_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`,
`META_API_KEY` / `MODEL_API_KEY`, `NVIDIA_API_KEY`, `MISTRAL_API_KEY`).
"""

from kg_agentic_extraction.llm.base import (
    LLMClient,
    LLMCompletionError,
    LLMError,
    LLMStructuredOutputError,
    TextLLMClient,
)
from kg_agentic_extraction.llm.factory import (
    available_providers,
    build_llm,
    register_provider,
)

__all__ = [
    "LLMClient",
    "LLMCompletionError",
    "LLMError",
    "LLMStructuredOutputError",
    "TextLLMClient",
    "available_providers",
    "build_llm",
    "register_provider",
]
