"""
Pipeline configuration.

One settings object, populated from the environment (or overridden explicitly
in tests), passed down through the composition root in `graph.py`. Nothing
below the composition root reads `os.environ` — agents receive what they need
as constructor arguments.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from kg_agentic_extraction.models.ontology import OntologyConfig


class PipelineSettings(BaseSettings):
    """Everything tunable about a pipeline run. See `.env.example`."""

    model_config = SettingsConfigDict(
        env_prefix="KG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────
    llm_provider: str = Field("groq", description="Key registered in llm/factory.py.")
    model: str = Field("llama-3.3-70b-versatile", description="Provider-specific model id.")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(
        16384,
        description=(
            "Completion-token cap. MUST be set explicitly: Groq defaults to 3072, and a "
            "reasoning model (openai/gpt-oss-*) spends nearly all of that on reasoning "
            "tokens, leaving nothing for the answer — the call then fails as "
            "json_validate_failed with an empty generation. None = provider default."
        ),
    )
    # Provider keys are read from their un-prefixed, vendor-documented names
    # rather than KG_-prefixed ones, so an existing key already exported in the
    # environment works without being renamed.
    cerebras_api_key: str = Field("", validation_alias="CEREBRAS_API_KEY")
    groq_api_key: str = Field("", validation_alias="GROQ_API_KEY")
    # Google AI Studio issues the key as GEMINI_API_KEY, while the LangChain
    # integration conventionally reads GOOGLE_API_KEY. Both are accepted so
    # whichever name is already in the environment or .env just works;
    # GEMINI_API_KEY wins if both are set.
    gemini_api_key: str = Field(
        "", validation_alias=AliasChoices("GEMINI_API_KEY", "GOOGLE_API_KEY")
    )

    # ── Loop policy ───────────────────────────────────────────────────
    max_iterations: int = Field(
        5,
        ge=1,
        le=10,
        description="Extractor↔grader rounds before the loop stops unconverged.",
    )

    # ── Prompts ───────────────────────────────────────────────────────
    prompt_version: str = Field(
        "v1", description="Template version each agent resolves; see prompts/templates/."
    )

    # ── Grounding ─────────────────────────────────────────────────────
    mcp_url: str = Field(
        "http://localhost:8931/mcp",
        description="Streamable-HTTP endpoint of the DBpedia MCP server.",
    )
    grounding_enabled: bool = Field(
        True, description="Set False to stop after convergence, skipping the grounder."
    )

    # ── Output ────────────────────────────────────────────────────────
    graph_output_dir: Path = Field(
        Path("kg_dataset/data"),
        description="Where the runner writes the extracted graph as HDF5.",
    )

    # ── Extraction scope ──────────────────────────────────────────────
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    max_documents: int = Field(
        10, ge=1, description="Cap on documents fed to the extractor in one cluster."
    )
