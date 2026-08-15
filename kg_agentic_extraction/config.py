"""
Pipeline configuration.

One settings object, populated from the environment (or overridden explicitly
in tests), passed down through the composition root in `graph.py`. Nothing
below the composition root reads `os.environ` — agents receive what they need
as constructor arguments.
"""

from __future__ import annotations

from pydantic import Field
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
    llm_provider: str = Field("cerebras", description="Key registered in llm/factory.py.")
    model: str = Field("gpt-oss-120b", description="Provider-specific model id.")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    cerebras_api_key: str = Field(
        "",
        # Read from the un-prefixed CEREBRAS_API_KEY, which is what the provider
        # documents and what the rest of the repo already uses.
        validation_alias="CEREBRAS_API_KEY",
    )

    # ── Loop policy ───────────────────────────────────────────────────
    max_iterations: int = Field(
        3,
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

    # ── Extraction scope ──────────────────────────────────────────────
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    max_documents: int = Field(
        10, ge=1, description="Cap on documents fed to the extractor in one cluster."
    )
