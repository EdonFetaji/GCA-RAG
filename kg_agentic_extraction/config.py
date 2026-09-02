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
    # Meta documents the key as MODEL_API_KEY — a name generic enough to
    # already mean something else in an environment that predates it, so the
    # unambiguous META_API_KEY is accepted first and wins if both are set.
    meta_api_key: str = Field(
        "", validation_alias=AliasChoices("META_API_KEY", "MODEL_API_KEY")
    )
    nvidia_api_key: str = Field("", validation_alias="NVIDIA_API_KEY")
    mistral_api_key: str = Field("", validation_alias="MISTRAL_API_KEY")

    # ── Meta Model API ────────────────────────────────────────────────
    # Only the `meta` provider reads these; they are inert otherwise.
    meta_base_url: str = Field(
        "https://api.meta.ai/v1",
        description="OpenAI-compatible base of Meta Model API. Change for a proxy or gateway.",
    )
    meta_reasoning_effort: str | None = Field(
        "low",
        description=(
            "How long Muse Spark thinks before answering: minimal | low | medium | high | "
            "xhigh. Reasoning tokens bill as output and come out of `max_tokens`, so the "
            "default is low — schema-constrained extraction wants the budget in the answer. "
            "None omits the parameter entirely, for an endpoint that does not accept it."
        ),
    )
    meta_structured_method: str = Field(
        "json_schema",
        description=(
            "How structured output is requested: json_schema | function_calling. Meta "
            "documents tool calling but not the schema mode, so this is the escape hatch "
            "if the endpoint rejects json_schema. See MetaClient._structured_runnable for "
            "what falling back costs."
        ),
    )

    # ── Mistral La Plateforme ─────────────────────────────────────────
    # Only the `mistral` provider reads these; they are inert otherwise.
    mistral_base_url: str = Field(
        "https://api.mistral.ai/v1",
        description="La Plateforme's chat base. Change for a proxy, gateway, or self-deployment.",
    )
    mistral_timeout_seconds: int = Field(
        600,
        ge=1,
        description=(
            "Read timeout for one Mistral call. ChatMistralAI defaults to 120s, which a "
            "16k-token schema-constrained generation overruns on the free tier — the "
            "failure then arrives as an httpx.ReadTimeout wrapped in "
            "LLMStructuredOutputError, which reads like a schema problem and is not one. "
            "Multiplies with retries: an unresponsive endpoint costs timeout × 3."
        ),
    )
    mistral_structured_method: str = Field(
        "json_schema",
        description=(
            "How structured output is requested: json_schema | function_calling. "
            "json_schema is pinned by default for the reason GroqClient documents; it "
            "decodes strictly, so switch to function_calling only if Mistral starts "
            "rejecting the schema. See MistralClient for what falling back costs."
        ),
    )

    # ── NVIDIA NIM ────────────────────────────────────────────────────
    # Only the `nvidia` provider reads these; they are inert otherwise.
    nvidia_base_url: str = Field(
        "https://integrate.api.nvidia.com/v1",
        description="NVIDIA's hosted NIM catalog. Point at a self-hosted NIM container instead.",
    )
    nvidia_top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description=(
            "Nucleus-sampling cutoff. None omits the parameter, which is the default "
            "because the pipeline runs at temperature 0 — top_p only starts to matter "
            "once you raise it (NVIDIA's Gemma sample pairs temperature 1 with 0.95)."
        ),
    )
    nvidia_enable_thinking: bool = Field(
        False,
        description=(
            "Send chat_template_kwargs={'enable_thinking': True}, which is how Gemma-4-it "
            "and similar NIM models gate reasoning. Off by default for the reason on "
            "`max_tokens` above: reasoning tokens come out of the same completion budget, "
            "and schema-constrained extraction wants that budget in the answer."
        ),
    )

    # ── Loop policy ───────────────────────────────────────────────────
    max_iterations: int = Field(
        6,
        ge=3,
        le=10,
        description="Extractor↔grader rounds before the loop stops unconverged.",
    )

    # ── Prompts ───────────────────────────────────────────────────────
    prompt_version: str = Field(
        "v3",
        description=(
            "Template version the extractor and grader resolve; see prompts/templates/. "
            "v2 dropped the ontology from both: the extractor names its own types and the "
            "grader audits them for aptness and consistency instead of list membership. "
            "v3 keeps v2's rules and adds the downstream task — the graph is merged across "
            "a cluster and summarized without the source text — so salience, canonical "
            "entity names and label precision are judged against the summary they feed. "
            "v1 and v2 are kept on disk, so setting this back reproduces either earlier "
            "regime for comparison."
        ),
    )
    grounder_prompt_version: str = Field(
        "v2",
        description=(
            "Version the grounder resolves, separately from the others. The "
            "grounder became a tool-calling agent in v2 while the extractor and "
            "grader stayed on v1, and a single shared version would have forced "
            "a no-op v2 of both just to move one of them."
        ),
    )

    # ── Grounding ─────────────────────────────────────────────────────
    mcp_url: str = Field(
        "http://127.0.0.1:8931/mcp/",
        description=(
            "Streamable-HTTP endpoint of the DBpedia MCP server. FastMCP mounts "
            "it at /mcp/ — the trailing slash matters."
        ),
    )
    grounding_enabled: bool = Field(
        True, description="Set False to stop after convergence, skipping the grounder."
    )
    grounding_max_tool_rounds: int = Field(
        8,
        ge=1,
        le=30,
        description=(
            "Tool-calling rounds the grounder gets before it must answer. A "
            "round can carry several parallel calls, so this is a budget on "
            "reasoning depth, not on the number of lookups."
        ),
    )
    grounding_tool_timeout_seconds: float = Field(
        30.0, gt=0, description="Per-request timeout on the MCP session."
    )

    # ── Output ────────────────────────────────────────────────────────
    graph_output_dir: Path = Field(
        Path("kg_dataset/data"),
        description="Where the runner writes the extracted graph as HDF5.",
    )

    # ── Google Cloud Storage ──────────────────────────────────────────
    # The local file stays the source of truth; the upload is an extra copy, so
    # an empty bucket name simply disables it rather than being an error.
    gcs_bucket: str = Field(
        "",
        description=(
            "Bucket the saved .h5 is uploaded to after it is written. Empty "
            "disables the upload. Name only — no gs:// scheme, no path."
        ),
    )
    gcs_prefix: str = Field(
        "",
        description=(
            "Object-name prefix inside the bucket, e.g. 'graphs' → "
            "gs://<bucket>/graphs/cluster_0.h5. Leading and trailing slashes are "
            "ignored."
        ),
    )

    # ── Extraction scope ──────────────────────────────────────────────
    # Read by the grounder alone. Since v2 prompts the extractor invents its own
    # types, so these lists are no longer a constraint on extraction — they are
    # the controlled vocabulary the grounder maps onto, and the source of its
    # DBpedia hint tables. `domain_context` is the one field the extractor still
    # sees, and it steers subject matter rather than types.
    ontology: OntologyConfig = Field(default_factory=OntologyConfig)
    max_documents: int = Field(
        10, ge=1, description="Cap on documents fed to the extractor in one cluster."
    )
