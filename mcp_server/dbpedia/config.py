"""
Server-side configuration.

Deliberately separate from `kg_agentic_extraction.config.PipelineSettings`: these
are the server's own endpoints and limits, and the server is a standalone
service that must be startable without the pipeline package being importable.
Hence the `DBPEDIA_` env names rather than the pipeline's `KG_` prefix.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBpediaSettings(BaseSettings):
    """Endpoints and limits for the three DBpedia services this server fronts."""

    model_config = SettingsConfigDict(
        env_prefix="DBPEDIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Endpoints ─────────────────────────────────────────────────────
    sparql_endpoint: str = Field(
        "https://dbpedia.org/sparql",
        description="SPARQL query endpoint. Queried over GET.",
    )
    lookup_endpoint: str = Field(
        "https://lookup.dbpedia.org/api/search",
        description="DBpedia Lookup — keyword search over resource labels.",
    )
    spotlight_endpoint: str = Field(
        "https://api.dbpedia-spotlight.org/en",
        description=(
            "DBpedia Spotlight base URL, without the trailing /candidates or "
            "/annotate. The public instance is frequently unavailable; point "
            "this at a local container to make spotlight_link reliable."
        ),
    )

    # ── HTTP behaviour ────────────────────────────────────────────────
    http_timeout: float = Field(30.0, gt=0, description="Per-request timeout, seconds.")
    max_retries: int = Field(
        2, ge=0, description="Retries on 429/5xx/timeout, with exponential backoff."
    )

    # ── Result shaping ────────────────────────────────────────────────
    max_results: int = Field(10, ge=1, le=50, description="Default cap on rows per tool call.")
    abstract_chars: int = Field(
        600, ge=100, description="dbo:abstract is truncated to this many characters."
    )

    # ── Caching ───────────────────────────────────────────────────────
    # The public endpoints rate-limit, and one grounding pass issues many
    # near-identical queries (the same class lookups for every entity of a
    # type). Caching is what keeps a run from being throttled halfway through.
    cache_ttl_seconds: float = Field(900.0, ge=0, description="0 disables caching.")
    cache_max_entries: int = Field(2048, ge=1)


@lru_cache(maxsize=1)
def get_settings() -> DBpediaSettings:
    """The process-wide settings instance. Cached so env is read once."""
    return DBpediaSettings()
