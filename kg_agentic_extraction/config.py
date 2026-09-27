"""
Pipeline configuration.

One settings object, populated from the environment (or overridden explicitly
in tests), passed down through the composition root in `graph.py`. Nothing
below the composition root reads `os.environ` — agents receive what they need
as constructor arguments.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from kg_agentic_extraction.models.ontology import OntologyConfig

# `env_file` below covers the declared fields, but not `_collect_worker_keys`,
# which scans `os.environ` for an open-ended group pydantic-settings cannot
# declare. This is what puts `.env` there for it to find. Import-time rather
# than call-time so every entry point gets it — nothing reads settings before
# this module is imported.
load_dotenv()

#: `KG_WORKER_<n>_GEMINI_KEYS` / `KG_WORKER_<n>_GRADER_KEY` — the per-worker key
#: bundles read out of the environment by `_collect_worker_keys`.
_WORKER_GEMINI_RE = re.compile(r"^KG_WORKER_(\d+)_GEMINI_KEYS$", re.IGNORECASE)
_WORKER_GRADER_RE = re.compile(r"^KG_WORKER_(\d+)_GRADER_KEY$", re.IGNORECASE)

#: Which `PipelineSettings` field each provider's factory reads its key from.
#: Mirrors the registry in `llm/factory.py` — a provider added there needs an
#: entry here before it can be a batch worker's grader.
_PROVIDER_KEY_FIELD = {
    "cerebras": "cerebras_api_key",
    "groq": "groq_api_key",
    "gemini": "gemini_api_keys",
    "meta": "meta_api_key",
    "nvidia": "nvidia_api_key",
    "mistral": "mistral_api_key",
}


def split_keys(raw: str) -> list[str]:
    """Split a comma-, space- or newline-separated key list, de-duplicated, order kept."""
    keys: list[str] = []
    for candidate in re.split(r"[\s,]+", raw or ""):
        cleaned = candidate.strip()
        if cleaned and cleaned not in keys:
            keys.append(cleaned)
    return keys


class WorkerKeyBundle(BaseModel):
    """
    The API keys one batch worker process owns exclusively.

    Bundles are the unit of isolation in batch mode: no two workers share a key,
    so one worker burning through its Gemini quota cannot retire a key another
    worker is still using. That is only true because each worker is a separate
    *process* — a shared, in-process rotating client would defeat it.

    Two Gemini keys because the extractor sends the documents and receives a full
    graph on every repair round, which is where the token budget goes; one grader
    key because the grader's call is a single pass per round.

    `grader_key` names no vendor on purpose. Which provider it belongs to is
    decided by `KG_GRADER_PROVIDER`, and `for_worker` below is what routes it to
    that provider's field — so retargeting the grader is an `.env` change, not a
    change here.
    """

    worker_id: int
    gemini_keys: list[str] = Field(default_factory=list)
    grader_key: str = ""

    @property
    def is_complete(self) -> bool:
        return bool(self.gemini_keys) and bool(self.grader_key)

    def missing(self) -> list[str]:
        """The environment variables this bundle still needs, for a startup error."""
        gaps = []
        if not self.gemini_keys:
            gaps.append(f"KG_WORKER_{self.worker_id}_GEMINI_KEYS")
        if not self.grader_key:
            gaps.append(f"KG_WORKER_{self.worker_id}_GRADER_KEY")
        return gaps


def _collect_worker_keys() -> list[WorkerKeyBundle]:
    """
    Every `KG_WORKER_<n>_*` bundle in the environment, ordered by `n`.

    Scanned rather than declared as fixed fields so that adding a fifth worker is
    two more environment variables and no code change. `config.py` is the one
    module allowed to read `os.environ` — see the module docstring — and
    pydantic-settings offers no declarative form for an open-ended indexed group.
    """
    found: dict[int, WorkerKeyBundle] = {}

    def bundle(index: int) -> WorkerKeyBundle:
        return found.setdefault(index, WorkerKeyBundle(worker_id=index))

    for name, value in os.environ.items():
        if match := _WORKER_GEMINI_RE.match(name):
            bundle(int(match.group(1))).gemini_keys = split_keys(value)
        elif match := _WORKER_GRADER_RE.match(name):
            bundle(int(match.group(1))).grader_key = value.strip()

    return [found[index] for index in sorted(found)]


class PipelineSettings(BaseSettings):
    """Everything tunable about a pipeline run. See `.env.example`."""

    model_config = SettingsConfigDict(
        env_prefix="KG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ───────────────────────────────────────────────────────────
    # `llm_provider` / `model` are the *fallback* pair: what an agent uses when
    # its role is not bound to a provider of its own below. A single-cluster
    # `runner.py` run with no role bindings set still behaves exactly as before.
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
    # Extra Gemini keys for rotation during a long batch run. Comma-, space- or
    # newline-separated. The gemini adapter moves to the next key when one hits
    # its per-day free-tier quota and resumes where it stopped; see
    # llm/gemini_client.py. `gemini_key_list()` merges this with the single
    # `gemini_api_key` above, de-duplicated, order preserved.
    gemini_api_keys: str = Field(
        "", validation_alias=AliasChoices("GEMINI_API_KEYS", "GOOGLE_API_KEYS")
    )
    # Meta documents the key as MODEL_API_KEY — a name generic enough to
    # already mean something else in an environment that predates it, so the
    # unambiguous META_API_KEY is accepted first and wins if both are set.
    meta_api_key: str = Field("", validation_alias=AliasChoices("META_API_KEY", "MODEL_API_KEY"))
    nvidia_api_key: str = Field("", validation_alias="NVIDIA_API_KEY")
    mistral_api_key: str = Field("", validation_alias="MISTRAL_API_KEY")

    # ── Per-agent provider binding ────────────────────────────────────
    # The extractor and the grader run on different vendors, because they are
    # different workloads. The extractor sends the documents and receives a whole
    # graph back on every repair round; the grader reads that graph once and
    # answers with a verdict. So the extractor sits on Gemini with two rotating
    # keys per worker, and the grader on a single key from whichever vendor
    # `grader_provider` names.
    #
    # Empty means "use `llm_provider` / `model` above" — that is what keeps a
    # plain `runner.py` invocation working unchanged. `for_role()` below is what
    # turns these into something `llm/factory.py` can build, and the factory
    # itself needed no modification to support any of this.
    extractor_provider: str = Field("", description="Provider for the extractor. '' = default.")
    extractor_model: str = Field("", description="Model id for the extractor. '' = default.")
    grader_provider: str = Field("", description="Provider for the grader. '' = default.")
    grader_model: str = Field("", description="Model id for the grader. '' = default.")

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
    nvidia_timeout_seconds: int = Field(
        300,
        ge=1,
        description=(
            "Read timeout for one NVIDIA call. The package default is 60s, which the "
            "grader overruns every time: it reads the whole graph plus the source "
            "documents before it writes a word. The failure arrives as a "
            "requests.ReadTimeout on a fixed 60s boundary — identical across workers, "
            "which is how you tell a client-side deadline from a slow endpoint. "
            "NVIDIA is also given no SDK retries, so this is the whole budget for a call."
        ),
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
        "v4",
        description=(
            "Template version the extractor resolves; see prompts/templates/extractor/. "
            "v2 dropped the ontology: the extractor names its own types rather than "
            "drawing them from a list. v3 added the downstream task — the graph is merged "
            "across a cluster and summarized without the source text — and judged salience "
            "and label precision against the summary they feed. v4 rebalances v3 for "
            "recall: v3 measured under 1 relation per entity and captured no dates or "
            "figures at all, so v4 lets two verbatim quotes establish one relation (news "
            "prose states most links across sentences), reifies dates, amounts and tallies "
            "as entities since the schema has no attribute field, states a coverage floor, "
            "and closes on both failure modes rather than on 'prefer a smaller graph'. "
            "v1-v3 are kept on disk, so setting this back reproduces any earlier regime "
            "for comparison."
        ),
    )
    grader_prompt_version: str = Field(
        "v6",
        description=(
            "Version the grader resolves, separately from the extractor's, for the "
            "same reason the grounder pins its own. v4 and v5 both ask for "
            "`SimpleSchemaGraderReport` as a constrained JSON object — two flat "
            "fields, `converged` beside the Markdown the model wrote — and the "
            "provider is held to decoding it. v5 makes the grader a coverage "
            "instrument rather than only a precision one: it drafts the summary the "
            "graph would yield and reports what that draft cannot say, may ask for "
            "an entity and its edges together, checks for isolated nodes, missing "
            "dates and figures, and uniform salience counts, and refuses to converge "
            "on a thin graph. v6 is v5 plus the structural validator's flags (`gnn_mode`), "
            "and renders exactly as v5 when nothing is flagged. v1-v3 ask for typed "
            "issues and pair with "
            "`GraderReport`, rendered by `report_to_markdown`. The template and the "
            "call shape in `GraderAgent` are one decision — setting this back to "
            "v1-v3 without also changing the agent asks a prompt for one contract "
            "and reads it as another, and the run fails at the first grade."
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

    # ── Structural validator (GNN) ────────────────────────────────────
    # off: no validate node. advise: flagged elements go to the grader's prompt.
    # veto: advise, plus overruling a converged grader while consistency is low.
    gnn_mode: Literal["off", "advise", "veto"] = Field(
        "off", description="Structural validator in the loop: off | advise | veto."
    )
    gnn_checkpoint_dir: Path = Field(
        Path("data/gnn_checkpoints_gcs"),
        description=(
            "Directory holding best_model.pt and feature_vocab.json from one "
            "`gnn_validator.train` run. Always loaded together: the vocab is part of the model."
        ),
    )
    gnn_top_k: int = Field(
        5, ge=1, le=20, description="Most elements shown to the grader per round."
    )
    gnn_min_score: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="An element is only flagged at or above this suspicion score.",
    )
    gnn_veto_threshold: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description=(
            "veto mode: overrule a converged grader when graph consistency is below this. "
            "Extracted graphs score a median ~0.53; 0.3 vetoes ~17% of them."
        ),
    )
    gnn_max_vetoes: int = Field(
        1, ge=0, le=3, description="veto mode: most extra rounds the validator may force per run."
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

    # ── Batch mode (kg_agentic_extraction.batch) ─────────────────────
    # The inclusive range of Multi-News test-split clusters to extract. Both
    # ends are read from the environment (KG_CLUSTER_START / KG_CLUSTER_END) so a
    # run is fully described by .env; `--range START END` overrides them.
    cluster_start: int | None = Field(
        None, ge=0, description="First cluster index for batch mode (inclusive)."
    )
    cluster_end: int | None = Field(
        None, ge=0, description="Last cluster index for batch mode (inclusive)."
    )
    # Worker *processes*, one per core, each running a single-threaded graph —
    # not threads. What is being isolated is not CPU but API-key state: each
    # worker owns its own `WorkerKeyBundle` and its own LLM clients, so a Gemini
    # key exhausted in one worker is invisible to the other three. Threads in one
    # process cannot give that, because the rotating client's "this key is spent"
    # set would be shared by all of them.
    #
    # The ceiling is the number of key bundles we actually hold, not the number
    # of cores: a fifth worker with no keys of its own would have to borrow, which
    # is the failure this design exists to prevent. Raise it alongside
    # `KG_WORKER_4_*` and friends.
    max_workers: int = Field(
        4, ge=1, le=4, description="Batch worker processes (KG_MAX_WORKERS). One key bundle each."
    )

    #: Per-worker key bundles, scanned from `KG_WORKER_<n>_*`. Populated by the
    #: validator below rather than declared, so worker count is an env concern.
    worker_keys: list[WorkerKeyBundle] = Field(default_factory=list)

    @model_validator(mode="after")
    def _load_worker_keys(self) -> PipelineSettings:
        """Fill `worker_keys` from the environment unless a caller supplied them (tests)."""
        if not self.worker_keys:
            self.worker_keys = _collect_worker_keys()
        return self

    def gemini_key_list(self) -> list[str]:
        """
        Every Gemini API key, `gemini_api_key` first, de-duplicated.

        `gemini_api_keys` may be comma-, space- or newline-separated. This is
        what `llm/factory.py` hands the rotating `GeminiClient`.
        """
        return split_keys(" ".join([self.gemini_api_key, self.gemini_api_keys]))

    # ── Scoping ───────────────────────────────────────────────────────
    # Both of these return a *copy* with a few fields overridden rather than
    # teaching the factory about roles or workers. `llm/factory.py` keeps reading
    # exactly the fields it always read — `llm_provider`, `model`, and each
    # provider's own `*_api_key` — and every provider in its registry keeps
    # working, unmodified, for both an extractor and a grader.

    def for_role(self, role: str) -> PipelineSettings:
        """
        This settings object as the named agent sees it: `"extractor"` or `"grader"`.

        Falls back to `llm_provider` / `model` for any role that is not bound, so
        an unconfigured deployment behaves exactly as it did before roles existed.
        """
        provider = getattr(self, f"{role}_provider", "") or self.llm_provider
        model = getattr(self, f"{role}_model", "") or self.model
        return self.model_copy(update={"llm_provider": provider, "model": model})

    def for_worker(self, worker_id: int) -> PipelineSettings:
        """
        This settings object as batch worker `worker_id` sees it: its keys, nobody else's.

        The bundle *replaces* the process-wide keys rather than adding to them —
        a worker must not be able to fall back onto a key another worker owns, or
        the isolation the process model buys is gone. `worker_keys` is narrowed to
        the one bundle for the same reason: this copy is what gets pickled across
        to the child process, and a sibling's keys have no business travelling
        with it.

        `grader_key` is routed to whichever provider field the grader's factory
        will read, resolved through `_PROVIDER_KEY_FIELD` — which is why the
        bundle names no vendor and why retargeting the grader needs no code here.
        Binding the grader to `gemini` is the degenerate case: its field is the
        same `gemini_api_keys` the extractor uses, so the grader key is merged
        into that list and the two roles share one rotating pool rather than
        holding separate keys. Isolation between *workers* is unaffected.
        """
        bundle = self.worker_bundle(worker_id)
        provider = (self.grader_provider or self.llm_provider).lower()
        try:
            grader_field = _PROVIDER_KEY_FIELD[provider]
        except KeyError:
            raise ValueError(
                f"grader provider {provider!r} has no known API-key field; "
                f"known: {', '.join(sorted(_PROVIDER_KEY_FIELD))}. Fix "
                "KG_GRADER_PROVIDER, or add it to _PROVIDER_KEY_FIELD."
            ) from None

        gemini_keys = list(bundle.gemini_keys)
        if grader_field == "gemini_api_keys":
            gemini_keys = split_keys(" ".join([*gemini_keys, bundle.grader_key]))

        update: dict[str, object] = {
            "gemini_api_key": "",
            "gemini_api_keys": ",".join(gemini_keys),
            "worker_keys": [bundle],
        }
        update.setdefault(grader_field, bundle.grader_key)
        return self.model_copy(update=update)

    def worker_bundle(self, worker_id: int) -> WorkerKeyBundle:
        """The bundle for `worker_id`. Raises `KeyError` if it was never configured."""
        for bundle in self.worker_keys:
            if bundle.worker_id == worker_id:
                return bundle
        raise KeyError(f"no key bundle for worker {worker_id} (set KG_WORKER_{worker_id}_*)")
