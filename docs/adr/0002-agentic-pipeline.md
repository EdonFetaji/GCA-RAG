# ADR 0002: Agentic extraction pipeline

## Status
Accepted. **Supersedes ADR 0001.**

## Context
ADR 0001 named `extraction/service.py` the canonical extraction path. That
resolved which of the three implementations to use, and unified the ontology,
but it left the underlying shape unchanged: extraction was a set of synchronous
functions with the finder↔grader loop written as a `for` loop inside
`two_agent_extraction()` / `full_pipeline()`.

That shape has three limits worth naming:

1. **The loop is not inspectable.** There is no way to checkpoint it, resume it,
   pause for human review, or stream intermediate state — the iteration lives on
   a Python stack frame.
2. **Prompts are code.** `utils/prompt_utils.py` builds them with f-strings, so
   changing a prompt is a code change and no two runs can be told apart by which
   prompt version they used.
3. **No grounding step.** Entities are free text. Nothing ties "Meridian
   Robotics" in one cluster to the same organisation in another, which caps what
   downstream graph reasoning can do.

## Decision
Extraction is rebuilt as `kg_agentic_extraction/`, a **LangGraph-orchestrated
three-agent pipeline**:

- **extractor** — documents → evidence-traced `KnowledgeGraph`.
- **grader** — graph + documents → typed `GraderReport`, rendered to Markdown.
- **grounder** — refined graph → entities/relations mapped onto DBpedia, via a
  standalone MCP server in `mcp_servers/dbpedia/`.

Extractor and grader loop until the grader reports no issues or the iteration cap
is reached; the grounder then runs once.

This is a **greenfield package**. It does not import from `extraction/`,
`extractor_agent/`, or `utils/` — reusing the old code would have carried its
shape along with it, which is the thing being replaced.

### Consequences for the old code
- `extraction/`, `poc_extraction.py`, `poc_validator.py`, `poc_pipeline.py`, and
  `poc_kggen_extraction.py` move to `poc/` and are frozen. See `poc/README.md`.
- The ontology moves to `kg_agentic_extraction/models/ontology.py`, which is now
  the repo-wide source of truth. `validator/corruption.py` and
  `extractor_agent/constants.py` import from there.
- `poc/extraction/schemas.py` keeps a frozen copy of the old ontology so nothing
  at the repository root has to import out of `poc/`.
- `generate_training_data.py` still calls `poc.extraction.service.full_pipeline`.
  This is deliberate: the 13 clean KGs already in `data/training/clean/` came from
  that path, and repointing it mid-dataset would make later clusters inconsistent
  with earlier ones. Track 2 gets rewritten against the new pipeline as a separate
  task that regenerates from scratch.

## Decisions worth recording

**The grader's contract is a typed model, not Markdown.** The requirement was a
Markdown issue report, and that is what gets produced — but by rendering
`GraderReport` through `prompts/renderers.py`, not by asking the model for
Markdown. Convergence is `not report.issues`, evaluated on a Pydantic object. If
Markdown were the contract, the loop's termination condition would depend on
whether an LLM happened to format a heading the expected way.

**Structured output replaces JSON repair.** `poc/extraction/service.py` has a
`call_llm_json()` that retries with a repair prompt when parsing fails. The new
pipeline constrains decoding to the Pydantic schema at the provider instead, so
malformed output is prevented rather than repaired. This is why there is no
fence-stripping code anywhere in `kg_agentic_extraction/`.

**Grounding is additive, never destructive.** `GroundedKnowledgeGraph` wraps the
graph and carries mappings alongside it rather than rewriting entity names in
place. A failed or low-confidence mapping degrades to "ungrounded" instead of
corrupting a graph the grader already approved.

**The grounder disambiguates; it does not retrieve.** Candidates are fetched
deterministically through the `GroundingBackend` port, and the LLM only chooses
among them. This keeps the non-deterministic step to one call, makes retrieval
cacheable and testable, and prevents the model from emitting plausible-looking
URIs it reconstructed from memory. Giving it live tool access is a supported
extension — replace `_gather_candidates` — and the output contract is unchanged.

**Agents do not know about LangGraph.** They take dataclass payloads and return
models; `nodes/` adapts them onto graph state. An agent can therefore be tested
with a stub `LLMClient` and no graph, no state, and no network.

**Ports, not SDKs.** Agents depend on `LLMClient` and `GroundingBackend`
Protocols. `graph.py` is the only composition root. Swapping the LLM provider or
the ontology source is a new adapter plus a registry entry.

## Revisit this if
- The grounder needs multi-hop reasoning over DBpedia — pre-fetched candidates
  stop being sufficient and it should become a tool-calling agent.
- A fourth agent joins the loop, which would make the routing policy in
  `nodes/routing.py` a genuine state machine rather than one predicate.
- Track 2 is rewritten against this pipeline, at which point `poc/extraction/`
  loses its last live consumer and can be deleted outright.
