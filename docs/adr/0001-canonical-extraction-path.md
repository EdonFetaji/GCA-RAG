# ADR 0001: Canonical extraction path

## Status
Accepted

## Context
The repo had three parallel, divergent extraction implementations:

1. **`poc_extraction.py`** — flat POC script, single inline prompt, one LLM
   call, no evidence tracing, no grading, hardcoded provider/model logic.
2. **`extraction/` (FastAPI service)** — `extraction/service.py` implements
   five methods (ontology-constrained, evidence-traced, two-agent
   finder/grader self-correction, multi-document evidence accumulation, and
   a full pipeline combining all four), backed by Pydantic schemas
   (`extraction/schemas.py`) and exposed via `extraction/router.py`.
3. **`extractor_agent/`** — a cleaner modular package with separate,
   single-purpose steps (`extract_entities` → `validate_schema` →
   `normalize_entities` → `extract_relations` → `build_graph_object`),
   orchestrated by `extractor_agent.run_extractor_pipeline`.

These also used two different entity ontologies (`extractor_agent`'s
6-type `{PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, OTHER}` vs.
`extraction/schemas.py`'s 10-type, partly medical-domain-flavored enum).
Track 2 (training-data generation) needs one stable extractor to generate
clean KGs from, so this had to converge before any new pipeline code gets
built on top of it.

## Decision
**`extraction/service.py`'s methods are the canonical/production
extraction path going forward** — specifically `full_pipeline` (or
`two_agent_extraction` for a cheaper run without reasoning-path discovery)
for Track 2's batch KG generation.

Reasoning:
- It's the most mature of the three: evidence spans tying every entity/
  relation back to a verbatim source quote, a grader agent that scores and
  flags hallucinations/missing facts/contradictions, a refinement loop that
  acts on that feedback, multi-document evidence accumulation, and
  multi-hop reasoning-path discovery. Track 2's corruption-generation and
  Track 3's GNN training both benefit from evidence-traced, graded KGs
  more than from a faster-but-unaudited extraction.
- It already operates on document *clusters* (`list[str]`), matching how
  Multi-News data actually looks, whereas `extractor_agent/` takes a single
  document string.
- It's schema-validated end-to-end via Pydantic (`extraction/schemas.py`),
  which downstream training/corruption code can rely on directly.

`extractor_agent/`'s genuinely useful piece — clean separation of
normalize/validate as standalone steps, plus its entity-name normalization
(accent stripping, casing, longest-surface-form dedup in
`normalize_entities.py`) — is not currently duplicated in
`extraction/service.py`'s `_parse_kg`. That normalization step is worth
folding into the `extraction/` pipeline in a later pass (not done as part
of this ADR); until then, `extractor_agent/` remains available as a
lighter-weight, single-document extractor for quick tests, but new
pipeline work (Track 2+) should be built on `extraction/service.py`.

`poc_extraction.py` and `poc_kggen_extraction.py` are relabeled
legacy/exploratory respectively (see their module docstrings) rather than
deleted, since they're small, self-contained, and still useful as
reference/comparison points.

## Ontology
The two ontologies are reconciled into one: `extraction/schemas.py`'s
`EntityType`/`RelationType` enums are now the single source of truth.
The medical-domain types (`DISEASE`, `TREATMENT`, `SYMPTOM`, `METRIC`,
and the paired relation types `TREATS`, `MEASURED_BY`) were vestigial for
a project targeting Multi-News (general news, not a medical corpus) and
have been dropped; `PRODUCT` and `OTHER` were folded in from
`extractor_agent/constants.py`'s set. `extractor_agent/constants.py` now
re-exports `VALID_ENTITY_TYPES` from `extraction.schemas.EntityType`
instead of maintaining its own separate list.

## Consequences
- Track 2 (batch training-data generation) should call into
  `extraction/service.py` (`full_pipeline` or `two_agent_extraction`), not
  `extractor_agent/` or either POC script.
- Anything that assumed the old `extraction/schemas.py` medical entity
  types (`DISEASE`/`TREATMENT`/`SYMPTOM`/`METRIC`) or relation types
  (`TREATS`/`MEASURED_BY`) needs updating — a repo-wide grep found no
  other usages at the time of this ADR, so no other files needed changes.
- If `extractor_agent/`'s normalization step is later folded into
  `extraction/service.py`, this ADR should be revisited/superseded.
