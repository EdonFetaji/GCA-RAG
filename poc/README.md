# `poc/` — frozen reference implementations

Everything in this folder predates `kg_agentic_extraction/` and is kept for
reference only. **Do not build new work here.**

| Module | What it was | Superseded by |
|---|---|---|
| `extraction/` | FastAPI service exposing five hand-rolled extraction methods (ontology-constrained, evidence-traced, two-agent finder↔grader, accumulation, full pipeline). Declared canonical by ADR 0001. | `kg_agentic_extraction/` — the extractor↔grader loop is now a LangGraph graph rather than a `for` loop inside `service.py`. |
| `poc_extraction.py` | Single-prompt LLM extraction → NetworkX → matplotlib. | `kg_agentic_extraction/agents/extractor_agent.py` |
| `poc_validator.py` | `SimpleGNN` forward pass over corrupted graphs, random weights. | Track 3 (`validator/model.py`, not yet written). Its corruption logic already moved to `validator/corruption.py`. |
| `poc_pipeline.py` | End-to-end extract → heuristic score → simulated refinement → summarize. | `kg_agentic_extraction/graph.py` |
| `poc_kggen_extraction.py` | Spike against the `kg-gen` library. Dead end, kept as a comparison point. | — |

## Why these were not deleted

`poc_validator.py` still holds the only written-down version of the GNN
architecture Track 3 is meant to start from, and the `data/prompt_based_kg_extraction_results/`
artifacts were produced by `extraction/service.py` — deleting the code that
generated them would leave those files unexplainable.

## Known coupling

`generate_training_data.py` (Track 2.1) still imports `poc.extraction.service`.
That is deliberate: the 13 clean KGs under `data/training/clean/` were produced
by that code path, and repointing the generator at the new pipeline would make
future clusters inconsistent with the ones already generated. Rewriting Track 2
against `kg_agentic_extraction/` is a separate task that should regenerate the
dataset from scratch.

`poc/extraction/schemas.py` keeps its own copy of the entity/relation ontology so
that nothing at the repository root has to import out of `poc/`. The live
ontology is `kg_agentic_extraction/models/ontology.py`.
