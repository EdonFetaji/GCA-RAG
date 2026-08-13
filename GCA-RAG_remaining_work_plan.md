# Remaining Work Plan — GCA-RAG

Organized into 6 tracks, roughly in the order they need to happen (later tracks depend on earlier ones). Each item includes what "done" looks like.

---

## Track 0 — Stabilize what already exists (before building anything new)

These are blocking bugs. Nothing downstream should be built on top of broken code.

**0.1 Fix `extractor_agent/` import structure**
- Change `from constants import VALID_ENTITY_TYPES` → `from extractor_agent.constants import VALID_ENTITY_TYPES` in `extract_entities.py`, `normalize_entities.py`, `validate_schema.py`.
- Add `extractor_agent/__init__.py` so it's an explicit package, not relying on namespace-package behavior.
- Done when: `from extractor_agent.extractor_agent import run_extractor_pipeline` succeeds from the project root.

**0.2 Fix dependency gaps**
- Add `cerebras-cloud-sdk` to `requirements.txt` and `requirements-core.txt`.
- Decide whether `utils/llm_utils.py` should support multiple providers again or standardize on Cerebras — update whichever modules assume the other. If standardizing on Cerebras, update the README's "set LLM_PROVIDER = groq" instructions accordingly; if keeping multi-provider, restore the Anthropic/OpenAI/Gemini/Groq branches that were dropped.
- Done when: a fresh `pip install -r requirements-core.txt` + `.env` with one key lets every script in the repo run without `ModuleNotFoundError`.

**0.3 Fix `poc_extraction.py`'s graph save step**
- Replace `nx.write_gpickle(G, ...)` / any `read_gpickle` with `pickle.dump`/`pickle.load` (or `nx.write_graphml`/`nx.node_link_data` + JSON).
- Done when: `python poc_extraction.py` runs to completion and `data/extracted_graph.pkl` is actually created.

**0.4 Harden `extract_relations.py`**
- Reuse the fence-stripping/repair-retry logic from `extract_entities.py`'s `safe_parse_json` (or `extraction/service.py`'s `call_llm_json`) instead of a bare `json.loads`.
- Done when: relation extraction survives a response wrapped in ` ```json ` fences or with leading commentary.

**0.5 Run `extractor_agent/` end-to-end at least once**
- Get a Cerebras key, run `run_extractor_pipeline()` on one Multi-News document, save output to `data/extractor_agent_sample.json`.
- Done when: there's a real output artifact proving the module works, same as the other extraction paths already have.

---

## Track 1 — Unify the extraction layer

Right now there are **three parallel, divergent extraction implementations** in the repo (flat POC script, `extraction/` FastAPI service with 5 methods, `extractor_agent/` package) with two different entity ontologies. This needs to converge before Phase 3 (which needs one canonical extraction path to generate training data from).

**1.1 Decide the canonical extraction path**
- Likely candidate: `extraction/service.py`'s methods (most mature — evidence-tracing, grader loop, accumulation) as the "production" path, with `extractor_agent/`'s cleaner modular structure (separate entity/relation/normalize/validate steps) folded into it, OR `extractor_agent/` promoted to be the new backbone and `extraction/` methods ported into its structure.
- Output: a short ADR (architecture decision doc, even 1 paragraph) stating which wins and why, so this doesn't get re-litigated later.

**1.2 Reconcile the two ontologies**
- Merge `extractor_agent/constants.py` (`PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, OTHER`) with `extraction/schemas.py`'s `EntityType`/`RelationType` enums (10 types, domain-flavored with `DISEASE/TREATMENT/SYMPTOM/METRIC`) into a single ontology config. Since the project is targeting Multi-News (general news), the domain-specific medical types are probably vestigial — decide whether to keep them or trim back to a general-news set with `PRODUCT`/`OTHER` added in.
- Done when: one `OntologyConfig`-equivalent is imported everywhere extraction happens.

**1.3 Remove or clearly mark dead paths**
- If `poc_extraction.py`'s inline prompt-building is superseded by `extraction/service.py`, either delete it or explicitly relabel it "legacy/reference only" so it doesn't get confused with the real path.
- Same for `poc_kggen_extraction.py` (the kg-gen library exploration) — decide if that's a dead end or a real alternative worth keeping in the running.

**1.4 Update the README**
- It currently doesn't mention `extraction/`'s 5 methods or `extractor_agent/` at all — still describes only the 3 original POCs. Rewrite the "Project Structure" and "Quick Start" sections to reflect what's actually there and which entrypoint to use.

---

## Track 2 — Phase 3: Training data generation

This is the first real new-code phase. Needs Track 1 done first (need one stable extractor to generate clean KGs from).

**2.1 Batch clean-KG extraction**
- Script (`generate_training_data.py` or similar) that loops over N Multi-News clusters (target: 200–300 per the roadmap) and runs the canonical extractor on each, saving `data/training/clean/{cluster_idx}.json`.
- Handle failures gracefully — log and skip clusters where extraction/parsing fails after retries, rather than crashing the batch.
- Rate-limit / backoff for API calls; this will be hundreds of LLM calls, so add checkpointing (resume from last successful index) so a crash at cluster 150 doesn't mean starting over.

**2.2 Systematic corruption generation**
- Take the three corruption functions already prototyped in `poc_validator.py` (`missing_entities`, `contradictions`, `fragmentation`) and generalize them into a reusable module (`validator/corruption.py`).
- For each clean KG, generate multiple corrupted variants per corruption type with varying severity (e.g., remove 10%/20%/30% of nodes) rather than one fixed variant — gives the GNN more signal about degree of corruption, not just binary clean/dirty.
- Consider adding corruption types not yet covered: entity duplication (same entity as two nodes), relation-type swapping (correct entities, wrong relation label), and orphan-node injection (unsupported entity added).
- Save as `data/training/corrupted/{cluster_idx}_{corruption_type}_{severity}.json`, with the corruption type/severity as the label.

**2.3 Labeling scheme**
- Decide the exact target the GNN will be trained on: binary per-corruption-type classification (4 sigmoid outputs, matching the existing `SimpleGNN` head), or a single scalar consistency score with type breakdown as auxiliary. Given `SimpleGNN` already has 4 output heads, staying with that multi-label design keeps continuity — but confirm the corruption-generation labels line up 1:1 with those 4 heads (currently: consistency, missing_entities, contradictions, fragmentation).

**2.4 Train/val/test split**
- Split at the *cluster* level (not the corrupted-variant level) to avoid leakage — e.g., 70/15/15 cluster split, then generate corruptions within each split.
- Save split indices to `data/training/splits.json` for reproducibility.

**2.5 Dataset sanity checks**
- Class balance check (are corruption types represented evenly?).
- Spot-check a handful of corrupted graphs by eye/visualization to confirm corruptions are realistic and not degenerate (e.g., not reducing every graph to 0 nodes).

Deliverable: `data/training/` populated with clean + corrupted graphs, a splits file, and a short summary report (counts per split, per corruption type).

---

## Track 3 — Phase 4: Train the GNN validator

Depends on Track 2. Needs a GPU environment (Colab/cloud, per the roadmap).

**3.1 Finalize model architecture**
- Start from `SimpleGNN` in `poc_validator.py` (2-layer GCN + global mean pool + 4 linear heads) as the baseline — decide if it needs more capacity (more layers, attention-based conv like GAT) once you see how it performs, but don't over-engineer before a baseline exists.
- Move it into `validator/model.py`.

**3.2 Build the training loop**
- `validator/train.py`: PyG `DataLoader` over the batched clean+corrupted graphs, loss function — likely BCE loss per head (multi-label) matching the 4 sigmoid outputs, or a contrastive loss (clean vs. corrupted embeddings pushed apart) as the roadmap mentions. Decide which — contrastive loss requires a different training setup (pairs/triplets) than straightforward multi-label BCE; pick one and document why.
- Add standard training infra: train/val loop, early stopping on val loss, checkpointing best model.

**3.3 Feature engineering**
- Current node features (`build_node_features` in `poc_validator.py`) are minimal: one-hot type (5 dims), doc frequency, confidence, degree. Consider whether this is enough signal — may want to add embedding-based features (e.g., a small sentence-transformer embedding of the entity name) for the model to learn semantic patterns, not just structural ones.

**3.4 Train and evaluate**
- Run training, track metrics per head (accuracy/F1 per corruption type on held-out val set).
- Save final weights to `validator/checkpoints/`.
- Write a short results doc: does the trained model actually separate clean from corrupted graphs better than the heuristic `score_kg()` currently used in `poc_pipeline.py`? This comparison is the whole point of the research contribution — make it explicit.

**3.5 Wire the trained model back into inference**
- Replace the random-weight `SimpleGNN` in `poc_validator.py` / the heuristic `score_kg()` in `poc_pipeline.py` with a checkpoint-loading inference function (`validator/infer.py`) that both the pipeline and future refinement loop call.

Deliverable: trained model checkpoint + eval report showing it distinguishes clean/corrupted graphs, and the pipeline now calls the real model instead of heuristics.

---

## Track 4 — Phase 5: Real refinement

Depends on Track 3 (need real validator scores to act on, not heuristics).

**4.1 Design the refinement action space**
- `simulate_refinement()` in `poc_pipeline.py` already sketches the mapping: high `missing_entities` → expand retrieval; high `contradictions` → retrieve corroborating sources; high `fragmentation` → rerank/regroup documents. Turn each of these into an actual implemented action rather than a print statement.

**4.2 Implement each action**
- **Retrieval expansion**: given entities the validator flags as under-supported, issue targeted re-extraction prompts focused on those entities against the same document set (or expand to more documents in the cluster if only a subset was used).
- **Contradiction resolution**: for edges/entities flagged, prompt the LLM with the conflicting evidence spans and ask it to reconcile or flag as genuinely contradictory (don't just silently pick one).
- **Fragmentation repair**: re-run extraction with a prompt nudged toward connecting the disconnected components, or explicitly ask the LLM to find relations between components based on document co-occurrence.
- Each action should take the current KG + validator scores + original documents and return a revised KG — not print a description.

**4.3 Loop orchestration**
- `refinement/loop.py`: given a KG, run validator → if below threshold, pick actions based on which sub-scores are worst → apply → re-validate → repeat up to `max_iterations` (matching the existing `poc_pipeline.py` structure, just with real actions/scores instead of simulated ones).
- Add loop safety: cap iterations, detect non-improvement (if score doesn't improve after an iteration, don't loop forever — either try a different action or stop and flag as "could not refine").

**4.4 Cost/latency tracking**
- Since each refinement iteration is more LLM calls, log token usage / call count per pipeline run so cost estimates in the README can be updated with real numbers instead of guesses.

Deliverable: `refinement/` module where each action genuinely modifies the KG, and a full pipeline run shows the score actually improving iteration over iteration (or a documented case where it doesn't and the loop correctly bails out).

---

## Track 5 — Phase 6: Evaluation

Depends on Tracks 3 and 4 both existing, since this evaluates the whole closed loop against baselines.

**5.1 Build the evaluation harness**
- `evaluation/run_eval.py`: runs the full pipeline (extract → validate → refine → generate) over a fixed test set (~100 clusters per roadmap, held out from Track 2's training split), saving generated summaries + all intermediate KGs/scores.

**5.2 ROUGE scoring**
- Compute ROUGE-1/2/L of generated summaries against Multi-News reference summaries.
- Report against **baselines**, not just the full system in isolation — this is what actually proves the research contribution:
  - No-KG baseline (plain LLM summarization of the raw documents, no graph at all)
  - KG-with-no-validation baseline (extract → generate, skip validator/refinement entirely)
  - Full closed-loop system (extract → validate → refine → generate)
- Without these comparisons, there's no way to show the validator+refinement loop is actually earning its complexity.

**5.3 Graph diagnostic metrics**
- Track structural quality across the eval set: entity/relation count distributions, average validator consistency score before vs. after refinement, fraction of clusters that needed refinement, fraction that improved vs. didn't.

**5.4 Noise robustness tests**
- Deliberately inject corruption (using the Track 2 corruption functions) into extracted KGs before running them through the refinement loop, and measure how well the system recovers vs. the no-refinement baseline. This directly tests the core hypothesis of the project.

**5.5 Write up results**
- A results doc/notebook summarizing all of the above — this is the actual "proof" deliverable for the research project, distinct from all the engineering plumbing before it.

Deliverable: `evaluation/` module, a results report with ROUGE + graph metrics + robustness numbers across baselines, saved to `data/eval_results/`.

---

## Track 6 — Cleanup / polish (can happen in parallel with anything above)

- Replace `main.py` (still the PyCharm placeholder) with a real CLI entrypoint that ties extraction → validation → refinement → generation together as one command.
- Add `.env.example` with all required keys (currently missing entirely — README references `cp .env.example .env` but no such file exists in the repo).
- Add basic tests (`pytest`) at least for the JSON-parsing/repair logic and the corruption functions — these are the pieces most likely to silently break on edge cases (empty graphs, single-node graphs, malformed LLM output).
- Restore or add `roadmap.jsx`/`architecture.jsx` referenced in the README, or remove the references if they're not going to exist.

---

## Suggested sequencing summary

1. **Track 0** (bug fixes) — do first, small effort, unblocks everything.
2. **Track 1** (unify extraction) — do before Track 2, or Track 2 will generate training data from an extractor you later throw away.
3. **Track 2** (training data) — the actual start of "new" research work.
4. **Track 3** (train validator) — needs GPU access; can be parallelized with early Track 4 design work.
5. **Track 4** (real refinement) — needs Track 3's real scores to be meaningful.
6. **Track 5** (evaluation) — last, since it evaluates everything built above.
7. **Track 6** — ongoing, low-priority, do opportunistically.
