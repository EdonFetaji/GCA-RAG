# Graph-Consistency-Aware RAG Research Implementation

Building on CoKG (Lim et al., 2025): A learned GNN-based consistency validator inside a closed-loop RAG summarization pipeline.

## Project Structure

```
graph_rag_research/
├── data/                      # Generated data, graphs, results
├── docs/adr/                  # Architecture decision records
│   └── 0001-canonical-extraction-path.md
├── extraction/                # ★ Canonical KG extraction — FastAPI service, 5 methods
│   ├── schemas.py             #   Pydantic models + the single EntityType/RelationType ontology
│   ├── service.py             #   ontology / evidence / two-agent / accumulate / full_pipeline
│   └── router.py              #   FastAPI endpoints (POST /extract/*)
├── extractor_agent/           # Lightweight single-document extractor (see ADR 0001)
│   ├── extractor_agent.py     #   run_extractor_pipeline() orchestrator
│   ├── extract_entities.py / extract_relations.py
│   ├── normalize_entities.py / validate_schema.py / build_graph_object.py
│   └── constants.py           #   re-exports the ontology from extraction/schemas.py
├── validator/                 # GNN model, training, inference (Track 3, later)
│   └── corruption.py          #   Track 2.2/2.3 — corruption functions + labeling scheme
├── refinement/                # Loop orchestration, actions (later)
├── generation/                # Summary generation (later)
├── evaluation/                # Metrics, experiments (later)
├── notebooks/                 # Jupyter experiments (later)
├── generate_training_data.py  # Track 2.1 — batch clean-KG extraction
├── generate_corruptions.py    # Track 2.2 — corrupted-variant generation
├── generate_splits.py         # Track 2.4 — cluster-level train/val/test split
├── check_dataset.py           # Track 2.5 — sanity checks + summary report
├── poc_extraction.py          # Legacy POC — superseded by extraction/service.py, kept as reference
├── poc_kggen_extraction.py    # Exploratory — kg-gen library spike, not on the canonical path
├── poc_validator.py           # POC 2: GNN validator concept (its corruption logic now lives in validator/corruption.py)
├── poc_pipeline.py            # POC 3: end-to-end pipeline (still uses the legacy extractor)
└── requirements-core.txt      # Core dependencies
```

**Which entrypoint to use:** `extraction/service.py`'s methods are the
canonical extraction path (see `docs/adr/0001-canonical-extraction-path.md`
for the full reasoning). Use `extractor_agent/` for quick single-document
tests where you don't need evidence tracing or grading. `poc_extraction.py`
and `poc_kggen_extraction.py` are reference/exploratory only — don't build
new work on top of them.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-core.txt

# Configure API keys
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Run the Canonical Extraction Service

> set your CEREBRAS_API_KEY= **(your api key)**

> default model : **gpt-oss-120b** (all scripts — `poc_extraction.py`, `utils/llm_utils.py`, `extraction/service.py`, and `extractor_agent/` — read `CEREBRAS_MODEL` from the environment instead of hardcoding a model name)

`extraction/service.py` is the canonical extraction path (see
`docs/adr/0001-canonical-extraction-path.md`). Run it as a FastAPI service:

```bash
uvicorn extraction.router:app --reload
```

Then hit one of the five methods, e.g.:

```bash
curl -X POST http://localhost:8000/extract/full-pipeline \
  -H "Content-Type: application/json" \
  -d '{"cluster_num": 0}'
```

| Endpoint | Method | What it does |
|---|---|---|
| `POST /extract/ontology` | 1. Ontology-constrained | Extraction limited to a fixed entity/relation type "rulebook" |
| `POST /extract/evidence` | 2. Evidence-traced | Every entity/relation carries a verbatim source quote |
| `POST /extract/two-agent` | 3. Two-agent (finder ↔ grader) | Grader agent scores the KG; Finder refines until it passes or iterations run out |
| `POST /extract/accumulate` | 4. Evidence accumulation | Per-document extraction merged across a cluster, enriching rather than overwriting |
| `POST /extract/full-pipeline` | 5. Full pipeline | Accumulation → grader loop → reasoning-path discovery, all combined |

Or call the same methods directly from Python without the HTTP layer:

```python
from extraction.schemas import ExtractionRequest
from extraction.service import full_pipeline
from utils.dataset_utils import load_single_cluster

documents, _ = load_single_cluster(cluster_idx=0)
result = full_pipeline(documents, ExtractionRequest(cluster_num=0))
```

For a quick single-document extraction without evidence tracing or grading
(e.g. for a one-off test), `extractor_agent/` is available as a lighter
alternative:

```python
from extractor_agent.extractor_agent import run_extractor_pipeline

kg = run_extractor_pipeline(document_text)
```

### 3. Run the Legacy POC Scripts (sanity checks)

These predate `extraction/service.py` and are kept as quick standalone
sanity checks, not as the path to build new work on — see
`docs/adr/0001-canonical-extraction-path.md`.

**POC 1: Extraction** - Proves KG extraction works (legacy — superseded by `extraction/service.py`)
```bash
python poc_extraction.py
```
- Loads one Multi-News cluster
- Extracts entities and relations via LLM
- Converts JSON to NetworkX graph
- Visualizes the result
- Saves graph to `data/extracted_graph.pkl`

**POC 2: Validator** - Proves GNN forward pass works
```bash
python poc_validator.py
```
- Loads extracted graph from POC 1
- Creates corrupted variants (missing entities, contradictions, fragmentation)
- Converts to PyTorch Geometric format
- Passes through simple GNN (random weights, no training)
- Shows that tensor conversion works

**POC 3: End-to-End** - Proves complete pipeline works
```bash
python poc_pipeline.py
```
- Extraction → Validation → Refinement (simulated) → Generation
- Shows the full data flow
- Generates actual summary
- Saves results to `data/pipeline_results.json`

## What Each Script Proves

| Script | What It Tests | Success Criteria |
|-----|---------------|------------------|
| **`extraction/service.py`** (canonical) | Evidence-traced, graded, multi-document KG extraction | Valid `KnowledgeGraph` with evidence spans; grader score above threshold |
| **POC 1: Extraction** (legacy) | LLM prompt engineering, JSON parsing, graph building | Valid NetworkX graph with entities and relations |
| **POC 2: Validator** | Graph corruption, PyG conversion, GNN forward pass | Tensor shapes correct, no runtime errors |
| **POC 3: Pipeline** | Full flow from docs to summary | Generated summary looks reasonable |

## After POCs Work

Once all three POCs run successfully, you're ready to build the full system:

1. **Generate Training Data** (Phase 3 from roadmap)
   - Extract 200-300 clean KGs
   - Apply corruption functions
   - Create train/val/test splits

2. **Train GNN Validator** (Phase 4)
   - Move to GPU environment (Colab/cloud)
   - Train with contrastive loss
   - Save model weights

3. **Implement Real Refinement** (Phase 5)
   - Wire validator scores to actual LLM calls
   - Implement retrieval expansion
   - Implement contradiction resolution

4. **Full Evaluation** (Phase 6)
   - ROUGE scores on 100 clusters
   - Graph diagnostic metrics
   - Noise robustness tests

## Dependencies

Core libraries:
- `cerebras-cloud-sdk` - LLM API access
- `datasets` - HuggingFace Multi-News
- `networkx` - Graph manipulation
- `torch` + `torch-geometric` - GNN implementation
- `rouge-score` - Evaluation metric

See `requirements.txt` for complete list with pinned versions.

## Configuration

Edit `.env` to configure:
- `CEREBRAS_API_KEY`: Your Cerebras key (required — every script in this repo, POCs and `extractor_agent/`, calls Cerebras and only Cerebras)
- `CEREBRAS_MODEL`: Model ID to use everywhere in the repo (default `gpt-oss-120b`, Cerebras's current production-tier model). Cerebras periodically retires/rotates model IDs on its public endpoints — if you hit a `model_not_found` 404, check [Cerebras's model catalog](https://inference-docs.cerebras.ai/models/overview) for the current list and set this accordingly.

The repo previously supported Anthropic/OpenAI/Gemini/Groq as alternate providers; that's been dropped in favor of standardizing on Cerebras everywhere (`utils/llm_utils.py` and `extractor_agent/` were already Cerebras-only, so this just brings `poc_extraction.py` in line with them). If you want multi-provider support back, restore the Anthropic/OpenAI/Gemini/Groq branches removed from `poc_extraction.py`'s `call_llm()`, plus the corresponding branches in `utils/llm_utils.py`.

## Troubleshooting

**"Missing API key"**
- Make sure `.env` file exists and has `CEREBRAS_API_KEY` set

**"Graph validation failed"**
- LLM didn't output valid JSON
- Check `data/raw_extraction_response.json` to see raw output
- Adjust prompt if needed

**PyTorch Geometric install issues**
- Follow official install guide: https://pytorch-geometric.readthedocs.io/
- Match your CUDA version if using GPU
- For CPU: `pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.1+cpu.html`

**"Dataset not found"**
- First run downloads Multi-News from HuggingFace
- Requires ~500MB download
- Subsequent runs use cached version

## Cost Estimates

Per document cluster:
- Extraction: ~$0.01-0.05 (depends on cluster size)
- Generation: ~$0.005-0.02
- Evaluation (G-Eval): ~$0.001-0.01

For 100 test clusters:
- Full experiment: ~$3-7 in API costs
- Validator training: ~$5-20 one-time GPU cost

## Next Steps

After POCs work:
1. Review the interactive roadmap (`roadmap.jsx`)
2. Review the architecture diagram (`architecture.jsx`)
3. ~~Start Phase 3: Generate training data~~ — see "Generating Training Data (Track 2)" below
4. Move to GPU environment for validator training (Track 3)

## Generating Training Data (Track 2)

Four scripts, run in order, produce the labeled clean/corrupted dataset
Track 3's GNN training needs. Each is independently re-runnable — outputs
are skipped/resumed rather than regenerated from scratch.

```bash
# 2.1 — batch clean-KG extraction (needs CEREBRAS_API_KEY)
python generate_training_data.py --num-clusters 5     # smoke test first
python generate_training_data.py --num-clusters 250   # full run

# 2.2 — corrupted variants (offline, no API calls — pure Python on the JSON already saved)
python generate_corruptions.py

# 2.4 — cluster-level train/val/test split (run before or after 2.2, doesn't matter — it only looks at data/training/clean/)
python generate_splits.py

# 2.5 — sanity checks + summary report
python check_dataset.py
```

This produces:
```
data/training/
├── clean/{cluster_idx}.json                                  # 2.1
├── clean/_failures.jsonl                                      # 2.1 — clusters that failed extraction
├── corrupted/{cluster_idx}_{corruption_type}_{severity}.json  # 2.2
├── splits.json                                                 # 2.4
├── summary_report.md                                            # 2.5 — human-readable
└── summary_report.json                                          # 2.5 — machine-readable
```

**Corruption types**: `missing_entities`, `contradictions`, `fragmentation`
(the three `SimpleGNN` was designed around — see `poc_validator.py`), plus
`entity_duplication`, `relation_type_swap`, `orphan_node_injection` (new,
opt-in via `--include-extra-types` on `generate_corruptions.py`). See
`validator/corruption.py`'s `label_for()` docstring for how these six map
onto `SimpleGNN`'s four output heads — short version: the extra three
aren't mapped to a dedicated head yet, they're generated for dataset
completeness ahead of a possible Track 3.1 head expansion.

If `check_dataset.py` reports structural problems (a degenerate 0-entity
graph, a relation pointing at a missing entity id), it exits non-zero —
worth wiring into CI once this repo has any.

## Research Context

This implementation builds on:
- **CoKG** (Lim et al., 2025) - Chain of Knowledge Graph for multi-doc summarization
- **CoD** (Adams et al., 2023) - Chain of Density prompting
- **CoE** (Bao et al., 2024) - Chain of Event prompting

Our contribution: Replace CoKG's static quality check with a learned GNN validator that enables targeted, closed-loop refinement.

## Questions?

Check:
1. The roadmap artifact for phase-by-phase breakdown
2. The architecture artifact for system design
3. Code comments in POC scripts for inline documentation
