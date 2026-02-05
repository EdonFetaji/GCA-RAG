# Graph-Consistency-Aware RAG Research Implementation

Building on CoKG (Lim et al., 2025): A learned GNN-based consistency validator inside a closed-loop RAG summarization pipeline.

## Project Structure

```
graph_rag_research/
├── data/                      # Generated data, graphs, results
├── extraction/                # KG extraction logic (later)
├── validator/                 # GNN model, training, inference (later)
├── refinement/                # Loop orchestration, actions (later)
├── generation/                # Summary generation (later)
├── evaluation/                # Metrics, experiments (later)
├── notebooks/                 # Jupyter experiments (later)
├── poc_extraction.py          # POC 1: Basic KG extraction
├── poc_validator.py           # POC 2: GNN validator concept
├── poc_pipeline.py            # POC 3: End-to-end pipeline
└── requirements-core.txt      # Core dependencies
```

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

### 2. Run POC Scripts (In Order)

> set LLM_PROVIDER = groq

> set your GROQ_API_KEY= **(your api key)** 

> best model for kg extraction : **llama-3.3-70b-versatile ** 

**POC 1: Extraction** - Proves KG extraction works
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

## What Each POC Proves

| POC | What It Tests | Success Criteria |
|-----|---------------|------------------|
| **1: Extraction** | LLM prompt engineering, JSON parsing, graph building | Valid NetworkX graph with entities and relations |
| **2: Validator** | Graph corruption, PyG conversion, GNN forward pass | Tensor shapes correct, no runtime errors |
| **3: Pipeline** | Full flow from docs to summary | Generated summary looks reasonable |

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
- `anthropic` or `openai` - LLM API access
- `datasets` - HuggingFace Multi-News
- `networkx` - Graph manipulation
- `torch` + `torch-geometric` - GNN implementation
- `rouge-score` - Evaluation metric

See `requirements.txt` for complete list with pinned versions.

## Configuration

Edit `.env` to configure:
- `LLM_PROVIDER`: "anthropic", "openai", "gemini", or "groq"
- `ANTHROPIC_API_KEY`: Your Anthropic key
- `OPENAI_API_KEY`: Your OpenAI key
- `GEMINI_API_KEY`: Your Google AI Studio key
- `GROQ_API_KEY`: Your Groq key
- Model selection: `ANTHROPIC_MODEL`, `OPENAI_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`

## Troubleshooting

**"Missing API key"**
- Make sure `.env` file exists and has your key
- Check the key starts with `sk-ant-` (Anthropic) or `sk-` (OpenAI)

**"Graph validation failed"**
- LLM didn't output valid JSON
- Check `data/raw_extraction_response.json` to see raw output
- Adjust prompt if needed
- For Gemini, ensure `GEMINI_MODEL` is valid for `generate_content` (see ListModels)

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
3. Start Phase 3: Generate training data
4. Move to GPU environment for validator training

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
