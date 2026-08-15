# Graph-Consistency-Aware RAG

Building on CoKG (Lim et al., 2025): a learned GNN-based consistency validator
inside a closed-loop RAG summarization pipeline over Multi-News.

**Contribution:** replace CoKG's static quality check with a learned GNN
validator that enables targeted, closed-loop refinement.

## Structure

```
GCA-RAG/
├── kg_agentic_extraction/   ★ the extraction pipeline (LangGraph, 3 agents)
├── mcp_servers/dbpedia/     ★ standalone MCP service backing the grounder
├── validator/               Track 3 — GNN validator (corruption.py done, model/train pending)
├── extractor_agent/         standalone modular extractor, unchanged
├── poc/                     frozen pre-LangGraph generation — see poc/README.md
├── utils/                   Multi-News loading + legacy LLM/prompt helpers
├── docs/adr/                architecture decisions (0002 supersedes 0001)
├── data/training/           Track 2 dataset — clean + corrupted KGs, splits
└── generate_*.py            Track 2 dataset scripts
```

The pipeline itself:

```
kg_agentic_extraction/
├── graph.py         ★ the LangGraph graph object — composition + topology only
├── runner.py          public entrypoint / CLI
├── state.py           the graph's channel schema
├── config.py          PipelineSettings (env-driven)
├── models/            Pydantic contracts: ontology, knowledge_graph, grading, grounding
├── prompts/           ★ versioned Jinja2 templates + model→text renderers
├── llm/               LLMClient port, Cerebras adapter, provider factory
├── grounding/         GroundingBackend port, MCP adapter
├── agents/            extractor_agent.py, grader_agent.py, grounder_agent.py
└── nodes/             adapters binding agents to graph state + loop routing
```

Layers may import downward only: `models → prompts/llm/grounding → agents →
nodes → graph → runner`.

## The pipeline

```
START → extract → grade ─┬─ refine ──→ extract     (loop)
                         ├─ ground ──→ END
                         └─ end ─────→ END
```

| Agent | In | Out |
|---|---|---|
| **extractor** | documents (+ prior graph & grader report on repair rounds) | evidence-traced `KnowledgeGraph` |
| **grader** | graph + documents | `GraderReport`, rendered to Markdown |
| **grounder** | refined graph | entities/relations mapped to DBpedia |

Extractor and grader loop until the grader reports **no issues** or
`KG_MAX_ITERATIONS` is reached; the grounder then runs once.

Convergence is evaluated on the typed report (`not report.issues`), never by
parsing the Markdown — see [ADR 0002](docs/adr/0002-agentic-pipeline.md).

## Setup

Dependencies are managed with **uv**. One `uv.lock` covers the whole project.

```bash
uv sync                     # runtime
uv sync --all-groups        # + dev tools (pytest, ruff)

cp .env.example .env        # then set CEREBRAS_API_KEY
```

## Running it

```bash
# 1. start the DBpedia MCP server (needed only if grounding is on)
uv run python -m mcp_servers.dbpedia.server

# 2. run the pipeline
uv run python -m kg_agentic_extraction.runner --cluster 0 --report report.md
uv run python -m kg_agentic_extraction.runner --file docs.txt --no-grounding
uv run python -m kg_agentic_extraction.runner --draw          # print the topology
```

From Python:

```python
from kg_agentic_extraction import PipelineSettings, build_graph, run_pipeline
from utils.dataset_utils import load_single_cluster

documents, _ = load_single_cluster(cluster_idx=0)
result = run_pipeline(documents, settings=PipelineSettings())

print(result.converged, result.iterations)
print(result.grader_markdown)
print(result.grounded_graph.coverage)
```

Reuse one compiled graph across many clusters:

```python
app = build_graph(settings=settings)
for i in range(100):
    docs, _ = load_single_cluster(i)
    run_pipeline(docs, settings=settings, graph=app)
```

## Extending it

| To… | Do this |
|---|---|
| reword a prompt | edit `kg_agentic_extraction/prompts/templates/<agent>/v1.*.j2` |
| A/B a prompt | add `v2.*.j2` beside it, set `KG_PROMPT_VERSION=v2` |
| change the ontology | add to the enums in `models/ontology.py` — templates render them at runtime |
| add an LLM provider | write an adapter, `register_provider()` it in `llm/factory.py` |
| change the stopping rule | `nodes/routing.py` — nothing else knows about it |
| add a 4th agent | new model + template dir + `<name>_agent.py` + node + one edge in `graph.py` |
| swap DBpedia for Wikidata | new `GroundingBackend` implementation; the grounder is unchanged |

Agents take an `LLMClient` and a `PromptRegistry` by constructor injection and
never read the environment, so they can be tested with a stub client and no
graph, no network:

```python
deps = build_dependencies(settings, llm=StubLLM(), backend=NullGroundingBackend())
run_pipeline(docs, settings=settings, graph=build_graph(deps))
```

## Testing without the network

`mcp.Client` accepts a server *instance*, so the grounding path runs over a real
protocol session with no port bound:

```python
from mcp_servers.dbpedia.server import build_server
from kg_agentic_extraction.grounding import MCPGroundingBackend

backend = MCPGroundingBackend(url=build_server())
```

## Configuration

All of `PipelineSettings` is env-driven with the `KG_` prefix — see
[.env.example](.env.example). The commonly changed ones:

| Variable | Default | Meaning |
|---|---|---|
| `CEREBRAS_API_KEY` | — | required |
| `KG_MODEL` | `gpt-oss-120b` | model id |
| `KG_MAX_ITERATIONS` | `3` | extractor↔grader rounds before giving up |
| `KG_PROMPT_VERSION` | `v1` | which template generation to resolve |
| `KG_MCP_URL` | `http://localhost:8931/mcp` | DBpedia MCP server |

Cerebras rotates model ids on its public endpoints — on a `model_not_found`
404, check the [model catalog](https://inference-docs.cerebras.ai/models/overview).

## Track 2 — training data

Four scripts, run in order, produce the labeled clean/corrupted dataset Track 3
needs. Each is independently re-runnable; outputs are skipped/resumed rather
than regenerated.

```bash
uv run python generate_training_data.py --num-clusters 250   # 2.1 (needs API key)
uv run python generate_corruptions.py                        # 2.2 (offline)
uv run python generate_splits.py                             # 2.4
uv run python check_dataset.py                               # 2.5
```

Produces `data/training/{clean,corrupted}/`, `splits.json`, and a summary report.
Corruption types: `missing_entities`, `contradictions`, `fragmentation`, plus
`entity_duplication`, `relation_type_swap`, `orphan_node_injection` behind
`--include-extra-types`. See `validator/corruption.py`'s `label_for()` for how
these map onto `SimpleGNN`'s output heads.

> **Note:** `generate_training_data.py` still runs the legacy
> `poc.extraction.service` extractor, because the clean KGs already on disk came
> from it. Rewriting Track 2 against `kg_agentic_extraction/` means regenerating
> the dataset from scratch — see [poc/README.md](poc/README.md).

## Status

| Track | State |
|---|---|
| 0 — stabilize | done |
| 1 — unify extraction | superseded by the rewrite ([ADR 0002](docs/adr/0002-agentic-pipeline.md)) |
| 2 — training data | scripts done; 13 clean clusters generated of a 200–300 target (Cerebras daily token quota) |
| 3 — GNN validator | not started (`validator/model.py`, `train.py`, `infer.py`) |
| 4 — refinement | not started |
| 5 — evaluation | not started |

## Research context

- **CoKG** (Lim et al., 2025) — Chain of Knowledge Graph for multi-doc summarization
- **CoD** (Adams et al., 2023) — Chain of Density prompting
- **CoE** (Bao et al., 2024) — Chain of Event prompting
