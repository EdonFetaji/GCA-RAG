# Graph-Consistency-Aware RAG

Agentic knowledge-graph extraction over the Multi-News dataset: an **extractor**
and a **grader** loop until the graph is clean, then an optional **grounder**
maps entities and relations to DBpedia.

```
extract → grade ─┬─ issues found → extract      (loop, up to KG_MAX_ITERATIONS)
                 └─ clean        → ground → done
```

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```bash
# provider + its API key  (gemini | groq | cerebras | mistral | nvidia | meta)
KG_LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
KG_MODEL=gemini-2.5-flash

# for a long batch: several keys, comma-separated — the run rotates
# through them as each hits its daily quota
GEMINI_API_KEYS=key1,key2,key3

# batch range (inclusive) and how many clusters run at once
KG_CLUSTER_START=0
KG_CLUSTER_END=199
KG_MAX_WORKERS=4

# optional: also copy each finished graph to a GCS bucket
# (needs `gcloud auth application-default login`)
KG_GCS_BUCKET=
```

## Run one cluster

```bash
uv run python -m kg_agentic_extraction.runner --cluster 0 --no-grounding
```

Remove `--no-grounding` to also map the graph to DBpedia — that needs the MCP
server running in a second terminal:

```bash
uv run python -m mcp_server.server
```

## Run a batch

Extracts every cluster in the `.env` range, writes each to
`kg_dataset/data/cluster_<i>.h5`, and uploads it if `KG_GCS_BUCKET` is set.

```bash
uv run python -m kg_agentic_extraction.batch --no-grounding
```

Useful flags:

| flag | effect |
|---|---|
| `--range 0 40` | override the range from `.env` |
| `--workers 2` | override `KG_MAX_WORKERS` |
| `--force` | re-extract clusters that are already done |

**Resumable** — a cluster that's already done is skipped, so just re-run the same
command to continue after a crash or when the API keys hit their daily quota.
Done-ness is read from the GCS bucket when one is configured, otherwise from
`kg_dataset/data/`.

**Exit codes:** `0` all done · `1` some clusters failed · `2` every API key is
spent for the day — re-run after the quota resets (~24h).

For a run that outlives your SSH session:

```bash
nohup uv run python -m kg_agentic_extraction.batch --no-grounding >> batch.log 2>&1 &
tail -f batch.log
```

## Tests

```bash
uv run pytest
```

## Layout

| path | what |
|---|---|
| `kg_agentic_extraction/` | the pipeline — LangGraph graph, 3 agents, config |
| `mcp_server/` | standalone DBpedia service the grounder calls |
| `validator/`, `data/training/` | Track 3 GNN consistency validator + its dataset |
| `scripts/cleanup.sh` | clears caches / stale temp files (runs after each batch) |

All settings are `KG_`-prefixed environment variables read into
`PipelineSettings` (`kg_agentic_extraction/config.py`).

## Research context

- **CoKG** (Lim et al., 2025) — Chain of Knowledge Graph for multi-doc summarization
- **CoD** (Adams et al., 2023) — Chain of Density prompting
- **CoE** (Bao et al., 2024) — Chain of Event prompting
