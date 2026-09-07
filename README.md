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

Copy `.env.example` to `.env` and fill it in. The short version:

```bash
# provider + its API key  (gemini | groq | cerebras | mistral | nvidia | meta)
KG_LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key
KG_MODEL=gemini-2.5-flash

# the extractor and the grader run on different vendors — see "Which model
# runs what" below
KG_EXTRACTOR_PROVIDER=gemini
KG_EXTRACTOR_MODEL=gemini-2.5-flash
KG_GRADER_PROVIDER=mistral
KG_GRADER_MODEL=mistral-large-latest

# batch range (inclusive) and how many worker processes run it
KG_CLUSTER_START=0
KG_CLUSTER_END=199
KG_MAX_WORKERS=4

# one key bundle per worker — 2 Gemini (extractor) + 1 for the grader, on
# whatever KG_GRADER_PROVIDER names. Repeated for workers 0..3; no key may
# appear in two bundles.
KG_WORKER_0_GEMINI_KEYS=key-a,key-b
KG_WORKER_0_GRADER_KEY=key-g
# ...KG_WORKER_1_*, KG_WORKER_2_*, KG_WORKER_3_*

# optional: also copy each finished graph to a GCS bucket
# (needs `gcloud auth application-default login`)
KG_GCS_BUCKET=
```

## Which model runs what

| agent | provider | keys | why |
|---|---|---|---|
| extractor | Gemini | 2 per worker, rotated | sends the documents and gets a whole graph back every repair round — this is where the tokens go |
| grader | Mistral | 1 per worker | reads the graph once and answers with prose |
| grounder | — | — | off in batch mode; `runner.py` only |

The grader writes its Markdown report itself (prompt templates `grader/v4.*`)
rather than returning typed issues for Python to render. It reports convergence
as an explicit flag beside that prose, which is what keeps the loop's stopping
rule off the formatting.

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
Grounding is off here unconditionally.

```bash
uv run python -m kg_agentic_extraction.batch
```

`KG_MAX_WORKERS` **processes**, one per core, each running a single-threaded
pipeline over its own round-robin slice of the range. Processes rather than
threads because of the keys, not the CPU: each worker owns its bundle outright,
so a Gemini key one worker exhausts stays usable by the other three, and a worker
that runs out of quota stops alone instead of ending the batch. Threads in one
process would share the rotating client's "this key is spent" state.

Useful flags:

| flag | effect |
|---|---|
| `--range 0 40` | override the range from `.env` |
| `--workers 2` | override `KG_MAX_WORKERS` (capped by the number of key bundles) |
| `--force` | re-extract clusters that are already done |
| `--no-upload` | skip GCS even when a bucket is set |

**Resumable** — a cluster that's already done is skipped, so just re-run the same
command to continue after a crash or when the API keys hit their daily quota.
Done-ness is read from the GCS bucket when one is configured, otherwise from
`kg_dataset/data/`.

**Exit codes:** `0` all done · `1` some clusters failed · `2` at least one
worker's Gemini keys are spent for the day — re-run after the quota resets (~24h).

For a run that outlives your SSH session:

```bash
nohup uv run python -m kg_agentic_extraction.batch >> batch.log 2>&1 &
tail -f batch.log
```

Each worker prefixes its log lines with `[w0]`, `[w1]`, … so four interleaved
streams stay readable.

### Multi-day runs on a VM

A few hundred clusters on free-tier keys outlasts a daily quota, so the batch
will exit `2` partway and has to be re-entered. `scripts/run-until-done.sh` does
that unattended — it re-runs the batch (which resumes, skipping finished
clusters), sleeps an hour on a quota wall, and stops once the range is complete
or three rounds pass with no new graph:

```bash
uv run python scripts/preflight.py --range 0 199 --check-keys   # validate keys first
uv run python -m kg_agentic_extraction.batch --range 0 0        # warm the HF cache
nohup scripts/run-until-done.sh --range 0 199 >> supervisor.log 2>&1 &
```

Set `KG_GCS_BUCKET` for a run like this: resume state then lives in the bucket
rather than on a disk you might lose. `scripts/kg-batch.service` is the same
thing as a systemd unit, which also survives a reboot.

**[docs/vm-runbook.md](docs/vm-runbook.md)** is the full walkthrough — VM sizing,
`.env`, preflight, systemd, monitoring, and a troubleshooting table.

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
