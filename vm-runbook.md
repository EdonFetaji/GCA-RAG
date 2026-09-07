# Running a batch extraction on a VM

How to extract a few hundred Multi-News clusters on a cloud VM and leave it
alone for days. Written for GCE, but nothing here is Google-specific except the
bucket and the service account.

## What you are signing up for

The batch **is designed to stop**. Free-tier Gemini keys have a per-day request
cap; when every key in a worker's bundle hits it, that worker unwinds, and once
they all have, the process exits with status `2` — nothing can make progress
until Google resets the quotas (~24h, at midnight Pacific).

So a few hundred clusters is not one long process. It is a chain of processes,
each resuming where the last stopped. `scripts/run-until-done.sh` is what makes
that chain unattended, and `scripts/kg-batch.service` is what keeps it alive
across a reboot. The rest of this document is the setup around them.

Rough shape of a run: four workers, one cluster at a time each, a few minutes
per cluster depending on how many extractor↔grader rounds it takes to converge.
Expect to cover on the order of a hundred clusters a day on free-tier keys, and
to spend most of the wall clock waiting for the quota to reset.

## 1. Provision the VM

| | |
|---|---|
| machine type | `e2-standard-4` (4 vCPU, 16 GB) — batch runs one process per worker and `KG_MAX_WORKERS` is capped at 4, so more vCPUs buy nothing |
| disk | 30 GB+ — torch and its CUDA-less wheels, the uv cache, and the Hugging Face dataset cache |
| image | any Debian/Ubuntu LTS; uv installs its own Python 3.12 if the system one is older |
| GPU | none. Extraction is API calls; the GNN validator is a separate track |

**Do not use a spot/preemptible instance** unless you are running under systemd
*and* have `KG_GCS_BUCKET` set. Preemption mid-cluster is harmless in itself —
the `.h5` is written by atomic rename, so there is never a half-written graph —
but with `nohup` nothing restarts the run afterwards.

If you want the graphs in a bucket (recommended, see §3), give the VM a service
account with read+write on it at creation time. Application Default Credentials
then resolve automatically and no key file is needed.

## 2. Install

```bash
git clone <this repo> GCA-RAG && cd GCA-RAG
curl -LsSf https://astral.sh/uv/install.sh | sh
exec $SHELL -l                 # put ~/.local/bin on PATH
uv sync                        # resolves from uv.lock — do not use pip
```

`uv.lock` is tracked deliberately: four worker processes have to resolve to the
same dependency set as each other and as your laptop.

## 3. Configure `.env`

`.env` is **not** in the repo (it is gitignored). Recreate it on the VM:

```bash
cp .env.example .env && nano .env
```

For a batch run, these are the settings that matter. Everything else in
`.env.example` belongs to single-cluster runs via `runner.py`.

| setting | why |
|---|---|
| `KG_WORKER_0..3_GEMINI_KEYS` | two Gemini keys per worker, comma-separated. The extractor rotates them |
| `KG_WORKER_0..3_GRADER_KEY` | one key per worker for the grader |
| `KG_MAX_WORKERS` | worker processes, one bundle each. Hard-capped at 4 |
| `KG_GRADER_PROVIDER` / `KG_GRADER_MODEL` | which vendor the grader keys belong to |
| `KG_EXTRACTOR_PROVIDER` / `KG_EXTRACTOR_MODEL` | the extractor pair |
| `KG_GCS_BUCKET` | bucket for the graphs, and the resume source of truth |
| `KG_CLUSTER_START` / `KG_CLUSTER_END` | default range; `--range` overrides |

Two things that trip people up:

- **No key may appear in two bundles.** The entire reason batch mode uses
  processes rather than threads is that a key spent by one worker must stay
  usable by the others. `preflight.py` checks this.
- **The top-level provider keys (`GEMINI_API_KEY`, `NVIDIA_API_KEY`, …) are not
  used in batch mode.** Each worker's bundle *replaces* them, and the bundle's
  `grader_key` is routed to whichever provider `KG_GRADER_PROVIDER` names. You
  do not need to fill in section 1 of `.env.example` for a batch run.

### Why set `KG_GCS_BUCKET`

With a bucket configured, "which clusters are already done" is read from the
bucket rather than from local disk. On a VM you might resize, recreate, or lose
to preemption, that is the difference between resuming and re-extracting
everything. The local `.h5` is still written first and stays the source of
truth for the file itself; the upload is a copy.

Without a bucket, resume falls back to `kg_dataset/data/` and the run is only as
durable as the disk.

## 4. Validate before spending quota

```bash
uv run python scripts/preflight.py --range 0 199 --check-keys
```

Twelve keys are twelve chances to be one typo away from a run that dies forty
minutes in. `--check-keys` makes one trivial call per key through the real
adapters, and the failure modes that matter — a key pasted into two bundles, a
bundle missing its grader key, a revoked key — are otherwise invisible until the
worker that owns it starts. It also prints the shard plan.

Then warm the dataset cache with a single cluster:

```bash
uv run python -m kg_agentic_extraction.batch --range 0 0
```

This proves the whole path end to end (dataset → extract → grade → `.h5` →
upload) and downloads Multi-News once. Worth doing separately because the four
spawned workers each call `load_dataset` and, on a cold cache, race on the
Hugging Face download locks.

## 5. Run it

### Option A — systemd (recommended for multi-day runs)

Survives reboots, restarts the supervisor if it dies, keeps history in
`journalctl`. The unit in `scripts/` is a template — render it for your user and
path:

```bash
cd ~/GCA-RAG
sed -e "s|/home/edon/GCA-RAG|$PWD|g" \
    -e "s|^User=edon|User=$(whoami)|" \
    -e "s|/home/edon/.local/bin|$HOME/.local/bin|" \
    scripts/kg-batch.service | sudo tee /etc/systemd/system/kg-batch.service
```

Then set your range on the `ExecStart=` line in
`/etc/systemd/system/kg-batch.service`:

```ini
ExecStart=/home/you/GCA-RAG/scripts/run-until-done.sh --range 0 199
```

Start it:

```bash
sudo systemctl daemon-reload          # after every edit to the unit
sudo systemctl enable --now kg-batch  # enable = also start at boot
```

### Option B — nohup

Fine for a run you will be around for; does not survive a reboot.

```bash
nohup scripts/run-until-done.sh --range 0 199 >> supervisor.log 2>&1 &
```

### What the supervisor does

It re-runs the batch until the range is complete, reacting to the exit code:

| exit | meaning | supervisor |
|---|---|---|
| `0` | every cluster in range is done | stops, success |
| `2` | a worker's daily quota is spent | sleeps `QUOTA_SLEEP` (1h), retries |
| `1` | some clusters failed | sleeps `ERROR_SLEEP` (10m), retries |
| other | crash, OOM, SIGKILL | sleeps `ERROR_SLEEP`, retries |

A round that ends non-zero *and* wrote no new graph counts as a stall;
`MAX_STALLS` (3) consecutive stalls end the run, so a cluster that fails
deterministically cannot burn quota in a loop. Progress resets the counter — a
round that extracted clusters before hitting the quota wall is healthy, not
stuck.

Override any of them in the environment:

```bash
QUOTA_SLEEP=1800 MAX_STALLS=5 scripts/run-until-done.sh --range 0 199
```

## 6. Monitor

```bash
systemctl status kg-batch           # running? for how long? last exit?
journalctl -u kg-batch -f           # supervisor rounds
tail -f ~/GCA-RAG/batch.log         # per-cluster worker output
gsutil ls gs://<bucket>/ | wc -l    # ground truth on progress
```

The split is deliberate: the supervisor's own lines go to the journal, the
batch's verbose per-cluster output to `batch.log`. Worker lines are prefixed
`[w0]`, `[w1]`, … so four interleaved streams stay readable.

A healthy supervisor line looks like:

```
2026-01-14 03:11:02Z [supervisor] round 3 exited 2 — 41 new graph(s) this round, 128 total
```

## 7. Stop, resume, change range

```bash
sudo systemctl stop kg-batch        # safe at any point
sudo systemctl restart kg-batch     # after editing ExecStart (+ daemon-reload)
sudo systemctl disable kg-batch     # stop it returning at boot
```

Stopping is safe because the unit stops with `SIGINT`, not `SIGTERM`: the batch
catches `KeyboardInterrupt`, terminates its worker processes and joins them. The
in-flight cluster is abandoned and re-extracted on the next run.

To resume after any interruption — quota, crash, preemption, `Ctrl-C` — re-run
the exact same command. Finished clusters drop out; nothing is recomputed. To
re-extract clusters that are already done, add `--force`.

When the range completes, the supervisor exits `0` and systemd shows the unit
`inactive (dead)` with `status=0/SUCCESS`. That is success, not a crash —
`Restart=on-failure` deliberately does not restart a finished run.

## 8. Troubleshooting

| symptom | cause and fix |
|---|---|
| `no cluster range: pass --range START END` | no range on the command line and `KG_CLUSTER_START` / `KG_CLUSTER_END` unset in `.env` |
| `4 worker(s) requested but no key bundle for worker(s) 3` | `KG_MAX_WORKERS` exceeds the bundles you filled in. Add `KG_WORKER_3_*` or lower `--workers` |
| `worker 2 is missing: KG_WORKER_2_GRADER_KEY` | a bundle is half-filled. Fatal on purpose — a batch quietly running at half parallelism is only noticed hours later |
| `could not list gs://…` at startup | ADC cannot reach the bucket. Fatal on purpose: treating it as "nothing done" would re-extract everything. Check the VM's service account, or use `--resume local` |
| `--resume gcs needs KG_GCS_BUCKET set` | you asked for bucket resume without configuring one |
| unit fails instantly with status 203 | `ExecStart` path is wrong, or `run-until-done.sh` is not executable (`chmod +x`) |
| unit fails with `uv: command not found` | systemd's minimal PATH. Fix the `Environment=PATH=` line in the unit |
| supervisor gives up after 3 stalls | real, repeatable failures. Read `batch.log` for the cluster indices, then re-run that subset by hand |
| a worker never reported | killed before it could report (usually OOM). Its shard is counted failed and picked up on resume; consider fewer workers or a larger machine |
| dataset download hangs on first run | Hugging Face lock contention from four workers on a cold cache. Stop, run `scripts/cleanup.sh` (it clears stale HF locks, never cached data), warm with `--range 0 0` |

## Reference

**Batch exit codes:** `0` all done · `1` some clusters failed · `2` at least one
worker's keys are spent for the day.

**Off in batch mode:** grounding is disabled unconditionally, so no DBpedia MCP
server needs to run on the VM. Use `runner.py` for a single grounded cluster.

**Cleanup:** `scripts/cleanup.sh` runs after every batch (disable with
`--no-cleanup`). It removes stale `.h5.tmp`, Python caches, and dead Hugging
Face locks — never extracted graphs, the venv, or the dataset cache. Safe to run
in a loop, which is why the supervisor leaves it on.

**Related:** `README.md` for the batch flags, `.env.example` for every setting
with its explanation, `kg_agentic_extraction/config.py` for the authoritative
field descriptions.
