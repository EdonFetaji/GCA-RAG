#!/usr/bin/env bash
# Supervise `python -m kg_agentic_extraction.batch` across a multi-day run.
#
#   scripts/run-until-done.sh --range 0 199
#   nohup scripts/run-until-done.sh --range 0 199 >> supervisor.log 2>&1 &
#
# Every argument is forwarded verbatim to the batch module.
#
# Why this exists: the batch is *designed* to stop. When every Gemini key in a
# worker's bundle hits its per-day cap that worker unwinds and, once they all
# have, the process exits 2 — nothing can make progress until Google resets the
# quotas (~24h). A few hundred clusters on free-tier keys therefore spans
# several days and several processes. This loop is what turns "re-run it
# tomorrow" into an unattended run: it re-enters the batch, which reads the
# done-set, skips what is finished, and picks up the rest.
#
# Batch exit codes it reacts to (see `BatchSummary.exit_code`):
#   0  every cluster in range is done   -> stop, success
#   2  a worker's daily quota is spent  -> sleep QUOTA_SLEEP, retry
#   1  some clusters failed             -> sleep ERROR_SLEEP, retry
#   *  crash / OOM / SIGKILL            -> sleep ERROR_SLEEP, retry
#
# A cluster that fails deterministically (a document the extractor can never
# satisfy the grader on) would otherwise spin here forever, so a round that ends
# non-zero *and* wrote no new graph counts as a stall. MAX_STALLS consecutive
# stalls end the run for a human to look at, rather than burning quota on the
# same failure until the VM is turned off.
#
# Knobs, all environment variables:
#   QUOTA_SLEEP   seconds to wait after exit 2          (default 3600)
#   ERROR_SLEEP   seconds to wait after a failed round  (default 600)
#   MAX_STALLS    consecutive no-progress rounds        (default 3)
#   BATCH_LOG     where the batch's own output goes     (default batch.log)
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

QUOTA_SLEEP="${QUOTA_SLEEP:-3600}"
ERROR_SLEEP="${ERROR_SLEEP:-600}"
MAX_STALLS="${MAX_STALLS:-3}"
BATCH_LOG="${BATCH_LOG:-batch.log}"

# The batch reads .env itself; this is only so progress can be counted in the
# right directory when KG_GRAPH_OUTPUT_DIR lives there rather than the shell.
if [[ -z ${KG_GRAPH_OUTPUT_DIR:-} && -f .env ]]; then
	KG_GRAPH_OUTPUT_DIR="$(sed -n 's/^[[:space:]]*KG_GRAPH_OUTPUT_DIR[[:space:]]*=[[:space:]]*//p' .env | tail -1 | tr -d "\"'")"
fi
GRAPH_DIR="${KG_GRAPH_OUTPUT_DIR:-kg_dataset/data}"

log() { printf '%s [supervisor] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$*"; }

# Graphs on local disk. Written before the GCS upload, so this is a valid
# progress signal whether or not a bucket is configured.
graph_count() {
	[[ -d $GRAPH_DIR ]] || { echo 0; return; }
	find "$GRAPH_DIR" -maxdepth 1 -type f -name 'cluster_*.h5' 2>/dev/null | wc -l | tr -d ' '
}

on_signal() {
	log "signal received — stopping (the batch resumes where it left off)"
	exit 130
}
trap on_signal INT TERM

round=0
stalls=0
log "starting — batch args: $*"
log "batch output -> $BATCH_LOG   graphs -> $GRAPH_DIR"

while true; do
	round=$((round + 1))
	before="$(graph_count)"
	log "round $round starting ($before graph(s) on disk)"

	uv run python -m kg_agentic_extraction.batch "$@" >>"$BATCH_LOG" 2>&1
	code=$?

	after="$(graph_count)"
	made=$((after - before))
	log "round $round exited $code — $made new graph(s) this round, $after total"

	if (( code == 0 )); then
		log "range complete — nothing left to do"
		exit 0
	fi

	# Progress resets the stall counter: a quota wall that still extracted
	# clusters before hitting it is a healthy round, not a stuck one.
	if (( made > 0 )); then
		stalls=0
	else
		stalls=$((stalls + 1))
		log "no progress — stall $stalls/$MAX_STALLS"
		if (( stalls >= MAX_STALLS )); then
			log "giving up after $stalls rounds with no new graphs; see $BATCH_LOG"
			exit "$code"
		fi
	fi

	if (( code == 2 )); then
		# Free-tier quotas reset at midnight Pacific. Rather than reasoning
		# about the VM's timezone, wake up hourly and let the first attempt
		# past the reset succeed — a spent-quota round costs one call per
		# worker to discover.
		log "daily quota spent — sleeping ${QUOTA_SLEEP}s"
		sleep "$QUOTA_SLEEP"
	else
		log "failed round — sleeping ${ERROR_SLEEP}s before retry"
		sleep "$ERROR_SLEEP"
	fi
done
