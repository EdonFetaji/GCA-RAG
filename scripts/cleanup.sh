#!/usr/bin/env bash
# Remove the transient files a pipeline run leaves behind. Safe to run any time:
# every deletion is either a regenerated cache or a file untouched for 10+
# minutes (longer than any live write), so a run in progress is never disturbed.
#
#   scripts/cleanup.sh              # clean
#   scripts/cleanup.sh --dry-run    # list what would go, delete nothing
#
# Called automatically at the end of `python -m kg_agentic_extraction.batch`
# (disable with --no-cleanup there). It does NOT touch extracted graphs
# (cluster_*.h5), the venv, the uv cache, or the Hugging Face dataset — those
# are deliberate, not leftovers. For the big caches see `uv cache prune`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DRY=0
if [[ "${1:-}" == "--dry-run" || "${1:-}" == "-n" ]]; then DRY=1; fi

GRAPH_DIR="${KG_GRAPH_OUTPUT_DIR:-kg_dataset/data}"
HF_HOME="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}}"
# A finished write renames its .tmp within seconds; 10 min is safely dead.
STALE_MIN=10

log()  { printf '  %s\n' "$*"; }
zap() {
	# zap <find-args...> — delete matches, or list them under --dry-run.
	local found=0
	while IFS= read -r -d '' path; do
		found=1
		if (( DRY )); then log "would remove  $path"; else rm -rf -- "$path"; log "removed  $path"; fi
	done < <(find "$@" -print0 2>/dev/null)
	return $(( found == 0 ))
}

before=$(df -Pk . | awk 'NR==2 {print $4}')
if (( DRY )); then echo "cleanup — $ROOT  (dry run)"; else echo "cleanup — $ROOT"; fi

# 1. Half-written HDF5 graphs from a killed run (the real .h5 is written via an
#    atomic rename, so a lingering .tmp never contains anything usable).
zap "$GRAPH_DIR" -maxdepth 1 -name '*.h5.tmp' -mmin +$STALE_MIN || log "no stale .h5.tmp"

# 2. Python / tooling caches — all regenerated on next run.
zap . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.ruff_cache' \) \
	-not -path './.venv/*' || log "no python caches"
zap "${TMPDIR:-/tmp}" -maxdepth 1 -name "pytest-of-$USER" || true

# 3. LangGraph dev-server checkpoint cache (rebuilt by `langgraph dev`).
zap . -maxdepth 2 -type d -name '.langgraph_api' -not -path './.venv/*' || true

# 4. Stale Hugging Face lock / partial-download files — harmless, but a leftover
#    lock can wedge the next dataset load. Never removes cached data itself.
if [[ -d "$HF_HOME" ]]; then
	zap "$HF_HOME" -type f \( -name '*.lock' -o -name '*_builder.lock' \
		-o -name '*.incomplete' -o -name '*.incomplete_info.lock' \) -mmin +$STALE_MIN \
		|| log "no stale HF locks"
fi

# 5. Truncate the MCP server log once it gets large (it grows unbounded).
if [[ -f mcp_logs.txt && $(stat -c%s mcp_logs.txt) -gt 1048576 ]]; then
	if (( DRY )); then log "would truncate  mcp_logs.txt ($(du -h mcp_logs.txt | cut -f1))"
	else : > mcp_logs.txt; log "truncated  mcp_logs.txt"; fi
fi

after=$(df -Pk . | awk 'NR==2 {print $4}')
freed=$(( (after - before) / 1024 ))
(( DRY )) || echo "cleanup done — ${freed} MiB freed on $(df -Ph . | awk 'NR==2 {print $6}')"
