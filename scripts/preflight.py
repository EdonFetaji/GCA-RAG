"""
Check a batch configuration before spending any quota on it.

    uv run python scripts/preflight.py                      # config only, no network
    uv run python scripts/preflight.py --range 0 3          # ...and the shard plan
    uv run python scripts/preflight.py --check-keys         # ...and one live call per key

Worth running before any long batch. A twelve-key setup has twelve ways to be
one typo away from a run that dies forty minutes in, and the failure modes that
matter — a key pasted into two bundles, a bundle missing its grader key, a key
that was revoked — are all invisible until the worker that owns it starts.

`--check-keys` costs one trivial completion per key (12 calls for 4 workers) and
goes through the real adapters, so it exercises the same path the batch will.
"""

from __future__ import annotations

import argparse
import sys

from pydantic import BaseModel

from kg_agentic_extraction.batch import shard
from kg_agentic_extraction.config import PipelineSettings
from kg_agentic_extraction.llm.factory import available_providers, build_llm


class Ping(BaseModel):
    """Smallest possible schema — this is a key check, not a capability check."""

    ok: bool


def mask(key: str) -> str:
    """Enough of a key to recognise it by, not enough to use."""
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "…"


def check_config(settings: PipelineSettings, workers: int) -> list[str]:
    """Report on the bundles. Returns a list of problems, empty when all is well."""
    problems: list[str] = []
    bundles = settings.worker_keys

    print(f"\n── Key bundles ({len(bundles)} found, {workers} needed) " + "─" * 24)
    if not bundles:
        problems.append(
            "no KG_WORKER_<n>_* bundles in the environment — see .env.example section 3"
        )

    for worker_id in range(workers):
        try:
            bundle = settings.worker_bundle(worker_id)
        except KeyError:
            print(f"  w{worker_id}  MISSING — no KG_WORKER_{worker_id}_* variables")
            problems.append(f"worker {worker_id} has no key bundle")
            continue

        gaps = bundle.missing()
        mark = "ok " if not gaps else "BAD"
        gemini = ", ".join(mask(k) for k in bundle.gemini_keys) or "(none)"
        print(f"  w{worker_id}  {mark}  gemini[{len(bundle.gemini_keys)}]: {gemini}")
        print(f"           grader: {mask(bundle.grader_key) if bundle.grader_key else '(none)'}")
        if gaps:
            problems.append(f"worker {worker_id} is missing {', '.join(gaps)}")
        if len(bundle.gemini_keys) < 2:
            print(f"           note: only {len(bundle.gemini_keys)} Gemini key — no rotation")

    # The invariant the whole process model rests on. A key pasted into two
    # bundles silently halves that worker pair's effective quota and makes one
    # worker's exhaustion look like the other's.
    seen: dict[str, str] = {}
    for bundle in bundles[:workers]:
        for key in [*bundle.gemini_keys, bundle.grader_key]:
            if not key:
                continue
            owner = f"w{bundle.worker_id}"
            if key in seen:
                problems.append(f"key {mask(key)} is in both {seen[key]} and {owner}")
            seen[key] = owner

    print("\n── Roles " + "─" * 55)
    known = available_providers()
    for role in ("extractor", "grader"):
        scoped = settings.for_role(role)
        print(f"  {role:<10} {scoped.llm_provider} / {scoped.model}")
        if not scoped.llm_provider:
            problems.append(f"{role} has no provider and no KG_LLM_PROVIDER fallback")
        elif scoped.llm_provider.lower() not in known:
            problems.append(
                f"{role} names unknown provider {scoped.llm_provider!r}; "
                f"registered: {', '.join(known)}"
            )

    print("\n── Loop " + "─" * 56)
    print(f"  max_iterations   {settings.max_iterations}")
    print(f"  max_documents    {settings.max_documents}")
    print("  grounding        off (batch mode always disables it)")
    print(f"  output dir       {settings.graph_output_dir}")
    print(f"  gcs bucket       {settings.gcs_bucket or '(none — no upload)'}")

    return problems


def check_keys(settings: PipelineSettings, workers: int) -> list[str]:
    """One trivial live call per key, through the real adapters."""
    from kg_agentic_extraction.llm.gemini_client import GeminiClient

    problems: list[str] = []
    print("\n── Live key check " + "─" * 46)

    for worker_id in range(workers):
        try:
            bundle = settings.worker_bundle(worker_id)
        except KeyError:
            continue

        try:
            scoped = settings.for_worker(worker_id)
        except ValueError as exc:
            problems.append(str(exc))
            return problems
        extractor = scoped.for_role("extractor")
        grader = scoped.for_role("grader")

        for key in bundle.gemini_keys:
            label = f"w{worker_id} gemini {mask(key)}"
            try:
                # One key at a time: a list would rotate past a dead key and
                # report success for the wrong one.
                client = GeminiClient(
                    model=extractor.model,
                    api_keys=[key],
                    temperature=0.0,
                    max_tokens=64,
                )
                client.structured(system="Reply with ok=true.", user="ping", schema=Ping)
                print(f"  ok   {label}")
            except Exception as exc:
                print(f"  DEAD {label}: {type(exc).__name__}: {str(exc)[:120]}")
                problems.append(f"{label} is not usable")

        if bundle.grader_key:
            label = f"w{worker_id} grader {grader.llm_provider} {mask(bundle.grader_key)}"
            try:
                # Built through the factory rather than a named adapter, so this
                # probes whatever KG_GRADER_PROVIDER points at. `for_worker` has
                # already put this bundle's grader key in that provider's field.
                client = build_llm(grader.model_copy(update={"temperature": 0.0, "max_tokens": 64}))
                # Plain completion, not `structured` — that is the call the
                # grader actually makes, so this fails for the same reasons a
                # run would rather than probing a path nothing uses.
                client.complete(system="Reply with the word ok.", user="ping")
                print(f"  ok   {label}")
            except Exception as exc:
                print(f"  DEAD {label}: {type(exc).__name__}: {str(exc)[:120]}")
                problems.append(f"{label} is not usable")

    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a batch configuration.")
    parser.add_argument("--workers", type=int, help="Default: KG_MAX_WORKERS.")
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "END"))
    parser.add_argument(
        "--check-keys",
        action="store_true",
        help="Make one trivial live call per key. Costs a few requests.",
    )
    args = parser.parse_args(argv)

    settings = PipelineSettings()
    workers = args.workers or settings.max_workers

    problems = check_config(settings, workers)

    if args.range:
        start, end = args.range
        todo = list(range(start, end + 1))
        print(f"\n── Shard plan for {start}–{end} " + "─" * 40)
        for worker_id, assigned in enumerate(shard(todo, workers)):
            print(f"  w{worker_id}  {len(assigned):>3} cluster(s)  {assigned}")
        print("  (this ignores resume — already-finished clusters drop out first)")

    if args.check_keys:
        problems += check_keys(settings, workers)

    print("\n" + "─" * 64)
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("all good — safe to run the batch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
