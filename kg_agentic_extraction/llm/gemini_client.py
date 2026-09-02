"""
Google Gemini adapter for the `LLMClient` port, via langchain-google-genai.

Third sibling of `cerebras_client.py` / `groq_client.py`. Same shape, different
vendor — no agent, node, or graph change was needed to add it.

Note the constructor keyword differences: `ChatGoogleGenerativeAI` exposes the
key as `api_key`, the completion cap as `max_tokens` (aliasing its internal
`max_output_tokens`), and the retry count as `retries`. Those aliases are used
below so this adapter presents the same signature as its siblings.

**Key rotation.** Unlike the other adapters this one holds a *list* of API keys
and rotates through them. Google's free tier caps requests per day per key, so a
long batch run exhausts one key and has to move to the next. Rotation is handled
here, below `structured()` / `run_tool_loop()`, so nothing upstream — no agent,
node, or the batch runner — has to know a key changed mid-run. Two failure modes
are distinguished from the 429 body:

- **per-day quota** — the key is done until Google resets it (~24h). It is
  marked exhausted and skipped for the rest of the process.
- **per-minute / per-second rate limit** — transient. Rotate to the next key
  immediately; if every key is rate-limited at once, sleep for the server's
  `retryDelay` and try the cycle again.

When every key has hit its *daily* quota there is nothing left to do today, so
`AllGeminiKeysExhausted` is raised. It is a `BaseException` on purpose: it has to
propagate cleanly through the per-node ``except Exception`` handlers and abort
the whole batch rather than being logged as one cluster's error.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from kg_agentic_extraction.llm.base import LLMError, LLMStructuredOutputError
from kg_agentic_extraction.llm.tool_calling import LangChainToolLoopMixin

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class AllGeminiKeysExhausted(BaseException):
    """
    Every configured Gemini API key has hit its per-day free-tier quota.

    Deliberately a `BaseException`, not an `LLMError`: no further cluster can
    make progress until Google resets the quotas (~24h), so this must escape the
    per-node ``except Exception`` handlers and stop the batch, not be recorded as
    a single cluster's failure. The batch runner catches it, reports how far it
    got, and exits with a resume-friendly status.
    """


# 429 bodies that mean "this key is spent for the day" carry a quota id like
# `GenerateRequestsPerDayPerProjectPerModel-FreeTier`. A per-minute limit reads
# `...PerMinute...` instead and is transient.
_DAILY_MARKERS = ("perday", "per day", "per-day", "requests per day", "/day", "daily limit")
_QUOTA_MARKERS = ("resource_exhausted", "resourceexhausted", "rate limit", "ratelimit", "quota")
_RETRY_DELAY_RE = re.compile(r"retry[_\s-]*delay['\"}\s:]*['\"]?(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _error_chain_text(exc: BaseException, *, depth: int = 6) -> str:
    """
    Every message in an exception's cause/context chain, lowercased and joined.

    The 429 can arrive wrapped a few layers deep — `LLMStructuredOutputError`
    around `ChatGoogleGenerativeAIError` around `google.genai` `ClientError` —
    and only the innermost one carries the quota id, so match against all of it.
    """
    seen: set[int] = set()
    parts: list[str] = []
    cursor: BaseException | None = exc
    for _ in range(depth):
        if cursor is None or id(cursor) in seen:
            break
        seen.add(id(cursor))
        parts.append(str(cursor))
        # LLMStructuredOutputError keeps the provider exception on `.cause`;
        # google's APIError keeps the parsed body on `.details`.
        for attr in ("cause", "details"):
            extra = getattr(cursor, attr, None)
            if extra is not None:
                parts.append(str(extra))
        cursor = cursor.__cause__ or cursor.__context__
    return " ".join(parts).lower()


def _is_quota_error(exc: BaseException) -> bool:
    """True for any 429 / rate-limit / quota response from Gemini."""
    text = _error_chain_text(exc)
    if "429" in text and ("quota" in text or "rate" in text or "exhaust" in text):
        return True
    return any(marker in text for marker in _QUOTA_MARKERS)


def _is_daily_quota_error(exc: BaseException) -> bool:
    """True when the 429 is a per-day cap (key spent) rather than a rate limit."""
    text = _error_chain_text(exc)
    return any(marker in text for marker in _DAILY_MARKERS)


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The `retryDelay` Gemini suggests, if the body carries one."""
    match = _RETRY_DELAY_RE.search(_error_chain_text(exc))
    return float(match.group(1)) if match else None


class GeminiClient(LangChainToolLoopMixin):
    """
    `LLMClient` implementation backed by Google Gemini, with API-key rotation.

    `with_structured_output` already defaults to `method="json_schema"` here, so
    unlike the Groq adapter there is nothing to override — Gemini constrains
    decoding to the Pydantic schema natively.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str = "",
        api_keys: list[str] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        max_retries: int = 2,
        rotation_cooldown_seconds: float = 60.0,
        max_rotation_wait_seconds: float = 900.0,
    ) -> None:
        # Imported lazily so that merely importing the pipeline (to inspect the
        # graph, run unit tests with a fake client, etc.) does not require the
        # provider SDK to be installed or an API key to be present.
        from langchain_google_genai import ChatGoogleGenerativeAI

        keys: list[str] = []
        for raw in [api_key, *(api_keys or [])]:
            cleaned = (raw or "").strip()
            if cleaned and cleaned not in keys:
                keys.append(cleaned)
        if not keys:
            raise LLMError("GeminiClient needs at least one API key")

        self._make_llm = lambda key: ChatGoogleGenerativeAI(
            model=model,
            api_key=key,
            temperature=temperature,
            max_tokens=max_tokens,
            retries=max_retries,
        )
        self._model_name = model
        self._keys = keys
        self._idx = 0
        self._daily_exhausted: set[int] = set()
        self._cooldown = rotation_cooldown_seconds
        self._max_wait = max_rotation_wait_seconds
        # Guards rotation only, not `invoke` — a call in flight on a stale key
        # simply fails and rotates itself, which the retry loop below absorbs.
        self._lock = threading.RLock()
        self._llm = self._make_llm(keys[0])

        if len(keys) > 1:
            logger.info("GeminiClient — %d API keys available for rotation", len(keys))

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def _key_label(self) -> str:
        return f"key {self._idx + 1}/{len(self._keys)}"

    # ── Public port ───────────────────────────────────────────────────

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[TModel],
    ) -> TModel:
        """Invoke the model and return a validated `schema` instance."""
        logger.debug("Gemini call — model=%s schema=%s", self._model_name, schema.__name__)

        def once() -> TModel:
            runnable = self._llm.with_structured_output(schema, method="json_schema")
            try:
                result = runnable.invoke(
                    [SystemMessage(content=system), HumanMessage(content=user)]
                )
            except Exception as exc:  # provider/validation failures alike
                raise LLMStructuredOutputError(schema, exc) from exc
            if not isinstance(result, schema):
                raise LLMStructuredOutputError(
                    schema, TypeError(f"provider returned {type(result).__name__}")
                )
            return result

        return self._with_rotation(once, schema)

    def run_tool_loop(self, **kwargs: object) -> BaseModel:
        """Rotate keys around a whole tool-calling pass; see the port for the contract."""
        schema = kwargs["schema"]  # type: ignore[assignment]
        return self._with_rotation(
            lambda: LangChainToolLoopMixin.run_tool_loop(self, **kwargs),  # type: ignore[arg-type]
            schema,  # type: ignore[arg-type]
        )

    def _structured_runnable(self, schema: type[TModel]) -> Runnable:
        """`json_schema` for the tool loop's closing call, matching `structured()`."""
        return self._llm.with_structured_output(schema, method="json_schema")

    # ── Rotation ─────────────────────────────────────────────────────

    def _with_rotation(self, op, schema: type[BaseModel]):
        """
        Run `op`, rotating the API key on any 429 and retrying.

        A per-day 429 marks the key spent; a rate-limit 429 just moves on. When
        every remaining key is rate-limited at once, sleep once for the server's
        `retryDelay` (bounded by `max_rotation_wait_seconds`) and cycle again.
        """
        total_wait = 0.0
        # Consecutive rate-limit rotations with no other progress. Reset whenever
        # a key is permanently retired (daily quota) or after a sleep, so a
        # genuine per-minute wave triggers one bounded sleep rather than a spin.
        rate_rotations = 0
        while True:
            try:
                return op()
            except AllGeminiKeysExhausted:
                raise
            except BaseException as exc:  # noqa: BLE001 — re-raised unless it is a 429
                if not _is_quota_error(exc):
                    raise

                with self._lock:
                    daily = _is_daily_quota_error(exc)
                    if daily:
                        self._daily_exhausted.add(self._idx)
                        rate_rotations = 0
                        logger.warning(
                            "Gemini %s hit its daily quota (%d/%d keys spent)",
                            self._key_label,
                            len(self._daily_exhausted),
                            len(self._keys),
                        )
                    else:
                        rate_rotations += 1
                        logger.warning("Gemini %s rate-limited; rotating", self._key_label)

                    if len(self._daily_exhausted) >= len(self._keys):
                        raise AllGeminiKeysExhausted(
                            f"all {len(self._keys)} Gemini API keys have hit their daily quota"
                        ) from exc

                    self._advance_key()
                    active = len(self._keys) - len(self._daily_exhausted)
                    need_sleep = rate_rotations >= active

                if need_sleep:
                    delay = _retry_after_seconds(exc) or self._cooldown
                    if total_wait + delay > self._max_wait:
                        logger.error(
                            "every Gemini key still rate-limited after %.0fs; giving up",
                            total_wait,
                        )
                        raise LLMStructuredOutputError(schema, exc) from exc
                    logger.warning(
                        "all live Gemini keys rate-limited — sleeping %.0fs before retry", delay
                    )
                    time.sleep(delay)
                    total_wait += delay
                    rate_rotations = 0

    def _advance_key(self) -> None:
        """Move to the next key that has not hit its daily quota. Call under the lock."""
        n = len(self._keys)
        for step in range(1, n + 1):
            candidate = (self._idx + step) % n
            if candidate not in self._daily_exhausted:
                self._idx = candidate
                self._llm = self._make_llm(self._keys[candidate])
                logger.info("Gemini — switched to %s", self._key_label)
                return
        # Caller has already checked at least one key is live, so this is
        # unreachable; guard anyway rather than silently keep a spent key.
        raise AllGeminiKeysExhausted("no live Gemini API key remains")
