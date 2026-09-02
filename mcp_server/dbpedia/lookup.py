"""
The DBpedia Lookup transport.

Lookup is keyword search over resource labels and is what `search_resource`
leans on. Two quirks of its JSON drive most of the code here:

1. Every field is a **list of strings**, even single-valued ones
   (`"refCount": ["9401"]`).
2. `label` and `comment` come back with `<B>…</B>` highlight markup around the
   matched terms. Passing that through would put HTML in the grounder's prompt
   and, worse, into `OntologyMapping.label`.
"""

from __future__ import annotations

import html
import logging
import re
import time

import httpx

from mcp_server.dbpedia.cache import TTLCache
from mcp_server.dbpedia.config import DBpediaSettings, get_settings

logger = logging.getLogger(__name__)

_TAG = re.compile(r"<[^>]+>")

_cache: TTLCache | None = None


class LookupError_(RuntimeError):
    """Lookup could not be reached or returned something unusable."""


def _get_cache(settings: DBpediaSettings) -> TTLCache:
    global _cache
    if _cache is None:
        _cache = TTLCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
        )
    return _cache


def search(
    query: str,
    *,
    type_name: str | None = None,
    max_results: int = 10,
    settings: DBpediaSettings | None = None,
) -> list[dict[str, object]]:
    """
    Keyword-search DBpedia resources.

    `type_name` is Lookup's own filter and takes the *short* class name
    (`City`, not `dbo:City`) — see `classes.short_name`.

    Returns rows of `{uri, label, comment, types, ref_count, score}`.

    Raises
    ------
    LookupError_
        On transport failure or a non-200 response, after retries.
    """
    settings = settings or get_settings()
    cache = _get_cache(settings)
    cache_key = f"{settings.lookup_endpoint}|{query}|{type_name}|{max_results}"

    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Lookup cache hit for %r", query)
        return cached

    params: dict[str, str | int] = {
        "query": query,
        "format": "JSON",
        "maxResults": max_results,
    }
    if type_name:
        params["typeName"] = type_name

    payload = _request(params, settings)
    rows = [_row(doc) for doc in payload.get("docs", [])]
    cache.set(cache_key, rows)
    return rows


def _request(params: dict[str, str | int], settings: DBpediaSettings) -> dict[str, object]:
    last_error: Exception | None = None

    for attempt in range(settings.max_retries + 1):
        if attempt:
            time.sleep(2.0**attempt * 0.5)
        try:
            response = httpx.get(
                settings.lookup_endpoint,
                params=params,
                headers={"Accept": "application/json"},
                timeout=settings.http_timeout,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("Lookup transport error (attempt %d): %s", attempt + 1, exc)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last_error = LookupError_(f"lookup returned {response.status_code}")
            continue
        if response.status_code != 200:
            raise LookupError_(f"lookup returned {response.status_code}: {response.text[:200]}")

        try:
            return response.json()
        except ValueError as exc:
            raise LookupError_(f"lookup returned non-JSON: {response.text[:200]}") from exc

    raise LookupError_(f"lookup failed after {settings.max_retries + 1} attempts: {last_error}")


# ── Response shaping ──────────────────────────────────────────────────


def _row(doc: dict[str, object]) -> dict[str, object]:
    return {
        "uri": _first(doc, "resource") or _first(doc, "id") or "",
        "label": clean(_first(doc, "label")),
        "comment": clean(_first(doc, "comment")),
        "types": [t for t in _all(doc, "type") if t.startswith("http://dbpedia.org/ontology/")],
        "ref_count": _as_int(_first(doc, "refCount")),
        "score": _as_float(_first(doc, "score")),
    }


def _first(doc: dict[str, object], key: str) -> str:
    """Lookup wraps every value in a list, including the single-valued ones."""
    value = doc.get(key)
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value) if value is not None else ""


def _all(doc: dict[str, object], key: str) -> list[str]:
    value = doc.get(key)
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)] if value is not None else []


def clean(text: str) -> str:
    """Strip Lookup's `<B>` highlight markup and unescape entities."""
    return html.unescape(_TAG.sub("", text or "")).strip()


def _as_int(text: str) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


def _as_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def reset_cache() -> None:
    """Drop the module-level cache. For tests."""
    global _cache
    if _cache is not None:
        _cache.clear()
    _cache = None
