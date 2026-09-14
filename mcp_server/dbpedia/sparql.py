"""
The SPARQL transport.

One function — `run_select` — plus the two escaping helpers every caller must
route model-supplied input through. Queries go over **GET**: DBpedia's public
endpoint answers GET reliably, and POST was observed not to connect at all from
some networks.

The escaping helpers are not a nicety. `get_predicates_between` interpolates two
URIs that came from a language model, and a bare `>` in one of them would close
the angle-bracket term and let the rest of the string be read as query syntax.
Every interpolation in `tools.py` goes through `uri_term()` or `text_literal()`.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import httpx

from mcp_server.dbpedia.cache import TTLCache
from mcp_server.dbpedia.config import DBpediaSettings, get_settings

logger = logging.getLogger(__name__)

#: Prepended to every query so the tool bodies stay readable.
PREFIXES = """\
PREFIX dbo:  <http://dbpedia.org/ontology/>
PREFIX dbr:  <http://dbpedia.org/resource/>
PREFIX dbp:  <http://dbpedia.org/property/>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
"""

#: Characters RFC 3987 forbids inside an IRI term, plus the quote and backslash.
_URI_FORBIDDEN = re.compile(r'[<>"{}|\\^`\s]')

_cache: TTLCache | None = None


class SPARQLError(RuntimeError):
    """The endpoint could not be reached or returned something unusable."""


# ── Escaping ──────────────────────────────────────────────────────────


def uri_term(uri: str) -> str:
    """
    Wrap `uri` as a SPARQL IRI term, or raise.

    Rejects anything that is not an absolute http(s) URI, and anything
    containing a character that would terminate the term early. Raising is the
    right behaviour here: a URI that fails this check did not come from a tool
    result, so there is nothing to look up.
    """
    candidate = (uri or "").strip()
    if not candidate.startswith(("http://", "https://")):
        raise ValueError(f"not an absolute http(s) URI: {uri!r}")
    if _URI_FORBIDDEN.search(candidate):
        raise ValueError(f"URI contains characters that are illegal in a SPARQL term: {uri!r}")
    return f"<{candidate}>"


def text_literal(value: str) -> str:
    """Wrap `value` as a quoted SPARQL string literal, escaping what must be escaped."""
    escaped = (
        (value or "")
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


# ── Transport ─────────────────────────────────────────────────────────


def _get_cache(settings: DBpediaSettings) -> TTLCache:
    global _cache
    if _cache is None:
        _cache = TTLCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
        )
    return _cache


def run_select(query: str, *, settings: DBpediaSettings | None = None) -> list[dict[str, str]]:
    """
    Run a SELECT and return its bindings as plain `{var: value}` dicts.

    Unbound variables are simply absent from a row rather than present as None,
    so callers use `row.get(...)` and never have to distinguish "no value" from
    "the empty string".

    Raises
    ------
    SPARQLError
        On a transport failure, a non-200 response, or unparseable results —
        after `settings.max_retries` attempts.
    """
    settings = settings or get_settings()
    full_query = PREFIXES + query
    cache = _get_cache(settings)

    cache_key = f"{settings.sparql_endpoint}\n{full_query}"
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("SPARQL cache hit")
        return cached

    payload = _request(full_query, settings)
    rows = _bindings(payload)
    cache.set(cache_key, rows)
    return rows


def _request(query: str, settings: DBpediaSettings) -> dict[str, Any]:
    """GET the endpoint, retrying on the failures that are worth retrying."""
    last_error: Exception | None = None

    for attempt in range(settings.max_retries + 1):
        if attempt:
            # Exponential backoff. 429 is the common case on the public
            # endpoint, and retrying immediately just earns another one.
            time.sleep(2.0**attempt * 0.5)
        try:
            response = httpx.get(
                settings.sparql_endpoint,
                params={"query": query, "format": "application/sparql-results+json"},
                headers={
                    "Accept": "application/sparql-results+json",
                    "User-Agent": "gca-rag-mcp/0.1 (+https://dbpedia.org/sparql)",
                },
                timeout=settings.http_timeout,
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last_error = exc
            logger.warning("SPARQL transport error (attempt %d): %s", attempt + 1, exc)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            last_error = SPARQLError(f"endpoint returned {response.status_code}")
            logger.warning("SPARQL %d (attempt %d)", response.status_code, attempt + 1)
            continue
        if response.status_code != 200:
            # 400 means the query is wrong. Retrying an identical bad query is
            # pointless, so fail immediately with the endpoint's explanation.
            raise SPARQLError(f"endpoint returned {response.status_code}: {response.text[:300]}")

        try:
            return response.json()
        except ValueError as exc:
            raise SPARQLError(f"endpoint returned non-JSON: {response.text[:200]}") from exc

    raise SPARQLError(
        f"SPARQL query failed after {settings.max_retries + 1} attempts: {last_error}"
    )


def _bindings(payload: dict[str, Any]) -> list[dict[str, str]]:
    try:
        raw = payload["results"]["bindings"]
    except (KeyError, TypeError) as exc:
        raise SPARQLError(f"unexpected results shape: {str(payload)[:200]}") from exc
    return [{var: cell["value"] for var, cell in row.items() if "value" in cell} for row in raw]


def reset_cache() -> None:
    """Drop the module-level cache. For tests, and for a long-lived server that wants a reset."""
    global _cache
    if _cache is not None:
        _cache.clear()
    _cache = None
