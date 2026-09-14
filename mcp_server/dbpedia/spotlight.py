"""
The DBpedia Spotlight transport.

Spotlight is the only service here that reads *context* rather than a bare
string, which is what makes it the right first move for an entity that appears
in a sentence: "Jordan" in a basketball paragraph and "Jordan" in a Middle East
paragraph get different candidates.

Two things shape this module:

- **The public instance is unreliable.** `api.dbpedia-spotlight.org` was
  unreachable while this was written. Failures raise `SpotlightError`, which
  `tools.py` turns into an empty result with a `note` — the grounder must be
  able to fall back to `search_resource` rather than stall.
- **Two endpoints, two JSON shapes.** `/candidates` returns ranked alternatives
  per surface form, which is what disambiguation wants; `/annotate` returns one
  committed link each. `/candidates` is tried first and `/annotate` is the
  fallback, so both shapes are normalized here to one row format.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from mcp_server.dbpedia.cache import TTLCache
from mcp_server.dbpedia.classes import DBR
from mcp_server.dbpedia.config import DBpediaSettings, get_settings

logger = logging.getLogger(__name__)

_cache: TTLCache | None = None


class SpotlightError(RuntimeError):
    """Spotlight could not be reached or returned something unusable."""


def _get_cache(settings: DBpediaSettings) -> TTLCache:
    global _cache
    if _cache is None:
        _cache = TTLCache(
            ttl_seconds=settings.cache_ttl_seconds,
            max_entries=settings.cache_max_entries,
        )
    return _cache


def candidates(
    text: str,
    *,
    confidence: float = 0.3,
    support: int = 0,
    settings: DBpediaSettings | None = None,
) -> list[dict[str, Any]]:
    """
    Annotate `text` and return candidate links.

    Rows are `{surface_form, offset, uri, label, final_score, contextual_score,
    support, types}`, ordered as Spotlight ranked them within each surface form.

    Raises
    ------
    SpotlightError
        If neither `/candidates` nor `/annotate` could be reached.
    """
    settings = settings or get_settings()
    cache = _get_cache(settings)
    cache_key = f"{settings.spotlight_endpoint}|{confidence}|{support}|{text}"

    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Spotlight cache hit")
        return cached

    base = settings.spotlight_endpoint.rstrip("/")
    params = {"text": text, "confidence": confidence, "support": support}

    errors: list[str] = []
    for path, parse in (("/candidates", _parse_candidates), ("/annotate", _parse_annotate)):
        try:
            payload = _request(f"{base}{path}", params, settings)
        except SpotlightError as exc:
            errors.append(f"{path}: {exc}")
            continue
        rows = parse(payload)
        cache.set(cache_key, rows)
        return rows

    raise SpotlightError("; ".join(errors) or "no Spotlight endpoint responded")


def _request(url: str, params: dict[str, Any], settings: DBpediaSettings) -> dict[str, Any]:
    try:
        response = httpx.get(
            url,
            params=params,
            headers={"Accept": "application/json"},
            timeout=settings.http_timeout,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise SpotlightError(str(exc)) from exc

    if response.status_code != 200:
        raise SpotlightError(f"returned {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise SpotlightError(f"non-JSON response: {response.text[:200]}") from exc


# ── Response shaping ──────────────────────────────────────────────────
#
# Spotlight's JSON is a direct transliteration of its XML, so keys carry `@`
# prefixes and any element that *could* repeat is a list only when it actually
# does. `_as_list` is what keeps the single-match case from crashing.


def _parse_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    annotation = payload.get("annotation") or {}
    rows: list[dict[str, Any]] = []
    for surface in _as_list(annotation.get("surfaceForm")):
        name = surface.get("@name", "")
        offset = _as_int(surface.get("@offset"))
        for resource in _as_list(surface.get("resource")):
            rows.append(
                {
                    "surface_form": name,
                    "offset": offset,
                    "uri": _absolute(resource.get("@uri", "")),
                    "label": resource.get("@label") or resource.get("@uri", ""),
                    "final_score": _as_float(resource.get("@finalScore")),
                    "contextual_score": _as_float(resource.get("@contextualScore")),
                    "support": _as_int(resource.get("@support")),
                    "types": _types(resource.get("@types")),
                }
            )
    return rows


def _parse_annotate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resource in _as_list(payload.get("Resources")):
        uri = _absolute(resource.get("@URI", ""))
        rows.append(
            {
                "surface_form": resource.get("@surfaceForm", ""),
                "offset": _as_int(resource.get("@offset")),
                "uri": uri,
                "label": uri.rsplit("/", 1)[-1].replace("_", " "),
                # /annotate reports one similarity number, not the pair
                # /candidates gives; it is the closest thing to a final score.
                "final_score": _as_float(resource.get("@similarityScore")),
                "contextual_score": _as_float(resource.get("@similarityScore")),
                "support": _as_int(resource.get("@support")),
                "types": _types(resource.get("@types")),
            }
        )
    return rows


def _absolute(uri: str) -> str:
    """`/candidates` reports bare resource names; `/annotate` reports full URIs."""
    text = (uri or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    return f"{DBR}{text}"


def _types(raw: Any) -> list[str]:
    """Spotlight packs types into one comma-separated string, mixing vocabularies."""
    if not raw:
        return []
    return [t.strip() for t in str(raw).split(",") if t.strip()]


def _as_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    return [value] if isinstance(value, dict) else []


def _as_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def reset_cache() -> None:
    """Drop the module-level cache. For tests."""
    global _cache
    if _cache is not None:
        _cache.clear()
    _cache = None
