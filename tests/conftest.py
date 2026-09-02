"""
Shared fixtures.

The offline suite never touches the network. `stub_http` replaces `httpx.get`
with a router over canned payloads, which is enough because all three DBpedia
transports go through that one function — a design constraint worth keeping.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from mcp_server.dbpedia import lookup, sparql, spotlight


class StubResponse:
    """The parts of `httpx.Response` the transports actually read."""

    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> Any:
        if isinstance(self._payload, str):
            raise ValueError("not JSON")
        return self._payload


@pytest.fixture(autouse=True)
def _clear_caches():
    """
    Drop every transport cache around each test.

    Not optional: the caches are module-level and TTL'd for fifteen minutes, so
    without this a payload stubbed by one test would answer the next one's call.
    """
    for module in (sparql, lookup, spotlight):
        module.reset_cache()
    yield
    for module in (sparql, lookup, spotlight):
        module.reset_cache()


@pytest.fixture
def stub_http(monkeypatch) -> Callable[[Callable[[str, dict], Any]], None]:
    """
    Install a router in place of `httpx.get`.

    The router is called with `(url, params)` and returns the payload to answer
    with, a `StubResponse` for full control, or raises `httpx.ConnectError` to
    simulate an unreachable service.
    """

    def install(router: Callable[[str, dict], Any]) -> None:
        def fake_get(url: str, **kwargs: Any) -> StubResponse:
            result = router(url, kwargs.get("params") or {})
            return result if isinstance(result, StubResponse) else StubResponse(result)

        monkeypatch.setattr(httpx, "get", fake_get)

    return install


# ── Canned payloads ───────────────────────────────────────────────────


def sparql_results(*rows: dict[str, str]) -> dict[str, Any]:
    """A SPARQL JSON results document with these `{var: value}` rows."""
    variables = sorted({key for row in rows for key in row})
    return {
        "head": {"vars": variables},
        "results": {
            "bindings": [
                {k: {"type": "literal", "value": v} for k, v in row.items()} for row in rows
            ]
        },
    }


def lookup_docs(*docs: dict[str, Any]) -> dict[str, Any]:
    """A Lookup response. Every field is list-wrapped, as the real API does."""
    return {
        "docs": [
            {key: value if isinstance(value, list) else [value] for key, value in doc.items()}
            for doc in docs
        ]
    }
