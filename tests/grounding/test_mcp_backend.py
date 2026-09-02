"""
`MCPGroundingBackend` and the sync bridge underneath it.

These run over a real protocol session against an in-process server, so they
cover the thing that was previously broken end-to-end: the client actually
connecting, and a tool payload arriving back as a dict.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from kg_agentic_extraction.grounding.base import (
    GroundingBackend,
    NullGroundingBackend,
    mapping_from_hit,
)
from kg_agentic_extraction.grounding.mcp_backend import ALL_TOOLS, MCPGroundingBackend
from mcp_server.client import MCPClientError, SyncMCPClient
from mcp_server.dbpedia.classes import DBO, DBR
from mcp_server.server import build_server
from tests.conftest import lookup_docs, sparql_results


@pytest.fixture
def backend():
    return MCPGroundingBackend(url=build_server())


# ── The port ──────────────────────────────────────────────────────────


def test_both_backends_satisfy_the_port():
    assert isinstance(NullGroundingBackend(), GroundingBackend)
    assert isinstance(MCPGroundingBackend(url=build_server()), GroundingBackend)


def test_tool_specs_come_from_the_running_server(backend):
    with backend:
        specs = backend.tool_specs()

    assert {spec.name for spec in specs} == set(ALL_TOOLS)
    for spec in specs:
        # These are what the model reads and what it fills in. An empty one is
        # a tool the model cannot use correctly.
        assert spec.description
        assert spec.input_schema.get("properties")


def test_tool_specs_are_fetched_once(backend):
    with backend:
        first = backend.tool_specs()
        second = backend.tool_specs()
    assert first is second


# ── Session lifetime ──────────────────────────────────────────────────


def test_one_session_serves_many_calls(backend, stub_http):
    stub_http(lambda url, params: sparql_results({"uri": f"{DBO}City", "label": "city"}))

    with backend:
        for _ in range(3):
            assert backend.search_class("City")["results"]

    assert backend.call_count == 3


def test_nested_with_blocks_do_not_close_the_outer_session(backend, stub_http):
    stub_http(lambda url, params: sparql_results({"uri": f"{DBO}City", "label": "city"}))

    with backend:
        with backend:
            backend.search_class("City")
        # The inner block released its reference; the session must survive it.
        assert backend.search_class("City")["results"]


def test_calling_before_opening_is_a_clear_error(backend):
    with pytest.raises(Exception) as caught:
        backend.search_class("City")
    assert "not open" in str(caught.value)


def test_the_bridge_works_from_inside_a_running_event_loop(stub_http):
    """
    The failure mode the previous `asyncio.run()` bridge raised on.

    A notebook or an async server already owns the loop; the client must own its
    own thread rather than assume it can start one.
    """
    stub_http(lambda url, params: sparql_results({"uri": f"{DBO}City", "label": "city"}))
    backend = MCPGroundingBackend(url=build_server())

    async def drive() -> dict:
        with backend:
            return backend.search_class("City")

    result = asyncio.run(drive())
    assert result["results"][0]["uri"] == f"{DBO}City"


def test_sync_client_refuses_work_after_close():
    client = SyncMCPClient(build_server())
    client.open()
    client.close()
    with pytest.raises(MCPClientError):
        client.list_tools()


# ── Dispatch ──────────────────────────────────────────────────────────


def test_dispatch_returns_json_text_for_a_tool_message(backend, stub_http):
    stub_http(lambda url, params: sparql_results({"uri": f"{DBO}City", "label": "city"}))

    with backend:
        payload = backend.dispatch("search_class", {"label": "City"})

    assert json.loads(payload)["results"][0]["uri"] == f"{DBO}City"


def test_dispatch_reports_an_unknown_tool_instead_of_raising(backend):
    with backend:
        payload = json.loads(backend.dispatch("delete_everything", {}))

    # The model chose a name that does not exist. It needs to read that and
    # pick again, not have its loop torn down.
    assert "unknown tool" in payload["error"]
    assert set(payload["available"]) == set(ALL_TOOLS)


def test_dispatch_reports_a_dead_server_instead_of_raising():
    backend = MCPGroundingBackend(url="http://127.0.0.1:9/mcp/", timeout_seconds=1.0)
    payload = json.loads(backend.dispatch("search_class", {"label": "City"}))
    assert "error" in payload


def test_none_arguments_are_omitted_rather_than_sent(backend, stub_http):
    seen: list[dict] = []

    def router(url, params):
        seen.append(params)
        return lookup_docs({"resource": f"{DBR}Seattle", "label": "Seattle"})

    stub_http(router)
    with backend:
        backend.search_resource("Seattle", expected_types=None)

    assert seen and all("typeName" not in params for params in seen)


# ── The null backend ──────────────────────────────────────────────────


def test_null_backend_answers_every_tool_without_a_connection():
    null = NullGroundingBackend()
    with null:
        assert null.spotlight_link("x", "y")["candidates"] == []
        assert null.search_resource("x")["results"] == []
        assert null.get_resource_profile("x")["found"] is False
        assert null.search_class("x")["results"] == []
        assert null.find_object_properties("x")["results"] == []
        assert null.find_datatype_properties("x")["results"] == []
        assert null.get_property_profile("x")["found"] is False
        assert null.get_predicates_between("a", "b")["predicates"] == []
        # No tools means the agent falls back to a plain call, not that it fails.
        assert null.tool_specs() == []
        assert "error" in json.loads(null.dispatch("search_class", {}))


# ── Helpers ───────────────────────────────────────────────────────────


def test_mapping_from_hit_keeps_the_first_type_as_the_class():
    mapping = mapping_from_hit(
        {"uri": f"{DBR}Seattle", "label": "Seattle", "types": [f"{DBO}City"], "comment": "A city."},
        confidence=0.7,
    )
    assert mapping is not None
    assert mapping.uri == f"{DBR}Seattle"
    assert mapping.ontology_class == f"{DBO}City"
    assert mapping.confidence == 0.7


def test_mapping_from_hit_drops_a_row_with_no_uri():
    assert mapping_from_hit({"label": "Seattle"}) is None
