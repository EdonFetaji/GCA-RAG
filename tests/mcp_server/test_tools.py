"""
The eight tools, exercised over a real MCP session against stubbed transports.

Everything goes through `MCPClient(build_server())` rather than by calling the
tool functions directly. That is the point: it checks the tools are actually
registered, their schemas serialise, and their payloads survive the protocol
round-trip — three things a direct call would not catch.
"""

from __future__ import annotations

import httpx
import pytest

from mcp_server.client import MCPClient
from mcp_server.dbpedia.classes import DBO, DBR
from mcp_server.server import build_server
from tests.conftest import StubResponse, lookup_docs, sparql_results

ALL_TOOL_NAMES = {
    "spotlight_link",
    "search_resource",
    "get_resource_profile",
    "search_class",
    "find_object_properties",
    "find_datatype_properties",
    "get_property_profile",
    "get_predicates_between",
}


@pytest.fixture
def client():
    return MCPClient(build_server())


def _is(url: str, service: str) -> bool:
    return service in url


# ── Registration ──────────────────────────────────────────────────────


async def test_publishes_exactly_the_eight_tools(client):
    async with client:
        tools = await client.list_tools()

    names = {tool.name for tool in tools}
    assert names == ALL_TOOL_NAMES
    # The scaffold's placeholders must be gone, not merely outnumbered.
    assert "echo" not in names and "add" not in names


async def test_every_tool_advertises_a_description_and_schema(client):
    async with client:
        tools = await client.list_tools()

    for tool in tools:
        assert tool.description, f"{tool.name} has no description for the model to read"
        assert tool.inputSchema.get("properties"), f"{tool.name} advertises no parameters"


# ── Entity tools ──────────────────────────────────────────────────────


async def test_search_resource_filters_by_type_and_strips_markup(client, stub_http):
    stub_http(
        lambda url, params: (
            lookup_docs(
                {
                    "resource": f"{DBR}Seattle",
                    "label": "<B>Seattle</B>",
                    "comment": "<B>Seattle</B> is a seaport city.",
                    "type": [f"{DBO}City", f"{DBO}Place"],
                    "refCount": "1778",
                    "score": "32542.6",
                }
            )
            if _is(url, "lookup")
            else sparql_results()
        )
    )

    async with client:
        result = await client.call_tool_json(
            "search_resource", {"label": "Seattle", "expected_types": ["dbo:City"]}
        )

    assert result["expected_types"] == [f"{DBO}City"]
    hit = result["results"][0]
    # Lookup returns highlighted HTML; none of it may reach the model.
    assert hit["label"] == "Seattle"
    assert "<B>" not in hit["comment"]
    assert hit["ref_count"] == 1778


async def test_search_resource_retries_unfiltered_when_the_type_matches_nothing(client, stub_http):
    seen: list[dict] = []

    def router(url, params):
        if not _is(url, "lookup"):
            return sparql_results()
        seen.append(params)
        if "typeName" in params:
            return lookup_docs()
        return lookup_docs({"resource": f"{DBR}Zephyr_Inc", "label": "Zephyr Inc"})

    stub_http(router)
    async with client:
        result = await client.call_tool_json(
            "search_resource", {"label": "Zephyr Inc", "expected_types": ["dbo:Company"]}
        )

    assert [hit["uri"] for hit in result["results"]] == [f"{DBR}Zephyr_Inc"]
    assert "searched unfiltered" in result["note"]
    assert any("typeName" in params for params in seen)


async def test_get_resource_profile_surfaces_redirect_and_disambiguation(client, stub_http):
    stub_http(
        lambda url, params: (
            sparql_results(
                {"p": f"{DBO}wikiPageRedirects", "o": f"{DBR}Seattle"},
                {"p": f"{DBO}wikiPageDisambiguates", "o": f"{DBR}Seattle_(disambiguation)"},
            )
            if _is(url, "sparql")
            else lookup_docs()
        )
    )

    async with client:
        result = await client.call_tool_json("get_resource_profile", {"uri": f"{DBR}Seattle,_WA"})

    assert result["found"] is True
    assert result["redirects_to"] == f"{DBR}Seattle"
    assert result["is_disambiguation"] is True


async def test_get_resource_profile_reports_a_uri_with_no_triples(client, stub_http):
    stub_http(lambda url, params: sparql_results())

    async with client:
        result = await client.call_tool_json("get_resource_profile", {"uri": f"{DBR}Not_Real_XYZ"})

    assert result["found"] is False
    assert "not a real resource" in result["note"]


async def test_search_class_ranks_the_exact_label_first(client, stub_http):
    stub_http(
        lambda url, params: sparql_results(
            {"uri": f"{DBO}CityDistrict", "label": "city district", "supers": f"{DBO}Place"},
            {"uri": f"{DBO}City", "label": "city", "supers": f"{DBO}Settlement"},
        )
    )

    async with client:
        result = await client.call_tool_json("search_class", {"label": "City"})

    assert [hit["uri"] for hit in result["results"]] == [f"{DBO}City", f"{DBO}CityDistrict"]
    assert result["results"][0]["superclasses"] == [f"{DBO}Settlement"]


# ── Relation tools ────────────────────────────────────────────────────


async def test_find_object_properties_normalises_the_relation_phrase(client, stub_http):
    queries: list[str] = []

    def router(url, params):
        queries.append(params.get("query", ""))
        return sparql_results(
            {
                "uri": f"{DBO}locatedInArea",
                "label": "located in area",
                "domain": f"{DBO}Place",
                "range": f"{DBO}Place",
            }
        )

    stub_http(router)
    async with client:
        result = await client.call_tool_json(
            "find_object_properties", {"relation_text": "LOCATED_IN", "object_types": ["dbo:Place"]}
        )

    # The enum spelling must become words, or no label can ever match it.
    assert any('"located in"' in query for query in queries)
    assert result["results"][0]["uri"] == f"{DBO}locatedInArea"
    assert result["results"][0]["kind"] == "object"


async def test_property_without_a_declared_domain_survives_the_type_filter(client, stub_http):
    """Most of DBpedia's ontology leaves domain/range undeclared; filtering those out
    would hide the best matches."""
    stub_http(
        lambda url, params: sparql_results(
            {"uri": f"{DBO}location", "label": "location", "range": f"{DBO}Place"},
            {"uri": f"{DBO}wrongRange", "label": "location of", "range": f"{DBO}Person"},
        )
    )

    async with client:
        result = await client.call_tool_json(
            "find_object_properties",
            {"relation_text": "location", "subject_types": ["dbo:Organisation"]},
        )

    uris = [hit["uri"] for hit in result["results"]]
    assert f"{DBO}location" in uris  # no domain declared → kept
    assert f"{DBO}wrongRange" in uris  # no domain declared either → also kept


async def test_find_object_properties_excludes_a_mismatched_range(client, stub_http):
    stub_http(
        lambda url, params: sparql_results(
            {"uri": f"{DBO}location", "label": "location", "range": f"{DBO}Place"},
            {"uri": f"{DBO}spouse", "label": "location partner", "range": f"{DBO}Person"},
        )
    )

    async with client:
        result = await client.call_tool_json(
            "find_object_properties",
            {"relation_text": "location", "object_types": ["dbo:Place"]},
        )

    assert [hit["uri"] for hit in result["results"]] == [f"{DBO}location"]


async def test_get_predicates_between_tags_direction_and_drops_noise(client, stub_http):
    stub_http(
        lambda url, params: sparql_results(
            {"p": f"{DBO}wikiPageWikiLink", "direction": "forward"},
            {"p": f"{DBO}subdivision", "direction": "forward"},
            {"p": "http://dbpedia.org/property/largestcity", "direction": "reverse"},
        )
    )

    async with client:
        result = await client.call_tool_json(
            "get_predicates_between",
            {"subject_uri": f"{DBR}Seattle", "object_uri": f"{DBR}Washington"},
        )

    predicates = result["predicates"]
    assert [p["predicate"] for p in predicates] == [
        f"{DBO}subdivision",  # curated dbo: ranks ahead of scraped dbp:
        "http://dbpedia.org/property/largestcity",
    ]
    assert predicates[1]["direction"] == "reverse"
    assert predicates[0]["label"] == "subdivision"


async def test_get_predicates_between_reports_an_absent_edge_as_data(client, stub_http):
    stub_http(lambda url, params: sparql_results())

    async with client:
        result = await client.call_tool_json(
            "get_predicates_between",
            {"subject_uri": f"{DBR}A", "object_uri": f"{DBR}B"},
        )

    assert result["predicates"] == []
    assert "no direct edge" in result["note"]


# ── Degradation ───────────────────────────────────────────────────────


async def test_an_unreachable_service_returns_a_note_not_an_error(client, stub_http):
    def dead(url, params):
        raise httpx.ConnectError("connection refused")

    stub_http(dead)
    async with client:
        result = await client.call_tool_json(
            "spotlight_link", {"mention": "Seattle", "context": "A city in Washington."}
        )

    # A raise here would stall the agent loop mid-graph; a note lets it reroute.
    assert result["candidates"] == []
    assert "unavailable" in result["note"]
    assert "search_resource" in result["note"]


async def test_a_rejected_uri_comes_back_as_a_note(client, stub_http):
    stub_http(lambda url, params: sparql_results())

    async with client:
        result = await client.call_tool_json(
            "get_predicates_between",
            {"subject_uri": 'evil> } UNION { ?s ?p "x', "object_uri": f"{DBR}B"},
        )

    assert result["predicates"] == []
    assert "rejected URI" in result["note"]


async def test_a_server_error_is_retried_then_reported(client, stub_http):
    attempts = {"n": 0}

    def flaky(url, params):
        attempts["n"] += 1
        return StubResponse("upstream busy", status_code=503)

    stub_http(flaky)
    async with client:
        result = await client.call_tool_json("search_class", {"label": "City"})

    assert attempts["n"] > 1, "a 503 should be retried, not given up on"
    assert "unavailable" in result["note"]
