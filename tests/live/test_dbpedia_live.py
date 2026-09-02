"""
The tools against the real DBpedia, deselected by default.

    uv run pytest -m live

These exist because a stubbed transport cannot tell you a SPARQL query is
*valid* — it will happily return the canned rows for a query Virtuoso would
reject. Run them after changing any query in `tools.py`.

They are also the early warning for DBpedia changing under us. Two things
already found this way and worked around in `tools.py`: the public snapshot
serves no `dbo:abstract` for resources (the profile falls back to Lookup), and
Spotlight's public instance is often unreachable (`spotlight_link` degrades to a
note). Assertions here are about shape and reachability, never about a
particular resource's facts, which do change.
"""

from __future__ import annotations

import pytest

from mcp_server.client import MCPClient
from mcp_server.dbpedia.classes import DBO, DBR
from mcp_server.server import build_server

pytestmark = pytest.mark.live

SEATTLE = f"{DBR}Seattle"
WASHINGTON = f"{DBR}Washington_(state)"


@pytest.fixture
def client():
    return MCPClient(build_server())


async def test_search_class_finds_the_city_class(client):
    async with client:
        result = await client.call_tool_json("search_class", {"label": "City"})

    assert result["note"] == ""
    assert f"{DBO}City" in {hit["uri"] for hit in result["results"]}


async def test_search_resource_finds_seattle_under_its_class(client):
    async with client:
        result = await client.call_tool_json(
            "search_resource", {"label": "Seattle", "expected_types": ["dbo:City"]}
        )

    assert SEATTLE in {hit["uri"] for hit in result["results"]}


async def test_resource_profile_carries_an_abstract_and_types(client):
    async with client:
        result = await client.call_tool_json("get_resource_profile", {"uri": SEATTLE})

    assert result["found"] is True
    assert result["types"], "a well-known resource must report dbo: types"
    # Comes from Lookup, not SPARQL — see the module docstring.
    assert len(result["abstract"]) > 50
    assert result["is_disambiguation"] is False


async def test_a_disambiguation_page_is_flagged(client):
    async with client:
        result = await client.call_tool_json(
            "get_resource_profile", {"uri": f"{DBR}Jordan_(disambiguation)"}
        )

    assert result["found"] is True
    assert result["is_disambiguation"] is True


async def test_a_redirect_names_its_target(client):
    """Not every `(disambiguation)` URI is one — this one is a plain redirect,
    and the profile has to say which it is."""
    async with client:
        result = await client.call_tool_json(
            "get_resource_profile", {"uri": f"{DBR}Mercury_(disambiguation)"}
        )

    assert result["found"] is True
    assert result["redirects_to"].startswith(DBR)
    assert result["is_disambiguation"] is False


async def test_object_property_search_reaches_location_from_the_enum_spelling(client):
    """The stemming path: `LOCATED_IN` must reach `dbo:location`, whose label
    shares only the stem `locat`."""
    async with client:
        result = await client.call_tool_json(
            "find_object_properties",
            {"relation_text": "LOCATED_IN", "object_types": ["dbo:Place"]},
        )

    uris = {hit["uri"] for hit in result["results"]}
    assert f"{DBO}location" in uris or f"{DBO}locatedInArea" in uris


async def test_datatype_property_search_respects_the_literal_range(client):
    async with client:
        result = await client.call_tool_json(
            "find_datatype_properties",
            {"relation_text": "founding", "literal_datatype": "date"},
        )

    assert result["results"]
    assert all(
        hit["range"] == "http://www.w3.org/2001/XMLSchema#date" or not hit["range"]
        for hit in result["results"]
    )


async def test_property_profile_reports_range_and_usage(client):
    async with client:
        result = await client.call_tool_json(
            "get_property_profile", {"property_uri": f"{DBO}location"}
        )

    assert result["found"] is True
    assert result["kind"] == "object"
    assert result["range"] == f"{DBO}Place"
    assert result["usage_count"] > 0


async def test_predicates_between_two_linked_resources(client):
    async with client:
        result = await client.call_tool_json(
            "get_predicates_between", {"subject_uri": SEATTLE, "object_uri": WASHINGTON}
        )

    assert result["predicates"], "Seattle and Washington are linked in DBpedia"
    assert {p["direction"] for p in result["predicates"]} <= {"forward", "reverse"}


async def test_predicates_between_unrelated_resources_is_empty_not_an_error(client):
    async with client:
        result = await client.call_tool_json(
            "get_predicates_between",
            {"subject_uri": SEATTLE, "object_uri": f"{DBR}Photosynthesis"},
        )

    assert result["predicates"] == []
    assert "no direct edge" in result["note"]


async def test_spotlight_either_links_or_says_why_not(client):
    """The public Spotlight instance is frequently down. Either outcome is
    acceptable; a raise, or a silent empty with no note, is not."""
    async with client:
        result = await client.call_tool_json(
            "spotlight_link",
            {
                "mention": "Seattle",
                "context": "Amazon and Microsoft both have large offices near Seattle.",
            },
        )

    if result["candidates"]:
        assert result["note"] == ""
        assert all(c["uri"].startswith("http://dbpedia.org/") for c in result["candidates"])
    else:
        assert "unavailable" in result["note"] or "no link" in result["note"]
