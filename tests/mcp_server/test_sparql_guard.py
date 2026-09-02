"""
The SPARQL escaping guards.

`get_predicates_between` interpolates two URIs a language model produced, and
`search_class` interpolates a label. These are the checks standing between that
and a query the caller did not write.
"""

from __future__ import annotations

import pytest

from mcp_server.dbpedia.classes import DBO, XSD, normalize_class, normalize_datatype, short_name
from mcp_server.dbpedia.sparql import text_literal, uri_term


class TestUriTerm:
    def test_wraps_a_well_formed_uri(self):
        assert (
            uri_term("http://dbpedia.org/resource/Seattle")
            == "<http://dbpedia.org/resource/Seattle>"
        )

    def test_trims_surrounding_whitespace(self):
        assert uri_term("  https://dbpedia.org/resource/A  ") == "<https://dbpedia.org/resource/A>"

    @pytest.mark.parametrize(
        "hostile",
        [
            "http://dbpedia.org/resource/A> } UNION { ?s ?p ?o . #",  # closes the term
            'http://dbpedia.org/resource/A"',
            "http://dbpedia.org/resource/A B",  # a space also terminates it
            "http://dbpedia.org/resource/{A}",
            "http://dbpedia.org/resource/A\nSELECT",
            "http://dbpedia.org/resource/A|B",
            "http://dbpedia.org/resource/A^B",
            "http://dbpedia.org/resource/A\\B",
        ],
    )
    def test_rejects_anything_that_could_escape_the_term(self, hostile):
        with pytest.raises(ValueError):
            uri_term(hostile)

    @pytest.mark.parametrize(
        "not_a_uri",
        ["", "   ", "dbr:Seattle", "Seattle", "file:///etc/passwd", "javascript:alert(1)", "//x"],
    )
    def test_rejects_anything_that_is_not_an_absolute_http_uri(self, not_a_uri):
        with pytest.raises(ValueError):
            uri_term(not_a_uri)


class TestTextLiteral:
    def test_quotes_a_plain_string(self):
        assert text_literal("seattle") == '"seattle"'

    def test_escapes_the_closing_quote(self):
        assert text_literal('a" . ?s ?p ?o . #') == '"a\\" . ?s ?p ?o . #"'

    def test_escapes_backslashes_before_quotes(self):
        # Order matters: escaping the quote first would leave the backslash
        # free to escape the escape.
        assert text_literal('a\\"b') == '"a\\\\\\"b"'

    def test_escapes_newlines(self):
        assert "\n" not in text_literal("line\nbreak")


class TestClassNormalisation:
    @pytest.mark.parametrize(
        "given",
        ["Person", "person", "PERSON", "dbo:Person", "dbpedia-owl:Person", f"{DBO}Person"],
    )
    def test_every_spelling_lands_on_the_same_uri(self, given):
        assert normalize_class(given) == f"{DBO}Person"

    def test_american_spelling_reaches_the_british_class(self):
        # DBpedia has dbo:Organisation and no dbo:Organization. A model that
        # writes the American spelling must still get a real class.
        assert normalize_class("ORGANIZATION") == f"{DBO}Organisation"

    def test_an_unknown_prefix_is_rejected_rather_than_guessed(self):
        assert normalize_class("nosuch:Thing") is None

    def test_blank_input_is_not_an_error(self):
        assert normalize_class("") is None
        assert normalize_class("   ") is None

    def test_datatypes_default_to_xsd(self):
        assert normalize_datatype("date") == f"{XSD}date"
        assert normalize_datatype("xsd:integer") == f"{XSD}integer"
        assert normalize_datatype(None) is None

    def test_short_name_is_what_lookup_wants(self):
        assert short_name(f"{DBO}City") == "City"
        assert short_name("http://www.w3.org/2002/07/owl#Class") == "Class"
