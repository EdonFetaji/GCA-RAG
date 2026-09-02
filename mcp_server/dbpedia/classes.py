"""
Class- and property-name normalization.

Tools take type arguments from a language model, which will write the same class
four different ways — `Person`, `dbo:Person`, `http://dbpedia.org/ontology/Person`,
`PERSON`. Rather than make the model guess the one spelling the server wants,
every form is accepted here and canonicalised to a full URI.

This module knows nothing about the pipeline's local ontology. The mapping from
`EntityType.PERSON` to `dbo:Person` lives on the pipeline side, in
`kg_agentic_extraction/grounding/hints.py`, because that is the side that owns
the local enums — the server stays DBpedia-only and is reusable by any client.
"""

from __future__ import annotations

DBO = "http://dbpedia.org/ontology/"
DBR = "http://dbpedia.org/resource/"
DBP = "http://dbpedia.org/property/"
RDFS = "http://www.w3.org/2000/01/rdf-schema#"
OWL = "http://www.w3.org/2002/07/owl#"
XSD = "http://www.w3.org/2001/XMLSchema#"
SCHEMA = "http://schema.org/"
FOAF = "http://xmlns.com/foaf/0.1/"

_PREFIXES = {
    "dbo": DBO,
    "dbpedia-owl": DBO,
    "dbr": DBR,
    "dbp": DBP,
    "rdfs": RDFS,
    "owl": OWL,
    "xsd": XSD,
    "schema": SCHEMA,
    "foaf": FOAF,
}

#: Spellings that would otherwise resolve to a class DBpedia does not have.
#: Kept to genuine spelling divergences only — DBpedia's ontology is
#: British-spelled and models reliably write American. Semantic guesses
#: (LOCATION → Place and friends) belong in the pipeline's hint table, not
#: here, so that a wrong guess is visible in a prompt rather than buried in a
#: silent rewrite on the server.
_ALIASES = {
    "organization": "Organisation",
    "organizations": "Organisation",
    "sportsorganization": "SportsOrganisation",
    "governmentorganization": "GovernmentAgency",
    "educationalorganization": "EducationalInstitution",
    "militaryorganization": "MilitaryUnit",
}


def normalize_class(value: str) -> str | None:
    """
    Canonicalise a class name to a full URI.

    Accepts a full URI, a CURIE (`dbo:Person`), or a bare local name in any
    casing (`Person`, `PERSON`, `person`). Returns None for empty input, so
    callers can pass a model's `expected_types` straight through without
    filtering blanks out first.
    """
    text = (value or "").strip()
    if not text:
        return None

    if text.startswith(("http://", "https://")):
        return text

    if ":" in text:
        prefix, _, local = text.partition(":")
        base = _PREFIXES.get(prefix.lower())
        if base and local:
            return f"{base}{local}"
        return None

    return f"{DBO}{_local_name(text)}"


def normalize_classes(values: list[str] | None) -> list[str]:
    """Normalize a list, dropping blanks and unrecognised prefixes, preserving order."""
    if not values:
        return []
    seen: dict[str, None] = {}
    for value in values:
        uri = normalize_class(value)
        if uri:
            seen.setdefault(uri, None)
    return list(seen)


def normalize_datatype(value: str | None) -> str | None:
    """Canonicalise a literal datatype. A bare name is assumed to be `xsd:`."""
    text = (value or "").strip()
    if not text:
        return None
    if text.startswith(("http://", "https://")) or ":" in text:
        return normalize_class(text)
    return f"{XSD}{text[0].lower()}{text[1:]}"


def short_name(uri: str) -> str:
    """
    The local part of a URI — what the DBpedia Lookup API wants for `typeName`.

    `http://dbpedia.org/ontology/City` → `City`
    """
    for separator in ("#", "/"):
        if separator in uri:
            uri = uri.rsplit(separator, 1)[-1]
    return uri


def _local_name(text: str) -> str:
    """
    Best-effort bare name → DBpedia class name.

    ALL-CAPS input (which is how the pipeline's local enums are spelled) is
    title-cased, then run through the alias table so `ORGANIZATION` lands on
    `Organisation` rather than on a class that does not exist.
    """
    alias = _ALIASES.get(text.replace("_", "").replace(" ", "").lower())
    if alias:
        return alias
    if text.isupper():
        return text.capitalize()
    return text[0].upper() + text[1:]
