"""
Starting guesses from the local ontology to DBpedia's.

The pipeline's `EntityType.ORGANIZATION` is not `dbo:Organization` — DBpedia
spells it `Organisation` — and `LOCATED_IN` is not a property name at all.
Making the model discover that from scratch for every graph costs a `search_class`
call per entity type and gets it wrong about as often as not.

These are **hints, not mappings**. They are rendered into the grounder's prompt
as first guesses to try, explicitly labelled as unverified, and the model is
told to fall back to `search_class` when a hint does not fit. Keeping them here
rather than on the server is deliberate: the server is DBpedia-only and knows
nothing about these enums, and a wrong guess is easier to notice in a prompt
than inside a silent server-side rewrite.

Since extraction went open-vocabulary the lookup is keyed by the types a graph
*actually contains*, not by the enum. Most graphs now arrive carrying types the
enum never had — `GOVERNOR_OF`, `CANDIDATE_FOR` — and those legitimately have no
hint. An empty list is the honest answer and already means "search for it
yourself" in the grounder's prompt, so an unknown type needs no special case.
"""

from __future__ import annotations

from kg_agentic_extraction.models.ontology import EntityType, RelationType

DBO = "http://dbpedia.org/ontology/"

#: Local entity type → DBpedia classes worth passing as `expected_types`.
#: Ordered most specific first; several are genuinely ambiguous, which is why
#: more than one is offered rather than one being picked here.
ENTITY_TYPE_HINTS: dict[EntityType, list[str]] = {
    EntityType.PERSON: ["dbo:Person"],
    EntityType.ORGANIZATION: ["dbo:Organisation", "dbo:Company"],
    EntityType.LOCATION: ["dbo:Place", "dbo:Settlement", "dbo:Country"],
    EntityType.EVENT: ["dbo:Event", "dbo:SocietalEvent"],
    EntityType.CONCEPT: ["dbo:TopicalConcept"],
    EntityType.DATE: [],  # A date is a literal, not a resource. Never grounded to one.
    EntityType.PRODUCT: ["dbo:Work", "dbo:Software", "dbo:Device"],
    EntityType.OTHER: [],
}

#: Local relation type → DBpedia properties worth checking first. Every one of
#: these still has to be confirmed with `get_property_profile` or
#: `get_predicates_between`; several local types have no good DBpedia
#: equivalent at all, and an empty list says so honestly.
RELATION_TYPE_HINTS: dict[RelationType, list[str]] = {
    RelationType.LOCATED_IN: ["dbo:location", "dbo:locatedInArea", "dbo:country"],
    RelationType.AFFILIATED_WITH: ["dbo:affiliation", "dbo:employer", "dbo:team"],
    RelationType.ANNOUNCED: [],
    RelationType.ACQUIRED: ["dbo:owningCompany", "dbo:parentCompany"],
    RelationType.PARTICIPATED_IN: ["dbo:participant", "dbo:event"],
    RelationType.RESULTED_IN: ["dbo:result"],
    RelationType.CAUSES: ["dbo:cause"],
    RelationType.CONTRADICTS: [],
    RelationType.SUPPORTS: [],
    RelationType.RELATED_TO: ["dbo:related"],
}


#: The tables above, re-keyed by plain string for lookup by a type name that may
#: not be an enum member at all.
_ENTITY_BY_NAME: dict[str, list[str]] = {k.value: v for k, v in ENTITY_TYPE_HINTS.items()}
_RELATION_BY_NAME: dict[str, list[str]] = {k.value: v for k, v in RELATION_TYPE_HINTS.items()}


def entity_hints(types: list[str]) -> dict[str, list[str]]:
    """
    Hint table for the entity types present in a graph, keyed by the local name.

    Types with no known DBpedia counterpart map to an empty list rather than
    being dropped: the grounder still has to ground those entities, and seeing
    the type with no hint tells it to search rather than leaving it to wonder
    whether the type was withheld.
    """
    return {t: _ENTITY_BY_NAME.get(t, []) for t in types}


def relation_hints(types: list[str]) -> dict[str, list[str]]:
    """Hint table for the relation types present in a graph, keyed by the local name."""
    return {t: _RELATION_BY_NAME.get(t, []) for t in types}
