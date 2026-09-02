"""
Open-vocabulary types on the knowledge-graph contract.

The enum used to be the guardrail. With it gone from extraction, the only thing
still enforced on a type is its *spelling* — which matters more than it sounds,
because `Relation.key` is built from it and the grounder de-duplicates on it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kg_agentic_extraction.models.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    normalize_type,
)
from kg_agentic_extraction.models.ontology import EntityType, RelationType


def test_types_outside_the_enum_are_accepted():
    """The whole point: the extractor may name a type the ontology never had."""
    entity = Entity(id="e1", name="Rick Scott", type="POLITICIAN")
    relation = Relation(source="e1", target="e2", relation_type="GOVERNOR_OF")
    assert entity.type == "POLITICIAN"
    assert relation.relation_type == "GOVERNOR_OF"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("located in", "LOCATED_IN"),
        ("Located-In", "LOCATED_IN"),
        ("  located_in  ", "LOCATED_IN"),
        ("governor of", "GOVERNOR_OF"),
        ("PERSON", "PERSON"),
        ("co-founder/CEO", "CO_FOUNDER_CEO"),
    ],
)
def test_spelling_is_normalized(raw, expected):
    assert normalize_type(raw) == expected
    assert Entity(id="e", name="n", type=raw).type == expected


def test_normalization_collapses_variants_to_one_relation_key():
    """
    Three spellings of one relation must produce one key, or de-duplication
    downstream silently treats them as three different edges.
    """
    keys = {
        Relation(source="a", target="b", relation_type=spelling).key
        for spelling in ("GOVERNOR_OF", "governor of", "Governor-Of")
    }
    assert keys == {"a|GOVERNOR_OF|b"}


def test_enum_members_still_validate():
    """
    The enums did not go away — the grounder still uses them, and older code
    and fixtures pass them directly. `StrEnum` means they arrive as their value.
    """
    entity = Entity(id="e1", name="Seattle", type=EntityType.LOCATION)
    relation = Relation(source="e1", target="e1", relation_type=RelationType.LOCATED_IN)
    assert entity.type == "LOCATION"
    assert relation.relation_type == "LOCATED_IN"


def test_an_empty_type_is_still_rejected():
    """Free-form is not the same as optional."""
    with pytest.raises(ValidationError):
        Entity(id="e1", name="Seattle", type="")
    with pytest.raises(ValidationError):
        Entity(id="e1", name="Seattle", type="   ")


def test_vocabulary_properties_report_what_the_graph_uses():
    graph = KnowledgeGraph(
        entities=[
            Entity(id="e1", name="Rick Scott", type="POLITICIAN"),
            Entity(id="e2", name="Florida", type="LOCATION"),
            Entity(id="e3", name="Scott Walker", type="politician"),
        ],
        relations=[
            Relation(source="e1", target="e2", relation_type="GOVERNOR_OF"),
            Relation(source="e3", target="e2", relation_type="governor of"),
        ],
    )
    # Sorted, de-duplicated, and normalization has already merged the variants.
    assert graph.entity_type_vocabulary == ["LOCATION", "POLITICIAN"]
    assert graph.relation_type_vocabulary == ["GOVERNOR_OF"]
