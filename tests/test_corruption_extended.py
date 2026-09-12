import random

import pytest

from kg_dataset.corruption import (
    EXTENDED_CORRUPTION_TYPES,
    HEAD_NAMES_EXTENDED,
    corrupt_contradictions,
    corrupt_hallucinated_relations,
    generate_corrupted_variants,
    label_for_extended,
)


def _toy_kg():
    return {
        "entities": [
            {"id": "e1", "name": "A", "type": "PERSON", "document_frequency": 2, "confidence": 0.9, "evidence": []},
            {"id": "e2", "name": "B", "type": "LOCATION", "document_frequency": 1, "confidence": 0.9, "evidence": []},
            {"id": "e3", "name": "C", "type": "LOCATION", "document_frequency": 1, "confidence": 0.9, "evidence": []},
            {"id": "e4", "name": "D", "type": "ORGANIZATION", "document_frequency": 1, "confidence": 0.9, "evidence": []},
        ],
        "relations": [
            {"source": "e1", "target": "e2", "relation_type": "LOCATED_IN", "support_count": 2, "source_documents": [0], "confidence": 0.9, "evidence": []},
            {"source": "e1", "target": "e4", "relation_type": "AFFILIATED_WITH", "support_count": 1, "source_documents": [0], "confidence": 0.8, "evidence": []},
        ],
    }


def test_hallucinated_relations_only_adds_new_pairs():
    kg = _toy_kg()
    rng = random.Random(0)
    corrupted, meta = corrupt_hallucinated_relations(kg, severity=0.5, rng=rng)
    assert meta["corruption_type"] == "hallucinated_relations"
    assert len(corrupted["relations"]) > len(kg["relations"])
    original_pairs = {(r["source"], r["target"]) for r in kg["relations"]}
    new_relations = corrupted["relations"][len(kg["relations"]):]
    for r in new_relations:
        assert (r["source"], r["target"]) not in original_pairs
        assert r["confidence"] == 0.0  # fabricated -> zero confidence, matches orphan_node_injection's convention


def test_contradictions_violates_functional_relation():
    kg = _toy_kg()
    rng = random.Random(0)
    corrupted, meta = corrupt_contradictions(kg, severity=1.0, rng=rng)
    assert "fallback" not in meta  # toy KG has LOCATED_IN/AFFILIATED_WITH, so no fallback needed
    # e1 should now have >1 target for at least one functional relation type.
    by_source_type: dict[tuple, set] = {}
    for r in corrupted["relations"]:
        by_source_type.setdefault((r["source"], r["relation_type"]), set()).add(r["target"])
    assert any(len(targets) > 1 for targets in by_source_type.values())


def test_contradictions_falls_back_without_functional_relations():
    kg = {
        "entities": [{"id": "e1", "name": "A", "type": "CONCEPT"}, {"id": "e2", "name": "B", "type": "CONCEPT"}],
        "relations": [{"source": "e1", "target": "e2", "relation_type": "RELATED_TO", "support_count": 1, "confidence": 0.9, "evidence": []}],
    }
    rng = random.Random(0)
    _, meta = corrupt_contradictions(kg, severity=1.0, rng=rng)
    assert meta.get("fallback") == "no_functional_relations"


def test_label_for_extended_one_hot():
    for ctype in EXTENDED_CORRUPTION_TYPES:
        label = label_for_extended(ctype)
        assert set(label) == set(HEAD_NAMES_EXTENDED)
        assert label[ctype] == 1.0
        assert sum(label.values()) == 1.0
        assert label["consistency"] == 0.0


def test_label_for_extended_clean():
    label = label_for_extended(None)
    assert label["consistency"] == 1.0
    assert sum(v for k, v in label.items() if k != "consistency") == 0.0


def test_label_for_extended_rejects_unmapped_type():
    with pytest.raises(ValueError):
        label_for_extended("entity_duplication")


def test_generate_corrupted_variants_extended_scheme():
    kg = _toy_kg()
    variants = generate_corrupted_variants(
        kg, corruption_types=EXTENDED_CORRUPTION_TYPES, severities=(0.3,), seed="test", label_fn=label_for_extended,
    )
    assert len(variants) == len(EXTENDED_CORRUPTION_TYPES)
    for v in variants:
        assert set(v["label"]) == set(HEAD_NAMES_EXTENDED)
