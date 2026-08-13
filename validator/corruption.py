"""
Track 2.2 — Systematic corruption generation.

Generalizes the three corruption functions prototyped in poc_validator.py
(corrupt_graph()'s missing_entities/contradictions/fragmentation branches)
into a reusable module: severity-parameterized (not one fixed variant),
operating on the KG dict shape produced by extraction.schemas.KnowledgeGraph
(entities/relations lists) rather than a NetworkX graph, since that's the
format data/training/clean/*.json is actually saved in.

Also implements Track 2.3's labeling scheme — see label_for()'s docstring
for the decision on how the (now six) corruption types map onto
SimpleGNN's four output heads.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from extraction.schemas import RelationType

# The three types poc_validator.py originally prototyped. These map 1:1
# onto SimpleGNN's three corruption-specific heads (see label_for()).
DEFAULT_CORRUPTION_TYPES = ["missing_entities", "contradictions", "fragmentation"]

# Newer corruption types not yet covered by poc_validator.py. Generated for
# dataset completeness / a future head-expansion, but not mapped to a
# dedicated head under the current 4-head SimpleGNN — see label_for().
EXTRA_CORRUPTION_TYPES = ["entity_duplication", "relation_type_swap", "orphan_node_injection"]

ALL_CORRUPTION_TYPES = DEFAULT_CORRUPTION_TYPES + EXTRA_CORRUPTION_TYPES

DEFAULT_SEVERITIES = (0.1, 0.2, 0.3)


# ── Helpers ──────────────────────────────────────────────────────────────


def _degree_map(kg: dict) -> dict[str, int]:
    """Count how many relations touch each entity id (source or target)."""
    degrees: dict[str, int] = {e["id"]: 0 for e in kg["entities"]}
    for r in kg["relations"]:
        if r["source"] in degrees:
            degrees[r["source"]] += 1
        if r["target"] in degrees:
            degrees[r["target"]] += 1
    return degrees


def _n_to_affect(count: int, severity: float, *, cap_below_total: bool = False) -> int:
    """
    How many items `severity` fraction of `count` works out to, with a
    floor of 1 (so a low severity on a small graph still does *something*)
    and, when cap_below_total is set, a ceiling of count - 1 (so a
    corruption can never zero out an entire graph — see Track 2.5's
    "not reducing every graph to 0 nodes" sanity check).
    """
    if count == 0:
        return 0
    n = max(1, round(severity * count))
    if cap_below_total:
        n = min(n, max(1, count - 1))
    else:
        n = min(n, count)
    return n


def _seeded_rng(seed, corruption_type: str, severity: float) -> random.Random:
    return random.Random(f"{seed}:{corruption_type}:{severity}")


# ── Original three corruption types (map 1:1 to SimpleGNN's heads) ─────────


def corrupt_missing_entities(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Remove the highest-degree `severity` fraction of entities (and any relations touching them)."""
    kg = copy.deepcopy(kg)
    entities = kg["entities"]

    if len(entities) < 2:
        return kg, {"corruption_type": "missing_entities", "severity": severity, "skipped": True, "reason": "too few entities"}

    degrees = _degree_map(kg)
    # Shuffle first so ties don't always break the same way, then sort by degree desc.
    shuffled = entities[:]
    rng.shuffle(shuffled)
    ranked = sorted(shuffled, key=lambda e: degrees.get(e["id"], 0), reverse=True)

    num_to_remove = _n_to_affect(len(entities), severity, cap_below_total=True)
    removed_ids = {e["id"] for e in ranked[:num_to_remove]}

    kg["entities"] = [e for e in entities if e["id"] not in removed_ids]
    kept_relations = [r for r in kg["relations"] if r["source"] not in removed_ids and r["target"] not in removed_ids]
    removed_relation_count = len(kg["relations"]) - len(kept_relations)
    kg["relations"] = kept_relations

    return kg, {
        "corruption_type": "missing_entities",
        "severity": severity,
        "removed_entities": len(removed_ids),
        "removed_relations": removed_relation_count,
    }


def corrupt_contradictions(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Reverse source/target on `severity` fraction of relations, simulating a contradictory claim."""
    kg = copy.deepcopy(kg)
    relations = kg["relations"]

    if not relations:
        return kg, {"corruption_type": "contradictions", "severity": severity, "skipped": True, "reason": "no relations"}

    num_to_flip = _n_to_affect(len(relations), severity)
    indices = rng.sample(range(len(relations)), num_to_flip)

    for i in indices:
        r = relations[i]
        r["source"], r["target"] = r["target"], r["source"]

    return kg, {
        "corruption_type": "contradictions",
        "severity": severity,
        "flipped_relations": num_to_flip,
    }


def corrupt_fragmentation(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Remove `severity` fraction of relations at random to increase disconnection."""
    kg = copy.deepcopy(kg)
    relations = kg["relations"]

    if not relations:
        return kg, {"corruption_type": "fragmentation", "severity": severity, "skipped": True, "reason": "no relations"}

    num_to_remove = _n_to_affect(len(relations), severity)
    indices = set(rng.sample(range(len(relations)), num_to_remove))
    kg["relations"] = [r for i, r in enumerate(relations) if i not in indices]

    return kg, {
        "corruption_type": "fragmentation",
        "severity": severity,
        "removed_relations": num_to_remove,
    }


# ── Extra corruption types (not yet mapped to a SimpleGNN head) ────────────


def corrupt_entity_duplication(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """
    Duplicate `severity` fraction of entities under new ids (same name/type,
    same evidence — simulating a coreference-resolution failure where one
    real-world entity ends up as two nodes), and split some of the
    original's relations onto the duplicate.
    """
    kg = copy.deepcopy(kg)
    entities = kg["entities"]

    if not entities:
        return kg, {"corruption_type": "entity_duplication", "severity": severity, "skipped": True, "reason": "no entities"}

    num_to_duplicate = _n_to_affect(len(entities), severity)
    originals = rng.sample(entities, num_to_duplicate)

    new_entities = []
    reassigned = 0
    for orig in originals:
        dup_id = f"{orig['id']}_dup"
        dup = copy.deepcopy(orig)
        dup["id"] = dup_id
        dup["confidence"] = max(0.0, orig.get("confidence", 1.0) - 0.2)
        new_entities.append(dup)

        for r in kg["relations"]:
            if r["source"] == orig["id"] and rng.random() < 0.5:
                r["source"] = dup_id
                reassigned += 1
            elif r["target"] == orig["id"] and rng.random() < 0.5:
                r["target"] = dup_id
                reassigned += 1

    kg["entities"] = entities + new_entities

    return kg, {
        "corruption_type": "entity_duplication",
        "severity": severity,
        "duplicated_entities": num_to_duplicate,
        "reassigned_relations": reassigned,
    }


def corrupt_relation_type_swap(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Keep the correct entities but swap the relation label for `severity` fraction of relations."""
    kg = copy.deepcopy(kg)
    relations = kg["relations"]

    if not relations:
        return kg, {"corruption_type": "relation_type_swap", "severity": severity, "skipped": True, "reason": "no relations"}

    all_types = [t.value for t in RelationType]
    num_to_swap = _n_to_affect(len(relations), severity)
    indices = rng.sample(range(len(relations)), num_to_swap)

    for i in indices:
        r = relations[i]
        choices = [t for t in all_types if t != r.get("relation_type")]
        if choices:
            r["relation_type"] = rng.choice(choices)

    return kg, {
        "corruption_type": "relation_type_swap",
        "severity": severity,
        "swapped_relations": num_to_swap,
    }


def corrupt_orphan_node_injection(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Add unsupported entities with no relations and no evidence — fabricated/hallucinated nodes."""
    kg = copy.deepcopy(kg)
    entities = kg["entities"]
    base_count = len(entities) if entities else 5
    num_to_add = _n_to_affect(base_count, severity)

    from extraction.schemas import EntityType
    all_types = [t.value for t in EntityType]

    new_entities = []
    for i in range(num_to_add):
        new_entities.append({
            "id": f"orphan_{rng.randint(100000, 999999)}",
            "name": f"Unverified Entity {i}",
            "type": rng.choice(all_types),
            "document_frequency": 0,
            "confidence": 0.0,
            "evidence": [],
        })

    kg["entities"] = entities + new_entities

    return kg, {
        "corruption_type": "orphan_node_injection",
        "severity": severity,
        "injected_entities": num_to_add,
    }


CORRUPTION_FUNCS = {
    "missing_entities": corrupt_missing_entities,
    "contradictions": corrupt_contradictions,
    "fragmentation": corrupt_fragmentation,
    "entity_duplication": corrupt_entity_duplication,
    "relation_type_swap": corrupt_relation_type_swap,
    "orphan_node_injection": corrupt_orphan_node_injection,
}


# ── Track 2.3 — Labeling scheme ─────────────────────────────────────────────


def label_for(corruption_type: Optional[str]) -> dict:
    """
    Track 2.3's labeling decision.

    SimpleGNN (poc_validator.py) has exactly 4 sigmoid output heads:
    consistency, missing_entities, contradictions, fragmentation. The
    three original corruption types map onto those 1:1 — that part was
    easy. The three "extra" types added in this module
    (entity_duplication, relation_type_swap, orphan_node_injection) do
    NOT have a dedicated head yet.

    Decision: under the *current* 4-head architecture, extra-type
    corruptions only flip the `consistency` label to 0.0 (same as any
    other corruption — the graph genuinely isn't clean) and leave
    missing_entities/contradictions/fragmentation at 0.0. They are
    generated and saved (with full corruption_type/severity metadata) so
    the dataset is forward-compatible if Track 3.1 decides to grow
    SimpleGNN's head count, but Track 3's *default* training target
    should only treat missing_entities/contradictions/fragmentation
    variants as positive examples for those three heads — an
    entity_duplication variant is a real corruption, but training the
    current 3-head-specific outputs to fire on it would teach the model
    the wrong association (it's not fragmentation, missing entities, or
    a reversed edge).

    A clean (uncorrupted) KG gets consistency=1.0 and everything else 0.0.
    """
    if corruption_type is None:
        return {"consistency": 1.0, "missing_entities": 0.0, "contradictions": 0.0, "fragmentation": 0.0}

    return {
        "consistency": 0.0,
        "missing_entities": 1.0 if corruption_type == "missing_entities" else 0.0,
        "contradictions": 1.0 if corruption_type == "contradictions" else 0.0,
        "fragmentation": 1.0 if corruption_type == "fragmentation" else 0.0,
    }


# ── Orchestration ────────────────────────────────────────────────────────


def generate_corrupted_variants(
    kg: dict,
    corruption_types: list[str] = None,
    severities: tuple = DEFAULT_SEVERITIES,
    seed=None,
) -> list[dict]:
    """
    Generate one corrupted variant per (corruption_type, severity) pair.

    Each corrupted graph gets its own deterministic RNG seeded from
    (seed, corruption_type, severity), so re-running this on the same
    clean KG with the same `seed` reproduces the exact same corruptions.

    Returns a list of:
        {
            "corruption_type": str,
            "severity": float,
            "kg": <corrupted KG dict>,
            "label": <Track 2.3 label dict>,
            "corruption_metadata": <what the corruption function actually did>,
        }
    """
    if corruption_types is None:
        corruption_types = DEFAULT_CORRUPTION_TYPES

    variants = []
    for corruption_type in corruption_types:
        if corruption_type not in CORRUPTION_FUNCS:
            raise ValueError(f"Unknown corruption type: {corruption_type}")
        fn = CORRUPTION_FUNCS[corruption_type]

        for severity in severities:
            rng = _seeded_rng(seed, corruption_type, severity)
            corrupted_kg, metadata = fn(kg, severity, rng)
            variants.append({
                "corruption_type": corruption_type,
                "severity": severity,
                "kg": corrupted_kg,
                "label": label_for(corruption_type),
                "corruption_metadata": metadata,
            })

    return variants
