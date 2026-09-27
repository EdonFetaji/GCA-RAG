"""
Track 2.2 — Systematic corruption generation.

Generalizes the three corruption functions prototyped in poc_validator.py
(corrupt_graph()'s missing_entities/contradictions/fragmentation branches)
into a reusable module: severity-parameterized (not one fixed variant),
operating on the KG dict shape (entities/relations lists) rather than a
NetworkX graph, since that's the format data/training/clean/*.json is
actually saved in.

Each corruption records the elements it damaged (`bad_entity_ids` /
`bad_relation_keys` in its metadata) as element-level training labels.
Fabricated elements copy confidence and evidence from real ones, so they
cannot be spotted by a value no real element has.

Also implements Track 2.3's labeling scheme — see label_for()'s docstring
for the decision on how the (now six) corruption types map onto
SimpleGNN's four output heads.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from kg_agentic_extraction.models.ontology import EntityType, RelationType

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


def relation_key(r: dict) -> str:
    """The same identity kg_agentic_extraction's `Relation.key` uses."""
    return f"{r['source']}|{r['relation_type']}|{r['target']}"


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


def _labels(metadata: dict, bad_entity_ids=(), bad_relation_keys=()) -> dict:
    """Attach the element-level labels to a corruption's metadata."""
    metadata["bad_entity_ids"] = sorted(set(bad_entity_ids))
    metadata["bad_relation_keys"] = sorted(set(bad_relation_keys))
    return metadata


def _skipped(corruption_type: str, severity: float, reason: str) -> dict:
    return _labels({"corruption_type": corruption_type, "severity": severity, "skipped": True, "reason": reason})


#: A removal labels an entity once it has lost this fraction of its edges.
LOST_EDGE_FRACTION = 0.5


def _lost_most_edges(relations: list[dict], removed: list[dict], survivors: set[str]) -> set[str]:
    """Surviving entities that lost at least LOST_EDGE_FRACTION of their edges to `removed`."""
    degree: dict[str, int] = {}
    lost: dict[str, int] = {}
    for r in relations:
        for x in (r["source"], r["target"]):
            degree[x] = degree.get(x, 0) + 1
    for r in removed:
        for x in (r["source"], r["target"]):
            lost[x] = lost.get(x, 0) + 1
    return {x for x, n in lost.items() if x in survivors and n / degree[x] >= LOST_EDGE_FRACTION}


def _fabricated_relation_fields(kg: dict, rng: random.Random, near: str | None = None) -> dict:
    """Realistic support/confidence/evidence for a fabricated relation, borrowed from real ones."""
    relations = kg["relations"]
    if not relations:
        return {"support_count": 1, "source_documents": [], "confidence": 0.9, "evidence": []}
    touching = [r for r in relations if near in (r["source"], r["target"])] if near else []
    donor = rng.choice(touching or relations)
    return {
        "support_count": 1,
        "source_documents": list(donor.get("source_documents", [])),
        "confidence": rng.choice(relations).get("confidence", 1.0),
        "evidence": copy.deepcopy(donor.get("evidence", [])),
    }


def _fabricated_entity_fields(kg: dict, rng: random.Random) -> dict:
    """Realistic frequency/confidence/evidence for a fabricated entity, borrowed from real ones."""
    entities = kg["entities"]
    if not entities:
        return {"document_frequency": 1, "confidence": 0.9, "evidence": []}
    return {
        "document_frequency": 1,
        "confidence": rng.choice(entities).get("confidence", 1.0),
        "evidence": copy.deepcopy(rng.choice(entities).get("evidence", [])),
    }


# ── Original three corruption types (map 1:1 to SimpleGNN's heads) ─────────


def corrupt_missing_entities(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Remove the highest-degree `severity` fraction of entities (and any relations touching them)."""
    kg = copy.deepcopy(kg)
    entities = kg["entities"]

    if len(entities) < 2:
        return kg, _skipped("missing_entities", severity, "too few entities")

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

    dropped = [r for r in kg["relations"] if r["source"] in removed_ids or r["target"] in removed_ids]
    survivors = {e["id"] for e in kg["entities"]}
    bereaved = _lost_most_edges(kg["relations"], dropped, survivors)
    kg["relations"] = kept_relations

    return kg, _labels(
        {
            "corruption_type": "missing_entities",
            "severity": severity,
            "removed_entities": len(removed_ids),
            "removed_relations": removed_relation_count,
        },
        bad_entity_ids=bereaved,
    )


# Relation types where a given source realistically has *one* correct target
# at a time (a person has one current affiliation, an entity is located in
# one place, a company is acquired by one acquirer) — so giving the same
# source a second target of the same type is a genuine logical contradiction,
# not just structural noise. This is the fix noted as outstanding: the old
# implementation (uniform random edge-direction flipping) didn't actually
# simulate contradictory claims, just topology noise.
FUNCTIONAL_RELATION_TYPES = {
    "LOCATED_IN",
    "AFFILIATED_WITH",
    "ACQUIRED",
    "OCCURRED_ON",
    "OCCURRED_IN",
    "HAS_AGE",
    "BORN_IN",
    "BORN_ON",
    "DIED_IN",
    "DIED_ON",
    "HEADQUARTERED_IN",
    "CAPITAL_OF",
    "SPOUSE_OF",
    "MARRIED_TO",
}


def corrupt_contradictions(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """
    Inject `severity`-proportional logical contradictions: for a relation
    whose type is in FUNCTIONAL_RELATION_TYPES, add a second relation with
    the same source and relation_type but a different target — i.e. two
    incompatible claims about the same subject ("X is LOCATED_IN Paris" and
    "X is LOCATED_IN Tokyo"), rather than just noise in the graph topology.

    The conflicting target has the same entity type as the real one, so the
    type alone does not give it away.

    Falls back to the old behavior (reversing source/target on a random
    relation) for graphs that have no functional-relation edges to violate,
    so this never silently no-ops on a graph that's all RELATED_TO/CAUSES/etc.
    """
    kg = copy.deepcopy(kg)
    relations = kg["relations"]
    entities = kg["entities"]

    if not relations:
        return kg, _skipped("contradictions", severity, "no relations")

    functional = [r for r in relations if r.get("relation_type") in FUNCTIONAL_RELATION_TYPES]
    ids = [e["id"] for e in entities]
    type_of = {e["id"]: e.get("type") for e in entities}

    if not functional or len(ids) < 3:
        # Fallback: original edge-reversal behavior.
        num_to_flip = _n_to_affect(len(relations), severity)
        indices = rng.sample(range(len(relations)), num_to_flip)
        for i in indices:
            r = relations[i]
            r["source"], r["target"] = r["target"], r["source"]
        return kg, _labels(
            {
                "corruption_type": "contradictions",
                "severity": severity,
                "flipped_relations": num_to_flip,
                "fallback": "no_functional_relations",
            },
            bad_relation_keys=[relation_key(relations[i]) for i in indices],
        )

    num_to_violate = _n_to_affect(len(functional), severity)
    chosen = rng.sample(functional, num_to_violate)

    existing = {relation_key(r) for r in relations}
    injected = []
    for base in chosen:
        others = [i for i in ids if i != base["source"] and i != base["target"]]
        same_type = [i for i in others if type_of[i] == type_of.get(base["target"])]
        candidates = same_type or others
        if not candidates:
            continue
        new = {
            "source": base["source"],
            "target": rng.choice(candidates),
            "relation_type": base["relation_type"],
            "support_count": 1,
            "source_documents": list(base.get("source_documents", [])),
            "confidence": base.get("confidence", 1.0),
            "evidence": copy.deepcopy(base.get("evidence", [])),
        }
        if relation_key(new) in existing:
            continue
        existing.add(relation_key(new))
        injected.append(new)

    kg["relations"] = relations + injected

    return kg, _labels(
        {
            "corruption_type": "contradictions",
            "severity": severity,
            "contradictory_relations_added": len(injected),
        },
        bad_relation_keys=[relation_key(r) for r in injected],
    )


def corrupt_fragmentation(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Remove `severity` fraction of relations at random to increase disconnection."""
    kg = copy.deepcopy(kg)
    relations = kg["relations"]

    if not relations:
        return kg, _skipped("fragmentation", severity, "no relations")

    num_to_remove = _n_to_affect(len(relations), severity)
    indices = set(rng.sample(range(len(relations)), num_to_remove))
    kg["relations"] = [r for i, r in enumerate(relations) if i not in indices]

    survivors = {e["id"] for e in kg["entities"]}
    endpoints = _lost_most_edges(relations, [relations[i] for i in indices], survivors)

    return kg, _labels(
        {
            "corruption_type": "fragmentation",
            "severity": severity,
            "removed_relations": num_to_remove,
        },
        bad_entity_ids=endpoints,
    )


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
        return kg, _skipped("entity_duplication", severity, "no entities")

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

    return kg, _labels(
        {
            "corruption_type": "entity_duplication",
            "severity": severity,
            "duplicated_entities": num_to_duplicate,
            "reassigned_relations": reassigned,
        },
        bad_entity_ids=[e["id"] for e in new_entities],
    )


def corrupt_relation_type_swap(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """
    Keep the correct entities but swap the relation label for `severity` fraction of relations.

    Replacement labels come from the graph's own vocabulary (ADR 0004).
    """
    kg = copy.deepcopy(kg)
    relations = kg["relations"]

    if not relations:
        return kg, _skipped("relation_type_swap", severity, "no relations")

    observed = sorted({r["relation_type"] for r in relations})
    all_types = observed if len(observed) > 1 else [t.value for t in RelationType]
    num_to_swap = _n_to_affect(len(relations), severity)
    indices = rng.sample(range(len(relations)), num_to_swap)

    for i in indices:
        r = relations[i]
        choices = [t for t in all_types if t != r.get("relation_type")]
        if choices:
            r["relation_type"] = rng.choice(choices)

    return kg, _labels(
        {
            "corruption_type": "relation_type_swap",
            "severity": severity,
            "swapped_relations": num_to_swap,
        },
        bad_relation_keys=[relation_key(relations[i]) for i in indices],
    )


def corrupt_orphan_node_injection(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """Add unsupported entities with no relations — fabricated/hallucinated nodes."""
    kg = copy.deepcopy(kg)
    entities = kg["entities"]
    base_count = len(entities) if entities else 5
    num_to_add = _n_to_affect(base_count, severity)

    all_types = [e["type"] for e in entities if e.get("type")] or [t.value for t in EntityType]

    new_entities = []
    for i in range(num_to_add):
        new_entities.append({
            "id": f"orphan_{rng.randint(100000, 999999)}",
            "name": f"Unverified Entity {i}",
            "type": rng.choice(all_types),
            **_fabricated_entity_fields(kg, rng),
        })

    kg["entities"] = entities + new_entities

    return kg, _labels(
        {
            "corruption_type": "orphan_node_injection",
            "severity": severity,
            "injected_entities": num_to_add,
        },
        bad_entity_ids=[e["id"] for e in new_entities],
    )


def corrupt_hallucinated_relations(kg: dict, severity: float, rng: random.Random) -> tuple[dict, dict]:
    """
    Add `severity`-proportional fabricated relations between *existing*
    entities that have no supporting evidence — simulating the extractor
    inventing a relation between two real entities it never actually saw
    connected (as opposed to orphan_node_injection, which fabricates whole
    entities). This is the "additional / hallucinated edges" case that
    orphan_node_injection alone doesn't cover.

    The fabricated relation_type is sampled from the graph's OWN observed
    vocabulary (falling back to a generic placeholder only if the graph has
    no relations to draw from), not from kg_agentic_extraction's closed
    RelationType enum. Per ADR 0004 ("Extraction is open-vocabulary"), the
    extractor now names its own relation types per-cluster, and the ADR
    explicitly flags this exact mistake in corrupt_relation_type_swap:
    substituting an enum type into an open-vocabulary graph "makes the
    corruption trivially detectable and the negative sample too easy",
    because the injected type is foreign to the vocabulary the grader/GNN
    would ever see this graph use. Reusing the graph's own types keeps the
    task hard: a hallucinated edge should be a real relation the pair just
    never had, not a type that's an instant tell on its own.

    Only adds pairs that aren't already connected (either direction), so a
    hallucinated relation is always a genuinely new edge, not a duplicate of
    a real one.
    """
    kg = copy.deepcopy(kg)
    entities = kg["entities"]
    relations = kg["relations"]

    if len(entities) < 2:
        return kg, _skipped("hallucinated_relations", severity, "too few entities")

    existing_pairs = {(r["source"], r["target"]) for r in relations} | {(r["target"], r["source"]) for r in relations}
    ids = [e["id"] for e in entities]
    type_of = {e["id"]: e.get("type") for e in entities}
    observed_types = sorted({r["relation_type"] for r in relations if r.get("relation_type")})
    candidate_types = observed_types or ["RELATED_TO"]
    by_type_pair: dict[tuple, list[str]] = {}
    for r in relations:
        by_type_pair.setdefault((type_of.get(r["source"]), type_of.get(r["target"])), []).append(r["relation_type"])

    num_to_add = _n_to_affect(len(entities), severity)
    added = []
    attempts = 0
    max_attempts = num_to_add * 20 + 20
    while len(added) < num_to_add and attempts < max_attempts:
        attempts += 1
        source, target = rng.sample(ids, 2)
        if (source, target) in existing_pairs:
            continue
        existing_pairs.add((source, target))
        added.append({
            "source": source,
            "target": target,
            "relation_type": rng.choice(by_type_pair.get((type_of[source], type_of[target])) or candidate_types),
            **_fabricated_relation_fields(kg, rng, near=source),
        })

    kg["relations"] = relations + added

    return kg, _labels(
        {
            "corruption_type": "hallucinated_relations",
            "severity": severity,
            "injected_relations": len(added),
        },
        bad_relation_keys=[relation_key(r) for r in added],
    )


CORRUPTION_FUNCS = {
    "missing_entities": corrupt_missing_entities,
    "contradictions": corrupt_contradictions,
    "fragmentation": corrupt_fragmentation,
    "entity_duplication": corrupt_entity_duplication,
    "relation_type_swap": corrupt_relation_type_swap,
    "orphan_node_injection": corrupt_orphan_node_injection,
    "hallucinated_relations": corrupt_hallucinated_relations,
}

# The five types trained against the *extended*, 6-head validator (see
# label_for_extended()): missing nodes, missing edges (fragmentation),
# hallucinated/additional nodes (orphans), hallucinated/additional edges,
# and contradictions. entity_duplication/relation_type_swap stay unmapped
# "extra" types, same as before.
EXTENDED_CORRUPTION_TYPES = [
    "missing_entities",
    "fragmentation",
    "orphan_node_injection",
    "hallucinated_relations",
    "contradictions",
]

HEAD_NAMES_EXTENDED = [
    "consistency",
    "missing_entities",
    "fragmentation",
    "orphan_node_injection",
    "hallucinated_relations",
    "contradictions",
]


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


def label_for_extended(corruption_type: Optional[str]) -> dict:
    """
    6-head labeling scheme for the graph-transformer validator: the four
    original heads plus dedicated heads for the two EXTENDED_CORRUPTION_TYPES
    that label_for() couldn't place (orphan_node_injection, hallucinated_relations).

    Kept as a separate function rather than changing label_for() in place —
    label_for()'s 4-key dict is still what the legacy 3-corruption-type
    dataset (data/training/corrupted/*, generated with the default scheme)
    is labeled with, and existing tooling that reads those files expects
    exactly those 4 keys.
    """
    if corruption_type is None:
        return {name: (1.0 if name == "consistency" else 0.0) for name in HEAD_NAMES_EXTENDED}

    if corruption_type not in EXTENDED_CORRUPTION_TYPES:
        raise ValueError(f"{corruption_type!r} is not one of EXTENDED_CORRUPTION_TYPES: {EXTENDED_CORRUPTION_TYPES}")

    return {name: (1.0 if name == corruption_type else 0.0) for name in HEAD_NAMES_EXTENDED}


# ── Orchestration ────────────────────────────────────────────────────────


def generate_corrupted_variants(
    kg: dict,
    corruption_types: list[str] = None,
    severities: tuple = DEFAULT_SEVERITIES,
    seed=None,
    label_fn=label_for,
) -> list[dict]:
    """
    Generate one corrupted variant per (corruption_type, severity) pair.

    Each corrupted graph gets its own deterministic RNG seeded from
    (seed, corruption_type, severity), so re-running this on the same
    clean KG with the same `seed` reproduces the exact same corruptions.

    `label_fn` defaults to the legacy 4-head label_for(); pass
    label_for_extended to label for the 6-head graph-transformer validator
    (requires corruption_types to be a subset of EXTENDED_CORRUPTION_TYPES).

    Returns a list of:
        {
            "corruption_type": str,
            "severity": float,
            "kg": <corrupted KG dict>,
            "label": <label dict>,
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
                "label": label_fn(corruption_type),
                "corruption_metadata": metadata,
            })

    return variants
