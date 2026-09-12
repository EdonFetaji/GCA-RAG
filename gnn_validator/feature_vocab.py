"""
FeatureVocab — the fixed index maps node/edge feature construction relies on.

Two ways to build one:

- FeatureVocab() (default): the closed ontology from
  kg_agentic_extraction.models.ontology — 8 entity types, 10 relation types.
  Correct for data/training/clean/*.json, which was produced by the *legacy*
  extractor bound to that closed ontology.

- FeatureVocab.fit(kgs): counts the entity/relation type strings a corpus of
  KGs actually uses and keeps the top-N most frequent, everything else falling
  into an explicit OOV bucket. Required for graphs produced by the current
  kg_agentic_extraction pipeline: per ADR 0004 ("Extraction is
  open-vocabulary"), the extractor names its own types — a single cluster can
  emit `GOVERNOR_OF`, `IS_GOVERNOR_OF`, etc. — and the ADR names "the Track 3
  validator especially" as needing to either canonicalize through the
  grounder's DBpedia mappings or "treat the local type as a free text
  feature". fit() is the latter: a frequency-based vocabulary is the simplest
  version of that, rather than crushing everything into a handful of enum
  buckets it will mostly miss.

Unknown types at inference time (whichever construction path was used) fall
back to a dedicated OTHER/UNKNOWN slot rather than raising.

The vocab actually used for training MUST be saved (save()) and the exact
same one loaded back for eval (load()) — the one-hot index a type maps to is
an artifact of construction order/corpus frequency, not a stable ontology
position, so training and eval drifting to two different FeatureVocab
instances silently corrupts every prediction.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from kg_agentic_extraction.models.knowledge_graph import normalize_type
from kg_agentic_extraction.models.ontology import EntityType, RelationType

UNKNOWN_RELATION = "__UNKNOWN_RELATION__"
UNKNOWN_ENTITY = "__UNKNOWN_ENTITY__"


@dataclass
class FeatureVocab:
    entity_types: list[str] = field(default_factory=lambda: [t.value for t in EntityType])
    relation_types: list[str] = field(default_factory=lambda: [t.value for t in RelationType] + [UNKNOWN_RELATION])

    def __post_init__(self) -> None:
        self._entity_idx = {t: i for i, t in enumerate(self.entity_types)}
        self._relation_idx = {t: i for i, t in enumerate(self.relation_types)}

    @classmethod
    def fit(cls, kgs: list[dict], max_entity_types: int = 32, max_relation_types: int = 48) -> "FeatureVocab":
        """
        Build an open-vocabulary FeatureVocab from a corpus of KG dicts
        (the {"entities": [...], "relations": [...]} shape).

        Types are folded through normalize_type() first — the same folding
        kg_agentic_extraction.models.knowledge_graph applies at extraction
        time — so "located in" / "Located-In" / "LOCATED_IN" count as one
        type rather than fragmenting the top-N cutoff across spelling variants.

        Always reserves one slot for OOV/rare types (UNKNOWN_ENTITY /
        UNKNOWN_RELATION) beyond max_entity_types / max_relation_types, so a
        type that shows up once at eval time on real-world long-tail data
        still gets a defined (if uninformative) feature rather than crashing.
        """
        entity_counts: Counter[str] = Counter()
        relation_counts: Counter[str] = Counter()
        for kg in kgs:
            for e in kg.get("entities", []):
                entity_counts[normalize_type(str(e.get("type", "")))] += 1
            for r in kg.get("relations", []):
                relation_counts[normalize_type(str(r.get("relation_type", "")))] += 1

        top_entities = [t for t, _ in entity_counts.most_common(max_entity_types)]
        top_relations = [t for t, _ in relation_counts.most_common(max_relation_types)]

        return cls(
            entity_types=top_entities + [UNKNOWN_ENTITY],
            relation_types=top_relations + [UNKNOWN_RELATION],
        )

    @property
    def num_entity_types(self) -> int:
        return len(self.entity_types)

    @property
    def num_relation_types(self) -> int:
        return len(self.relation_types)

    def entity_type_index(self, entity_type: str) -> int:
        key = normalize_type(str(entity_type)) if entity_type else ""
        if key in self._entity_idx:
            return self._entity_idx[key]
        # Closed-ontology vocabs (the default constructor) don't have
        # UNKNOWN_ENTITY — they use the ontology's own OTHER member instead.
        return self._entity_idx.get(UNKNOWN_ENTITY, self._entity_idx.get(EntityType.OTHER.value, 0))

    def relation_type_index(self, relation_type: str) -> int:
        key = normalize_type(str(relation_type)) if relation_type else ""
        return self._relation_idx.get(key, self._relation_idx[UNKNOWN_RELATION])

    # Node scalar features appended after the one-hot type block, in this
    # order — kept as a named constant so data.py and model.py can't drift
    # apart on how many/which scalar dims there are.
    NODE_SCALAR_FEATURES = ("confidence", "document_frequency_norm", "degree_norm", "evidence_count_norm")
    EDGE_SCALAR_FEATURES = ("confidence", "support_count_norm", "is_reverse")

    @property
    def node_feature_dim(self) -> int:
        return self.num_entity_types + len(self.NODE_SCALAR_FEATURES)

    @property
    def edge_feature_dim(self) -> int:
        return self.num_relation_types + len(self.EDGE_SCALAR_FEATURES)

    def to_json(self) -> dict:
        return {"entity_types": self.entity_types, "relation_types": self.relation_types}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureVocab":
        data = json.loads(Path(path).read_text())
        return cls(entity_types=data["entity_types"], relation_types=data["relation_types"])
