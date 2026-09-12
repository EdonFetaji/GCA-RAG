"""
gnn_data.py — KG dict -> PyTorch Geometric Data, and the on-disk Dataset.

Reads the same file layout kg_dataset/generate_corruptions.py writes:
    data/training/clean/{cluster_idx}.json                    (label: all-clean)
    data/training/corrupted_extended/{idx}_{type}_{sev}.json  (label: saved in the file)

and partitions by data/training/splits.json's cluster-level split (Track 2.4)
so a cluster's clean graph and every corrupted variant derived from it stay
in the same split — the same leakage guard generate_splits.py's docstring
describes, just enforced here on the *loading* side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from gnn_validator.feature_vocab import FeatureVocab
from kg_dataset.corruption import HEAD_NAMES_EXTENDED, label_for_extended

# Fixed-scale caps for the scalar node/edge features. Deliberately NOT
# per-graph max-normalized (unlike poc_validator.py's build_node_features) —
# normalizing by a graph's own max makes the same raw value mean different
# things in a 5-node graph vs. a 50-node graph, which is exactly the kind of
# inconsistency a validator meant to generalize across cluster sizes
# shouldn't be trained on.
_DOC_FREQ_CAP = 10.0
_EVIDENCE_COUNT_CAP = 5.0
_DEGREE_CAP = 20.0
_SUPPORT_COUNT_CAP = 5.0


def _clipped(value: float, cap: float) -> float:
    return max(0.0, min(1.0, value / cap))


def kg_to_pyg(kg: dict, vocab: FeatureVocab) -> Data | None:
    """
    Convert one KG dict (the {"entities": [...], "relations": [...]} shape
    used throughout kg_dataset/) into a PyG Data object.

    Returns None for a degenerate graph (0 entities) — the caller should
    skip these rather than feed an empty batch through the model.

    Edges are added in both directions (original + a reversed copy flagged
    via the is_reverse scalar) so message passing isn't limited to the
    original relation direction — a missing/hallucinated/contradictory edge
    can only be judged from a node's full local neighborhood, not just its
    outgoing edges.
    """
    entities = kg.get("entities", [])
    relations = kg.get("relations", [])

    if not entities:
        return None

    id_to_idx = {e["id"]: i for i, e in enumerate(entities)}

    degree = [0] * len(entities)
    for r in relations:
        if r["source"] in id_to_idx:
            degree[id_to_idx[r["source"]]] += 1
        if r["target"] in id_to_idx:
            degree[id_to_idx[r["target"]]] += 1

    node_features = []
    for i, e in enumerate(entities):
        type_onehot = [0.0] * vocab.num_entity_types
        type_onehot[vocab.entity_type_index(e.get("type", "OTHER"))] = 1.0
        scalars = [
            float(e.get("confidence", 1.0)),
            _clipped(float(e.get("document_frequency", 1)), _DOC_FREQ_CAP),
            _clipped(float(degree[i]), _DEGREE_CAP),
            _clipped(float(len(e.get("evidence", []) or [])), _EVIDENCE_COUNT_CAP),
        ]
        node_features.append(type_onehot + scalars)

    x = torch.tensor(node_features, dtype=torch.float)

    edge_index_list = []
    edge_attr_list = []
    for r in relations:
        src, tgt = r.get("source"), r.get("target")
        if src not in id_to_idx or tgt not in id_to_idx:
            # A dangling reference would be a bug elsewhere in the pipeline
            # (check_dataset.py's structural-validity pass should catch
            # these before they get here) — skip defensively rather than crash.
            continue
        rel_onehot = [0.0] * vocab.num_relation_types
        rel_onehot[vocab.relation_type_index(r.get("relation_type", ""))] = 1.0
        confidence = float(r.get("confidence", 1.0))
        support_norm = _clipped(float(r.get("support_count", 1)), _SUPPORT_COUNT_CAP)

        # Forward edge.
        edge_index_list.append([id_to_idx[src], id_to_idx[tgt]])
        edge_attr_list.append(rel_onehot + [confidence, support_norm, 0.0])
        # Reverse edge (same relation semantics, flagged is_reverse=1) so
        # information can flow both ways during message passing.
        edge_index_list.append([id_to_idx[tgt], id_to_idx[src]])
        edge_attr_list.append(rel_onehot + [confidence, support_norm, 1.0])

    if edge_index_list:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, vocab.edge_feature_dim), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(entities))


class KGValidationDataset(Dataset):
    """
    One sample = one (possibly corrupted) KG + its 6-head label vector,
    ordered per HEAD_NAMES_EXTENDED: [consistency, missing_entities,
    fragmentation, orphan_node_injection, hallucinated_relations, contradictions].

    `split` selects "train" / "val" / "test" using data/training/splits.json's
    cluster-level partition (same file generate_splits.py writes).
    """

    def __init__(
        self,
        split: str,
        clean_dir: str | Path = "data/training/clean",
        corrupted_dir: str | Path = "data/training/corrupted_extended",
        splits_path: str | Path = "data/training/splits.json",
        vocab: FeatureVocab | None = None,
    ):
        self.vocab = vocab or FeatureVocab()
        splits = json.loads(Path(splits_path).read_text())
        if split not in splits["splits"]:
            raise ValueError(f"Unknown split {split!r}, expected one of {list(splits['splits'])}")
        cluster_ids = set(splits["splits"][split])

        self._samples: list[tuple[dict, list[float]]] = []  # (kg dict, label vector)

        clean_dir = Path(clean_dir)
        for path in sorted(clean_dir.glob("*.json")):
            if not path.stem.isdigit() or int(path.stem) not in cluster_ids:
                continue
            record = json.loads(path.read_text())
            kg = record["extraction"]["knowledge_graph"]
            label = label_for_extended(None)
            self._samples.append((kg, [label[h] for h in HEAD_NAMES_EXTENDED]))

        corrupted_dir = Path(corrupted_dir)
        if corrupted_dir.exists():
            for path in sorted(corrupted_dir.glob("*.json")):
                record = json.loads(path.read_text())
                if record["cluster_idx"] not in cluster_ids:
                    continue
                self._samples.append((record["knowledge_graph"], [record["label"][h] for h in HEAD_NAMES_EXTENDED]))

        if not self._samples:
            raise RuntimeError(
                f"No samples found for split={split!r}. Did you run "
                f"generate_corruptions.py --scheme extended?"
            )

        # kg_to_pyg() rebuilds every tensor from the raw dict — cheap once,
        # but a training run re-reads __getitem__ every epoch. At ~700
        # clusters x 16 variants that's ~11k conversions/epoch x 60-100
        # epochs, so caching the converted Data (not the raw dict) turns an
        # O(epochs) cost into O(1) after the first epoch.
        self._cache: dict[int, Data] = {}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Data, torch.Tensor]:
        if idx in self._cache:
            data = self._cache[idx].clone()
        else:
            kg, _ = self._samples[idx]
            data = kg_to_pyg(kg, self.vocab)
            if data is None:
                # Degenerate (0-entity) sample — use a 1-node placeholder
                # rather than raising, so a bad upstream record doesn't crash
                # an entire training epoch. It carries no real signal either way.
                data = Data(
                    x=torch.zeros((1, self.vocab.node_feature_dim), dtype=torch.float),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, self.vocab.edge_feature_dim), dtype=torch.float),
                    num_nodes=1,
                )
            self._cache[idx] = data
            data = data.clone()

        _, label = self._samples[idx]
        data.y = torch.tensor(label, dtype=torch.float).unsqueeze(0)
        return data
