import json
import random

import torch
from torch_geometric.loader import DataLoader

from gnn_validator.data import UNLABELED, OnTheFlyCorruptionDataset, kg_to_pyg
from gnn_validator.feature_vocab import FeatureVocab, relation_tokens
from gnn_validator.model import GraphTransformerValidator
from kg_dataset.corruption import corrupt_hallucinated_relations, relation_key


def _kg(n=6):
    entities = [
        {
            "id": f"e{i}",
            "name": f"N{i}",
            "type": "PERSON" if i % 2 else "LOCATION",
            "document_frequency": 1,
            "confidence": 0.9,
            "evidence": [{"document_index": 0, "quote": "q"}],
        }
        for i in range(n)
    ]
    relations = [
        {
            "source": f"e{i}",
            "target": f"e{i + 1}",
            "relation_type": "EMPLOYED_BY" if i % 2 else "LOCATED_IN",
            "support_count": 1,
            "source_documents": [0],
            "confidence": 0.95,
            "evidence": [{"document_index": 0, "quote": "q"}],
        }
        for i in range(n - 1)
    ]
    return {"entities": entities, "relations": relations}


def test_token_vocab_shares_words_across_types():
    vocab = FeatureVocab.fit([_kg()], relation_mode="token")
    assert relation_tokens("employed by") == ["EMPLOYED", "BY"]
    a = vocab.relation_type_vector("EMPLOYED_BY")
    b = vocab.relation_type_vector("EMPLOYEE_BY")  # unseen type, shares "BY"
    assert sum(a) == 2.0
    assert any(x and y for x, y in zip(a, b, strict=True))


def test_token_vocab_round_trips_through_json(tmp_path):
    vocab = FeatureVocab.fit([_kg()], relation_mode="token")
    vocab.save(tmp_path / "v.json")
    loaded = FeatureVocab.load(tmp_path / "v.json")
    assert loaded.relation_mode == "token"
    assert loaded.relation_type_vector("LOCATED_IN") == vocab.relation_type_vector("LOCATED_IN")


def test_old_vocab_files_load_as_type_mode(tmp_path):
    path = tmp_path / "old.json"
    path.write_text(json.dumps(FeatureVocab().to_json() | {"relation_mode": None}))
    data = json.loads(path.read_text())
    del data["relation_mode"]
    path.write_text(json.dumps(data))
    assert FeatureVocab.load(path).relation_mode == "type"


def test_kg_to_pyg_element_labels_follow_the_corruption():
    kg = _kg()
    corrupted, meta = corrupt_hallucinated_relations(kg, severity=0.3, rng=random.Random(0))
    vocab = FeatureVocab.fit([kg], relation_mode="token")
    data = kg_to_pyg(corrupted, vocab, set(meta["bad_entity_ids"]), set(meta["bad_relation_keys"]))
    assert data.y_edge.shape[0] == data.edge_index.shape[1] == 2 * len(corrupted["relations"])
    bad = {relation_key(r) for r in corrupted["relations"]} & set(meta["bad_relation_keys"])
    assert int(data.y_edge.sum()) == 2 * len(bad)  # both directions labeled
    for i, pos in enumerate(data.rel_index.tolist()):
        assert data.y_edge[i] == (1.0 if relation_key(corrupted["relations"][pos]) in bad else 0.0)


def test_kg_to_pyg_without_labels_is_unlabeled():
    data = kg_to_pyg(_kg(), FeatureVocab())
    assert (data.y_node == UNLABELED).all() and (data.y_edge == UNLABELED).all()


def test_on_the_fly_dataset_and_model_shapes(tmp_path):
    clean = tmp_path / "clean"
    clean.mkdir()
    for i in range(3):
        (clean / f"{i}.json").write_text(
            json.dumps({"extraction": {"knowledge_graph": _kg(6 + i)}})
        )
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"splits": {"train": [0, 1, 2], "val": [], "test": []}}))

    vocab = FeatureVocab.fit([_kg()], relation_mode="token")
    ds = OnTheFlyCorruptionDataset("train", clean, splits, vocab, resample=True)
    assert len(ds) == 3 * 6
    first = [ds.sample(i)[0] for i in range(len(ds))]
    assert [ds.sample(i)[0] for i in range(len(ds))] == first  # deterministic within an epoch

    batch = next(iter(DataLoader(ds, batch_size=len(ds))))
    model = GraphTransformerValidator(
        vocab.node_feature_dim, vocab.edge_feature_dim, hidden_dim=16, num_heads=2
    )
    out = model.forward_all(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
    assert out["graph"].shape == (len(ds), 6)
    assert out["node"].shape == (batch.num_nodes,)
    assert out["edge"].shape == (batch.edge_index.shape[1],)
    assert batch.y_node.shape == out["node"].shape and batch.y_edge.shape == out["edge"].shape


def test_graph_only_model_still_builds():
    model = GraphTransformerValidator(10, 5, hidden_dim=16, num_heads=2, element_heads=False)
    x, ei, ea = torch.zeros(3, 10), torch.tensor([[0, 1], [1, 2]]), torch.zeros(2, 5)
    out = model.forward_all(x, ei, ea, torch.zeros(3, dtype=torch.long))
    assert set(out) == {"graph"}
