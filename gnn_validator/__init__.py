"""
Track 3 — graph-transformer KG validator.

Trains a GNN on the extended 6-head corruption scheme
(kg_dataset/corruption.py's EXTENDED_CORRUPTION_TYPES + label_for_extended):
consistency, missing_entities, fragmentation (missing edges),
orphan_node_injection (hallucinated/additional nodes), hallucinated_relations
(hallucinated/additional edges), contradictions.

Modules:
    feature_vocab.py — entity/relation type -> index maps, shared between
                        dataset construction and the model's embedding layers.
    data.py           — KG dict -> PyG Data conversion, and the Dataset that
                        reads data/training/clean + corrupted_extended via splits.json.
    model.py          — GraphTransformerValidator (stacked TransformerConv).
    train.py          — training loop, early stopping, checkpointing.
    eval.py           — per-head AUROC/F1 + heuristic baseline comparison.
"""
