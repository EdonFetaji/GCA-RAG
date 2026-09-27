"""
GNNValidator — the trained graph-transformer (gnn_validator/) as a GraphValidator.

Takes one checkpoint directory (best_model.pt + feature_vocab.json): the vocab
is part of the model, and any other one would scramble the input features.
The checkpoint needs element heads, since their scores are what the grader sees.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.validation import FlaggedElement, ValidationReport

logger = logging.getLogger(__name__)


class GNNValidator:
    """Runs the trained validator over one candidate graph at a time."""

    def __init__(
        self, checkpoint_dir: str | Path, *, top_k: int = 5, min_score: float = 0.5
    ) -> None:
        # Lazy: runs with the validator off should not import torch.
        import torch

        from gnn_validator.feature_vocab import FeatureVocab
        from gnn_validator.model import GraphTransformerValidator

        checkpoint_dir = Path(checkpoint_dir)
        model_path = checkpoint_dir / "best_model.pt"
        vocab_path = checkpoint_dir / "feature_vocab.json"
        for path in (model_path, vocab_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"GNN validator needs {path}. Train one with `python -m gnn_validator.train` "
                    "(it writes both files) or point KG_GNN_CHECKPOINT_DIR at an existing run."
                )

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        if not config.get("element_heads", False):
            raise ValueError(
                f"{model_path} was trained without element heads, so it cannot point at "
                "individual entities or relations. Retrain with the current gnn_validator.train."
            )

        self._torch = torch
        self.vocab = FeatureVocab.load(vocab_path)
        self.model = GraphTransformerValidator(
            node_feature_dim=config["node_feature_dim"],
            edge_feature_dim=config["edge_feature_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            head_names=config["head_names"],
            element_heads=True,
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.top_k = top_k
        self.min_score = min_score
        logger.info(
            "GNN validator loaded from %s (epoch %s, %d relation features, mode=%s)",
            checkpoint_dir,
            ckpt.get("epoch"),
            self.vocab.num_relation_types,
            self.vocab.relation_mode,
        )

    def validate(self, graph: KnowledgeGraph, *, iteration: int) -> ValidationReport:
        from gnn_validator.data import kg_to_pyg

        torch = self._torch
        kg = graph.model_dump(mode="json")
        data = kg_to_pyg(kg, self.vocab)
        if data is None:
            return ValidationReport(iteration=iteration)

        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            out = self.model.forward_all(data.x, data.edge_index, data.edge_attr, batch)
            graph_scores = torch.sigmoid(out["graph"][0]).tolist()
            node_scores = torch.sigmoid(out["node"]).tolist()
            edge_scores = torch.sigmoid(out["edge"]).tolist()

        names = {e.id: e.name for e in graph.entities}
        candidates: list[FlaggedElement] = [
            FlaggedElement(
                kind="entity", element_id=e.id, description=f"{e.name} ({e.type})", score=s
            )
            for e, s in zip(graph.entities, node_scores, strict=True)
        ]
        is_reverse = data.edge_attr[:, -1].tolist() if data.edge_attr.numel() else []
        for score, rev, pos in zip(edge_scores, is_reverse, data.rel_index.tolist(), strict=True):
            if rev >= 0.5:
                continue  # each relation is scored once, on its forward copy
            r = graph.relations[pos]
            candidates.append(
                FlaggedElement(
                    kind="relation",
                    element_id=r.key,
                    description=f"{names[r.source]} -[{r.relation_type}]-> {names[r.target]}",
                    score=score,
                )
            )

        flagged = sorted(
            (c for c in candidates if c.score >= self.min_score), key=lambda c: -c.score
        )
        return ValidationReport(
            iteration=iteration,
            scores={
                name: round(s, 4)
                for name, s in zip(self.model.head_names, graph_scores, strict=True)
            },
            flagged=flagged[: self.top_k],
        )
