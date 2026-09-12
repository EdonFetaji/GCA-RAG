"""
eval_gnn.py — per-head AUROC/F1 on the test split, plus a structural
heuristic baseline for comparison ("did the GNN actually learn something a
few graph-statistics thresholds couldn't already tell you").

AUROC is implemented by hand (Mann-Whitney U / rank-sum form) rather than
pulling in scikit-learn — pyproject.toml doesn't currently depend on it, and
one AUROC formula isn't worth adding a dependency to uv.lock for. Verified
against a known-value case in tests/gnn_validator/test_eval.py.

Run:
    python -m gnn_validator.eval --checkpoint data/gnn_checkpoints/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from gnn_validator.data import KGValidationDataset
from gnn_validator.feature_vocab import FeatureVocab
from gnn_validator.model import GraphTransformerValidator
from kg_dataset.corruption import HEAD_NAMES_EXTENDED

logger = logging.getLogger("eval_gnn")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def auroc(scores: list[float], labels: list[float]) -> float | None:
    """
    AUROC via the Mann-Whitney U statistic: the probability a random
    positive scores higher than a random negative, computed from rank sums
    so it's O(n log n) and needs no extra dependency.

    Returns None if a class is entirely absent (AUROC undefined) rather than
    a misleading 0.5/1.0.
    """
    pairs = sorted(zip(scores, labels), key=lambda p: p[0])
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed, averaged over ties
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1

    n_pos = sum(1 for _, y in pairs if y == 1.0)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    rank_sum_pos = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1.0)
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def f1_at_threshold(scores: list[float], labels: list[float], threshold: float = 0.5) -> float:
    tp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 1.0)
    fp = sum(1 for s, y in zip(scores, labels) if s >= threshold and y == 0.0)
    fn = sum(1 for s, y in zip(scores, labels) if s < threshold and y == 1.0)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def structural_heuristic_scores(loader: DataLoader, vocab: FeatureVocab) -> dict[str, list[float]]:
    """
    A non-learned baseline, one score per head, from raw graph statistics —
    the GNN validator's whole justification is that it should beat this by
    a real margin, not just barely.

    consistency:            1 - (isolated-node fraction)   [cheap proxy for "does this look intact"]
    missing_entities:       inverse of mean node degree (sparser -> more suspicious)
    fragmentation:          fraction of nodes in components other than the largest
    orphan_node_injection:  fraction of zero-confidence nodes
    hallucinated_relations: fraction of zero-confidence edges
    contradictions:         fraction of (source, relation_type) pairs with >1 distinct target

    `vocab` must be the SAME FeatureVocab the batch's x/edge_attr were built
    with — it's only used here to find where the scalar (confidence, etc.)
    columns sit after the one-hot type block, which shifts size under an
    open-vocabulary (--vocab fit) FeatureVocab.
    """
    import networkx as nx

    out = {h: [] for h in HEAD_NAMES_EXTENDED}
    for batch in loader:
        for i in range(batch.num_graphs):
            mask = batch.batch == i
            node_idx = mask.nonzero(as_tuple=True)[0].tolist()
            local = {n: k for k, n in enumerate(node_idx)}
            edge_mask = mask[batch.edge_index[0]]
            edges = batch.edge_index[:, edge_mask].t().tolist()
            edge_attr = batch.edge_attr[edge_mask]

            G = nx.DiGraph()
            G.add_nodes_from(range(len(node_idx)))
            for (s, t) in edges:
                G.add_edge(local[s], local[t])

            n_nodes = max(1, G.number_of_nodes())
            degrees = dict(G.degree())
            isolated_frac = sum(1 for d in degrees.values() if d == 0) / n_nodes
            mean_degree = sum(degrees.values()) / n_nodes

            components = list(nx.weakly_connected_components(G)) or [set(G.nodes())]
            largest = max(len(c) for c in components)
            fragmentation_frac = 1 - (largest / n_nodes)

            # node confidence lives in scalar feature index [num_entity_types + 0]
            node_conf = batch.x[mask][:, vocab.num_entity_types]
            zero_conf_node_frac = (node_conf <= 0.01).float().mean().item() if node_conf.numel() else 0.0

            if edge_attr.numel():
                edge_conf = edge_attr[:, vocab.num_relation_types]
                is_reverse = edge_attr[:, -1]
                forward = is_reverse < 0.5
                zero_conf_edge_frac = (edge_conf[forward] <= 0.01).float().mean().item() if forward.any() else 0.0
            else:
                zero_conf_edge_frac = 0.0

            seen_targets: dict[tuple, set] = {}
            for (s, t), a in zip(edges, edge_attr.tolist()):
                if a[-1] >= 0.5:  # skip synthetic reverse edges
                    continue
                rel_block = a[: vocab.num_relation_types]
                rel_idx = rel_block.index(1.0) if 1.0 in rel_block else -1
                key = (s, rel_idx)
                seen_targets.setdefault(key, set()).add(t)
            n_functional_pairs = max(1, len(seen_targets))
            contradiction_frac = sum(1 for v in seen_targets.values() if len(v) > 1) / n_functional_pairs

            out["consistency"].append(1 - isolated_frac)
            out["missing_entities"].append(1 / (1 + mean_degree))
            out["fragmentation"].append(fragmentation_frac)
            out["orphan_node_injection"].append(zero_conf_node_frac)
            out["hallucinated_relations"].append(zero_conf_edge_frac)
            out["contradictions"].append(contradiction_frac)

    return out


def evaluate(model, loader) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    model.eval()
    predictions = {h: [] for h in HEAD_NAMES_EXTENDED}
    labels = {h: [] for h in HEAD_NAMES_EXTENDED}
    with torch.no_grad():
        for batch in loader:
            scores = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            for i, h in enumerate(HEAD_NAMES_EXTENDED):
                predictions[h].extend(scores[h].tolist())
                labels[h].extend(batch.y[:, i].tolist())
    return predictions, labels


def main():
    parser = argparse.ArgumentParser(description="Evaluate the graph-transformer KG validator (Track 3.x)")
    parser.add_argument("--checkpoint", type=str, default="data/gnn_checkpoints/best_model.pt")
    parser.add_argument("--clean-dir", type=str, default="data/training/clean")
    parser.add_argument("--corrupted-dir", type=str, default="data/training/corrupted_extended")
    parser.add_argument("--splits-path", type=str, default="data/training/splits.json")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--vocab-path", type=str, default=None, help="Defaults to feature_vocab.json next to --checkpoint. MUST be the vocab saved during training — a fresh/default FeatureVocab() will silently misalign one-hot indices whenever training used --vocab fit.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the full report as JSON.")
    args = parser.parse_args()

    _configure_logging()

    ckpt = torch.load(args.checkpoint, weights_only=False)
    config = ckpt["config"]
    model = GraphTransformerValidator(
        node_feature_dim=config["node_feature_dim"],
        edge_feature_dim=config["edge_feature_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
        head_names=config["head_names"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info("Loaded checkpoint from epoch %d (val_loss=%.4f).", ckpt["epoch"], ckpt["val_loss"])

    vocab_path = Path(args.vocab_path) if args.vocab_path else Path(args.checkpoint).parent / "feature_vocab.json"
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"No feature_vocab.json found at {vocab_path}. train.py always saves one next to the "
            f"checkpoint — pass --vocab-path explicitly if you moved the checkpoint on its own."
        )
    vocab = FeatureVocab.load(vocab_path)
    logger.info("Loaded FeatureVocab from %s (%d entity types, %d relation types).", vocab_path, vocab.num_entity_types, vocab.num_relation_types)
    ds = KGValidationDataset(args.split, args.clean_dir, args.corrupted_dir, args.splits_path, vocab)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Evaluating on %s split: %d samples.", args.split, len(ds))

    predictions, labels = evaluate(model, loader)
    heuristic_scores = structural_heuristic_scores(loader, vocab)

    report = {"split": args.split, "n_samples": len(ds), "heads": {}}
    logger.info("%-24s %10s %10s %14s %14s", "head", "GNN AUROC", "GNN F1@0.5", "heuristic AUROC", "n_positive")
    for h in HEAD_NAMES_EXTENDED:
        gnn_auc = auroc(predictions[h], labels[h])
        gnn_f1 = f1_at_threshold(predictions[h], labels[h])
        heur_auc = auroc(heuristic_scores[h], labels[h])
        n_pos = sum(labels[h])
        report["heads"][h] = {"gnn_auroc": gnn_auc, "gnn_f1_at_0.5": gnn_f1, "heuristic_auroc": heur_auc, "n_positive": n_pos}
        logger.info(
            "%-24s %10s %10s %14s %14d",
            h,
            f"{gnn_auc:.3f}" if gnn_auc is not None else "n/a",
            f"{gnn_f1:.3f}",
            f"{heur_auc:.3f}" if heur_auc is not None else "n/a",
            int(n_pos),
        )

    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
        logger.info("Saved report to %s", args.output)


if __name__ == "__main__":
    main()
