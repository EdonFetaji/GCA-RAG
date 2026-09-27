"""
train_gnn.py — train GraphTransformerValidator on the extended 6-head scheme.

Run:
    python -m gnn_validator.train
    python -m gnn_validator.train --epochs 200 --hidden-dim 96 --num-layers 4
    python -m gnn_validator.train --gcs-bucket my-bucket --gcs-prefix gnn_checkpoints

    # Real extracted clusters, corrupted in memory, with per-node/per-edge heads:
    python -m gnn_validator.train --data on-the-fly --vocab fit-tokens \
        --clean-dir data/training/clean_gcs --splits-path data/training/splits_gcs.json

Loss = mean graph-head BCE + element_loss_weight x (node BCE + edge BCE), the
element terms over labeled elements only (files-mode samples are UNLABELED).

CPU-only by design (see pyproject.toml's "PyTorch: CPU-only build" note) —
this doesn't touch cuda, matching the 4-core VM the rest of the pipeline runs on.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from gnn_validator.data import (
    KGValidationDataset,
    OnTheFlyCorruptionDataset,
    UNLABELED,
    _load_clean_kgs,
)
from gnn_validator.feature_vocab import FeatureVocab
from gnn_validator.model import GraphTransformerValidator
from kg_dataset.corruption import HEAD_NAMES_EXTENDED

logger = logging.getLogger("train_gnn")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def _element_loss(criterion, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean BCE over the labeled elements; zero when there are none."""
    mask = labels != UNLABELED
    if not mask.any():
        return logits.sum() * 0.0
    return criterion(logits[mask], labels[mask]).mean()


def run_epoch(model, loader, criterion, optimizer=None, element_loss_weight: float = 1.0) -> tuple[float, dict[str, float]]:
    """One pass over `loader`. Trains if `optimizer` is given, else eval-only (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    per_head_loss = {h: 0.0 for h in HEAD_NAMES_EXTENDED}
    if model.element_heads:
        per_head_loss.update({"element_node": 0.0, "element_edge": 0.0})

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            out = model.forward_all(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_per_head = criterion(out["graph"], batch.y)  # [batch, num_heads], reduction='none'
            loss = loss_per_head.mean()
            if model.element_heads:
                node_loss = _element_loss(criterion, out["node"], batch.y_node)
                edge_loss = _element_loss(criterion, out["edge"], batch.y_edge)
                loss = loss + element_loss_weight * (node_loss + edge_loss)
                per_head_loss["element_node"] += node_loss.item()
                per_head_loss["element_edge"] += edge_loss.item()

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            for i, h in enumerate(HEAD_NAMES_EXTENDED):
                per_head_loss[h] += loss_per_head[:, i].mean().item()
            n_batches += 1

    avg_loss = total_loss / max(1, n_batches)
    avg_per_head = {h: v / max(1, n_batches) for h, v in per_head_loss.items()}
    return avg_loss, avg_per_head


def maybe_upload_to_gcs(local_path: Path, bucket: str | None, prefix: str) -> None:
    """
    Best-effort checkpoint upload — mirrors kg_agentic_extraction/storage/gcs.py's
    stance that the local file is the source of truth and a missing/misconfigured
    bucket should cost the run nothing, not crash a finished training run.
    """
    if not bucket:
        return
    try:
        from google.cloud import storage
        client = storage.Client()
        blob_name = f"{prefix.strip('/')}/{local_path.name}" if prefix else local_path.name
        client.bucket(bucket).blob(blob_name).upload_from_filename(str(local_path))
        logger.info("Uploaded %s to gs://%s/%s", local_path, bucket, blob_name)
    except Exception:
        logger.exception("GCS upload failed for %s — checkpoint is still saved locally, continuing.", local_path)


def main():
    parser = argparse.ArgumentParser(description="Train the graph-transformer KG validator (Track 3)")
    parser.add_argument(
        "--data", choices=["files", "on-the-fly"], default="files",
        help=(
            "'files' (default): clean + pre-generated corrupted_extended/*.json. "
            "'on-the-fly': clean files only, corrupted in memory per sample (fresh every "
            "epoch for train) with element-level labels — nothing is written anywhere."
        ),
    )
    parser.add_argument("--clean-dir", type=str, default="data/training/clean")
    parser.add_argument("--corrupted-dir", type=str, default="data/training/corrupted_extended")
    parser.add_argument("--splits-path", type=str, default="data/training/splits.json")
    parser.add_argument("--output-dir", type=str, default="data/gnn_checkpoints")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15, help="Early-stop after this many epochs without val-loss improvement.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gcs-bucket", type=str, default=None, help="Optional: upload best checkpoint here (not on the critical path — failures are logged, not raised).")
    parser.add_argument("--gcs-prefix", type=str, default="gnn_checkpoints")
    parser.add_argument(
        "--vocab", choices=["ontology", "fit", "fit-tokens"], default="ontology",
        help=(
            "'ontology' (default): the closed 8/10-type kg_agentic_extraction.models.ontology "
            "enums — correct for data/training/clean (the legacy, closed-ontology extractor). "
            "'fit': build an open-vocabulary FeatureVocab from the training split itself "
            "(top-N most frequent types + OOV) — use this for clusters synced from GCS via "
            "sync_from_gcs.py, since per ADR 0004 the current kg_agentic_extraction pipeline "
            "is open-vocabulary and the ontology enums would mostly miss. "
            "'fit-tokens': as 'fit', but relation types are encoded by their words "
            "(multi-hot over the top --vocab-max-relation-types words) — the right choice "
            "for open-vocabulary graphs, where whole relation types are too sparse to one-hot."
        ),
    )
    parser.add_argument("--vocab-max-entity-types", type=int, default=32)
    parser.add_argument("--vocab-max-relation-types", type=int, default=48, help="Whole types for 'fit', words for 'fit-tokens'.")
    parser.add_argument("--no-element-heads", action="store_true", help="Graph-level heads only (the original model).")
    parser.add_argument("--element-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    _configure_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vocab in ("fit", "fit-tokens"):
        splits = json.loads(Path(args.splits_path).read_text())
        train_cluster_ids = set(splits["splits"]["train"])
        fit_kgs = [kg for _, kg in _load_clean_kgs(args.clean_dir, train_cluster_ids)]
        if args.data == "files" and Path(args.corrupted_dir).exists():
            for path in sorted(Path(args.corrupted_dir).glob("*.json")):
                record = json.loads(path.read_text())
                if record["cluster_idx"] in train_cluster_ids:
                    fit_kgs.append(record["knowledge_graph"])
        vocab = FeatureVocab.fit(
            fit_kgs,
            max_entity_types=args.vocab_max_entity_types,
            max_relation_types=args.vocab_max_relation_types,
            relation_mode="token" if args.vocab == "fit-tokens" else "type",
        )
        logger.info(
            "Fit open-vocabulary FeatureVocab from %d train-split KGs: %d entity types, %d relation types (incl. OOV).",
            len(fit_kgs), vocab.num_entity_types, vocab.num_relation_types,
        )
    else:
        vocab = FeatureVocab()
    vocab.save(output_dir / "feature_vocab.json")

    if args.data == "on-the-fly":
        train_ds = OnTheFlyCorruptionDataset("train", args.clean_dir, args.splits_path, vocab, seed=args.seed, resample=True)
        val_ds = OnTheFlyCorruptionDataset("val", args.clean_dir, args.splits_path, vocab, seed=args.seed)
    else:
        train_ds = KGValidationDataset("train", args.clean_dir, args.corrupted_dir, args.splits_path, vocab)
        val_ds = KGValidationDataset("val", args.clean_dir, args.corrupted_dir, args.splits_path, vocab)
    logger.info("Loaded %d train / %d val samples.", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = GraphTransformerValidator(
        node_feature_dim=vocab.node_feature_dim,
        edge_feature_dim=vocab.edge_feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        element_heads=not args.no_element_heads,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []
    best_ckpt_path = output_dir / "best_model.pt"

    logger.info("Starting training: %d epochs, patience=%d, %d train batches/epoch.", args.epochs, args.patience, len(train_loader))
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.time()
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        train_loss, train_per_head = run_epoch(model, train_loader, criterion, optimizer, args.element_loss_weight)
        val_loss, val_per_head = run_epoch(model, val_loader, criterion, None, args.element_loss_weight)
        epoch_dt = time.time() - epoch_t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_per_head": train_per_head, "val_per_head": val_per_head, "seconds": epoch_dt})
        logger.info(
            "Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f%s  (%.1fs)",
            epoch, args.epochs, train_loss, val_loss,
            f"  (val node={val_per_head['element_node']:.4f} edge={val_per_head['element_edge']:.4f})" if model.element_heads else "",
            epoch_dt,
        )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "node_feature_dim": vocab.node_feature_dim,
                        "edge_feature_dim": vocab.edge_feature_dim,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "num_heads": args.num_heads,
                        "dropout": args.dropout,
                        "head_names": HEAD_NAMES_EXTENDED,
                        "element_heads": model.element_heads,
                    },
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                logger.info("Early stopping at epoch %d (no val improvement for %d epochs).", epoch, args.patience)
                break

    total_dt = time.time() - t0
    best_epoch = torch.load(best_ckpt_path, weights_only=False)["epoch"]
    logger.info("Training done in %.1fs (%.1fs/epoch avg). Best val_loss=%.4f at epoch %d.", total_dt, total_dt / epoch, best_val_loss, best_epoch)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2))
    maybe_upload_to_gcs(best_ckpt_path, args.gcs_bucket, args.gcs_prefix)

    logger.info("Best checkpoint: %s", best_ckpt_path)


if __name__ == "__main__":
    main()
