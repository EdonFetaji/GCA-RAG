"""
train_gnn.py — train GraphTransformerValidator on the extended 6-head scheme.

Run:
    python -m gnn_validator.train
    python -m gnn_validator.train --epochs 200 --hidden-dim 96 --num-layers 4
    python -m gnn_validator.train --gcs-bucket my-bucket --gcs-prefix gnn_checkpoints

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

from gnn_validator.data import KGValidationDataset
from gnn_validator.feature_vocab import FeatureVocab
from gnn_validator.model import GraphTransformerValidator
from kg_dataset.corruption import HEAD_NAMES_EXTENDED

logger = logging.getLogger("train_gnn")


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)])


def run_epoch(model, loader, criterion, optimizer=None) -> tuple[float, dict[str, float]]:
    """One pass over `loader`. Trains if `optimizer` is given, else eval-only (no grad)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_batches = 0
    per_head_loss = {h: 0.0 for h in HEAD_NAMES_EXTENDED}

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in loader:
            logits = model.forward_logits(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss_per_head = criterion(logits, batch.y)  # [batch, num_heads], reduction='none'
            loss = loss_per_head.mean()

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
        "--vocab", choices=["ontology", "fit"], default="ontology",
        help=(
            "'ontology' (default): the closed 8/10-type kg_agentic_extraction.models.ontology "
            "enums — correct for data/training/clean (the legacy, closed-ontology extractor). "
            "'fit': build an open-vocabulary FeatureVocab from the training split itself "
            "(top-N most frequent types + OOV) — use this for clusters synced from GCS via "
            "sync_from_gcs.py, since per ADR 0004 the current kg_agentic_extraction pipeline "
            "is open-vocabulary and the ontology enums would mostly miss."
        ),
    )
    parser.add_argument("--vocab-max-entity-types", type=int, default=32)
    parser.add_argument("--vocab-max-relation-types", type=int, default=48)
    args = parser.parse_args()

    _configure_logging()
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.vocab == "fit":
        splits = json.loads(Path(args.splits_path).read_text())
        train_cluster_ids = set(splits["splits"]["train"])
        fit_kgs = []
        for path in sorted(Path(args.clean_dir).glob("*.json")):
            if path.stem.isdigit() and int(path.stem) in train_cluster_ids:
                fit_kgs.append(json.loads(path.read_text())["extraction"]["knowledge_graph"])
        for path in sorted(Path(args.corrupted_dir).glob("*.json")) if Path(args.corrupted_dir).exists() else []:
            record = json.loads(path.read_text())
            if record["cluster_idx"] in train_cluster_ids:
                fit_kgs.append(record["knowledge_graph"])
        vocab = FeatureVocab.fit(fit_kgs, max_entity_types=args.vocab_max_entity_types, max_relation_types=args.vocab_max_relation_types)
        logger.info(
            "Fit open-vocabulary FeatureVocab from %d train-split KGs: %d entity types, %d relation types (incl. OOV).",
            len(fit_kgs), vocab.num_entity_types, vocab.num_relation_types,
        )
    else:
        vocab = FeatureVocab()
    vocab.save(output_dir / "feature_vocab.json")

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
        train_loss, train_per_head = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_per_head = run_epoch(model, val_loader, criterion, optimizer=None)
        epoch_dt = time.time() - epoch_t0

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "train_per_head": train_per_head, "val_per_head": val_per_head, "seconds": epoch_dt})
        logger.info("Epoch %3d/%d — train_loss=%.4f  val_loss=%.4f  (%.1fs)", epoch, args.epochs, train_loss, val_loss, epoch_dt)

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
