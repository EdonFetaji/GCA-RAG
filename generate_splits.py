"""
Track 2.4 — Train/val/test split.

Splits at the *cluster* level (not the corrupted-variant level) to avoid
leakage — a clean cluster and every corrupted variant derived from it
must end up in the same split, or the model could see near-duplicate
graphs across train and val/test.

Run:
    python generate_splits.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger("generate_splits")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_cluster_indices(clean_dir: Path) -> list[int]:
    indices = []
    for path in clean_dir.glob("*.json"):
        if path.stem.isdigit():
            indices.append(int(path.stem))
    return sorted(indices)


def split_clusters(
    cluster_indices: list[int],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> dict[str, list[int]]:
    """
    70/15/15 split by default. test_frac is whatever's left after
    train_frac + val_frac, so the three always sum to the full set even
    with rounding.
    """
    if not (0 < train_frac < 1) or not (0 <= val_frac < 1) or train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be < 1, and both must be valid fractions.")

    rng = random.Random(seed)
    shuffled = cluster_indices[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * train_frac)
    n_val = round(n * val_frac)
    # Whatever's left goes to test, so small/odd-sized sets don't silently
    # drop a cluster to rounding.
    n_val = min(n_val, n - n_train)

    train = sorted(shuffled[:n_train])
    val = sorted(shuffled[n_train:n_train + n_val])
    test = sorted(shuffled[n_train + n_val:])

    return {"train": train, "val": val, "test": test}


def main():
    parser = argparse.ArgumentParser(description="Cluster-level train/val/test split (Track 2.4)")
    parser.add_argument("--clean-dir", type=str, default="data/training/clean")
    parser.add_argument("--output", type=str, default="data/training/splits.json")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_logging()

    clean_dir = Path(args.clean_dir)
    if not clean_dir.exists():
        logger.error("Clean KG directory not found: %s (run generate_training_data.py first)", clean_dir)
        sys.exit(1)

    cluster_indices = load_cluster_indices(clean_dir)
    if not cluster_indices:
        logger.error("No clean KGs found in %s", clean_dir)
        sys.exit(1)

    splits = split_clusters(cluster_indices, args.train_frac, args.val_frac, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "train_frac": args.train_frac,
                "val_frac": args.val_frac,
                "total_clusters": len(cluster_indices),
                "splits": splits,
            },
            f,
            indent=2,
        )

    n = len(cluster_indices)
    logger.info(
        "Split %d clusters -> train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%). Saved to %s",
        n,
        len(splits["train"]), 100 * len(splits["train"]) / n,
        len(splits["val"]), 100 * len(splits["val"]) / n,
        len(splits["test"]), 100 * len(splits["test"]) / n,
        output_path,
    )


if __name__ == "__main__":
    main()
