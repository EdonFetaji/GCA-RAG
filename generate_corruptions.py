"""
Track 2.2 — orchestration script.

Reads every clean KG in data/training/clean/, generates corrupted variants
via validator.corruption, and saves each to
data/training/corrupted/{cluster_idx}_{corruption_type}_{severity}.json
with the corruption type/severity/label attached — per the work plan's
"Save as ..., with the corruption type/severity as the label."

Run:
    python generate_corruptions.py
    python generate_corruptions.py --include-extra-types   # also generate
                                                             # entity_duplication /
                                                             # relation_type_swap /
                                                             # orphan_node_injection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from validator.corruption import (
    ALL_CORRUPTION_TYPES,
    DEFAULT_CORRUPTION_TYPES,
    DEFAULT_SEVERITIES,
    generate_corrupted_variants,
)

logger = logging.getLogger("generate_corruptions")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _severity_tag(severity: float) -> str:
    """0.1 -> 'sev10', 0.2 -> 'sev20' — filesystem/sort-friendly."""
    return f"sev{round(severity * 100):02d}"


def load_clean_kgs(clean_dir: Path) -> list[tuple[int, dict]]:
    """Load every {cluster_idx}.json in clean_dir, skipping non-numeric filenames
    (like _failures.jsonl or generate_training_data.log)."""
    results = []
    for path in sorted(clean_dir.glob("*.json")):
        if not path.stem.isdigit():
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results.append((int(path.stem), data))
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate corrupted KG variants (Track 2.2)")
    parser.add_argument("--clean-dir", type=str, default="data/training/clean", help="Directory of clean {cluster_idx}.json files.")
    parser.add_argument("--output-dir", type=str, default="data/training/corrupted", help="Where to write corrupted variants.")
    parser.add_argument("--severities", type=float, nargs="+", default=list(DEFAULT_SEVERITIES), help="Severity levels (default: 0.1 0.2 0.3).")
    parser.add_argument("--include-extra-types", action="store_true", help="Also generate entity_duplication/relation_type_swap/orphan_node_injection (see Track 2.3 labeling note — these aren't mapped to a SimpleGNN head yet).")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for reproducible corruption.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate variants even if the output file already exists.")
    args = parser.parse_args()

    _configure_logging()

    clean_dir = Path(args.clean_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not clean_dir.exists():
        logger.error("Clean KG directory not found: %s (run generate_training_data.py first)", clean_dir)
        sys.exit(1)

    corruption_types = ALL_CORRUPTION_TYPES if args.include_extra_types else DEFAULT_CORRUPTION_TYPES
    clean_kgs = load_clean_kgs(clean_dir)

    if not clean_kgs:
        logger.error("No clean KGs found in %s", clean_dir)
        sys.exit(1)

    logger.info(
        "Loaded %d clean KGs. Generating %d corruption types x %d severities = %d variants each.",
        len(clean_kgs), len(corruption_types), len(args.severities), len(corruption_types) * len(args.severities),
    )

    total_written, total_skipped_existing, total_skipped_noop = 0, 0, 0

    for cluster_idx, clean_data in clean_kgs:
        kg = clean_data["extraction"]["knowledge_graph"]

        variants = generate_corrupted_variants(
            kg,
            corruption_types=corruption_types,
            severities=tuple(args.severities),
            seed=f"{args.seed}-{cluster_idx}",
        )

        for variant in variants:
            filename = f"{cluster_idx}_{variant['corruption_type']}_{_severity_tag(variant['severity'])}.json"
            output_path = output_dir / filename

            if output_path.exists() and not args.overwrite:
                total_skipped_existing += 1
                continue

            if variant["corruption_metadata"].get("skipped"):
                # Nothing to corrupt (e.g. a KG with 0-1 entities) — don't
                # write a "corrupted" file that's identical to the clean one.
                total_skipped_noop += 1
                continue

            record = {
                "cluster_idx": cluster_idx,
                "corruption_type": variant["corruption_type"],
                "severity": variant["severity"],
                "label": variant["label"],
                "corruption_metadata": variant["corruption_metadata"],
                "knowledge_graph": variant["kg"],
            }
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2)
            total_written += 1

    logger.info(
        "\n%s\nDone. %d variants written, %d skipped (already existed), %d skipped (nothing to corrupt)\n%s",
        "=" * 80, total_written, total_skipped_existing, total_skipped_noop, "=" * 80,
    )


if __name__ == "__main__":
    main()
