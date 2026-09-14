"""
Track 2.5 — Dataset sanity checks.

Runs after generate_training_data.py, generate_corruptions.py, and
generate_splits.py have all produced output. Checks:

1. Class balance — are corruption types represented evenly? (They won't
   be perfectly even: some clean KGs are too small/sparse for a given
   corruption type to apply, e.g. no relations means contradictions/
   fragmentation/relation_type_swap all get skipped for that cluster.
   This reports the skew rather than assuming it's a bug.)
2. Structural validity — every corrupted graph has >=1 entity (never
   reduced to a degenerate empty graph) and every relation references an
   entity id that actually exists in that graph.
3. A spot-check sample — a handful of clean-vs-corrupted pairs printed
   with enough detail (entity/relation counts, what changed) to eyeball
   whether corruptions look realistic rather than degenerate.
4. A summary report — counts per split x corruption type, saved to both
   JSON (for other scripts/Track 3 to consume) and Markdown (for a human
   to read), per the work plan's "short summary report (counts per
   split, per corruption type)".

Run:
    python check_dataset.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger("check_dataset")


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_splits(splits_path: Path) -> dict[int, str]:
    """Return {cluster_idx: split_name}."""
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cluster_to_split = {}
    for split_name, indices in data["splits"].items():
        for idx in indices:
            cluster_to_split[idx] = split_name
    return cluster_to_split


def load_clean_kgs(clean_dir: Path) -> dict[int, dict]:
    kgs = {}
    for path in clean_dir.glob("*.json"):
        if path.stem.isdigit():
            with open(path, "r", encoding="utf-8") as f:
                kgs[int(path.stem)] = json.load(f)
    return kgs


def load_corrupted_variants(corrupted_dir: Path) -> list[dict]:
    variants = []
    for path in corrupted_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["_path"] = str(path)
        variants.append(data)
    return variants


def check_structural_validity(clean_kgs: dict[int, dict], variants: list[dict]) -> list[str]:
    """Returns a list of problem descriptions (empty = all clean)."""
    problems = []

    for idx, clean_data in clean_kgs.items():
        kg = clean_data["extraction"]["knowledge_graph"]
        if len(kg["entities"]) == 0:
            problems.append(f"clean cluster {idx}: 0 entities")
        ids = {e["id"] for e in kg["entities"]}
        for r in kg["relations"]:
            if r["source"] not in ids or r["target"] not in ids:
                problems.append(f"clean cluster {idx}: relation references missing entity id")

    for v in variants:
        kg = v["knowledge_graph"]
        label = f"cluster {v['cluster_idx']} / {v['corruption_type']} / sev={v['severity']}"
        if len(kg["entities"]) == 0:
            problems.append(f"{label}: reduced to 0 entities (degenerate!)")
        ids = {e["id"] for e in kg["entities"]}
        for r in kg["relations"]:
            if r["source"] not in ids or r["target"] not in ids:
                problems.append(f"{label}: relation references missing entity id")

    return problems


def check_class_balance(variants: list[dict]) -> dict[str, int]:
    counts = Counter(v["corruption_type"] for v in variants)
    return dict(counts)


def spot_check_sample(clean_kgs: dict[int, dict], variants: list[dict], n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    eligible = [v for v in variants if v["cluster_idx"] in clean_kgs]
    sample = rng.sample(eligible, min(n, len(eligible)))

    report = []
    for v in sample:
        clean_kg = clean_kgs[v["cluster_idx"]]["extraction"]["knowledge_graph"]
        corrupted_kg = v["knowledge_graph"]
        report.append({
            "cluster_idx": v["cluster_idx"],
            "corruption_type": v["corruption_type"],
            "severity": v["severity"],
            "clean_entities": len(clean_kg["entities"]),
            "clean_relations": len(clean_kg["relations"]),
            "corrupted_entities": len(corrupted_kg["entities"]),
            "corrupted_relations": len(corrupted_kg["relations"]),
            "corruption_metadata": v["corruption_metadata"],
        })
    return report


def build_split_x_type_table(cluster_to_split: dict[int, str], variants: list[dict]) -> dict:
    table = defaultdict(lambda: defaultdict(int))
    for v in variants:
        split = cluster_to_split.get(v["cluster_idx"], "UNASSIGNED")
        table[split][v["corruption_type"]] += 1
    return {split: dict(types) for split, types in table.items()}


def write_markdown_report(
    path: Path,
    clean_count: int,
    split_sizes: dict[str, int],
    class_balance: dict[str, int],
    split_x_type: dict,
    problems: list[str],
    spot_check: list[dict],
) -> None:
    lines = ["# Training data summary report (Track 2.5)\n"]

    lines.append("## Overview\n")
    lines.append(f"- Clean KGs: {clean_count}")
    for split, size in split_sizes.items():
        lines.append(f"- {split.capitalize()} split: {size} clusters")
    lines.append("")

    lines.append("## Class balance (corrupted variants per corruption type)\n")
    lines.append("| Corruption type | Count |")
    lines.append("|---|---|")
    for ctype, count in sorted(class_balance.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {ctype} | {count} |")
    lines.append("")
    if class_balance:
        max_count = max(class_balance.values())
        skewed = [t for t, c in class_balance.items() if c < 0.8 * max_count]
        if skewed:
            lines.append(
                f"⚠️ Types below 80% of the max count (likely because many clean KGs "
                f"were too small/sparse for that corruption to apply): {', '.join(skewed)}"
            )
        else:
            lines.append("Corruption types are reasonably balanced (all within 80% of the max).")
    lines.append("")

    lines.append("## Counts per split x corruption type\n")
    lines.append("| Split | " + " | ".join(sorted(class_balance.keys())) + " |")
    lines.append("|---|" + "---|" * len(class_balance))
    for split in ("train", "val", "test"):
        row = split_x_type.get(split, {})
        lines.append(f"| {split} | " + " | ".join(str(row.get(t, 0)) for t in sorted(class_balance.keys())) + " |")
    lines.append("")

    lines.append("## Structural validity\n")
    if problems:
        lines.append(f"⚠️ {len(problems)} problem(s) found:\n")
        for p in problems[:50]:
            lines.append(f"- {p}")
        if len(problems) > 50:
            lines.append(f"- ... and {len(problems) - 50} more")
    else:
        lines.append("✓ No problems found — every corrupted graph has >=1 entity, no relation references a missing entity id.")
    lines.append("")

    lines.append("## Spot-check sample\n")
    for s in spot_check:
        lines.append(
            f"- Cluster {s['cluster_idx']}, **{s['corruption_type']}** (severity={s['severity']}): "
            f"{s['clean_entities']}e/{s['clean_relations']}r → {s['corrupted_entities']}e/{s['corrupted_relations']}r "
            f"— {s['corruption_metadata']}"
        )
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Dataset sanity checks (Track 2.5)")
    parser.add_argument("--clean-dir", type=str, default="data/training/clean")
    parser.add_argument("--corrupted-dir", type=str, default="data/training/corrupted")
    parser.add_argument("--splits-path", type=str, default="data/training/splits.json")
    parser.add_argument("--output", type=str, default="data/training/summary_report.md")
    parser.add_argument("--output-json", type=str, default="data/training/summary_report.json")
    parser.add_argument("--spot-check-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_logging()

    clean_dir = Path(args.clean_dir)
    corrupted_dir = Path(args.corrupted_dir)
    splits_path = Path(args.splits_path)

    for p, label in [(clean_dir, "clean dir"), (corrupted_dir, "corrupted dir"), (splits_path, "splits file")]:
        if not p.exists():
            logger.error("%s not found: %s — run the earlier Track 2 scripts first.", label, p)
            sys.exit(1)

    clean_kgs = load_clean_kgs(clean_dir)
    variants = load_corrupted_variants(corrupted_dir)
    cluster_to_split = load_splits(splits_path)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    split_sizes = {k: len(v) for k, v in splits_data["splits"].items()}

    logger.info("Loaded %d clean KGs, %d corrupted variants, %d clusters with a split assignment.",
                len(clean_kgs), len(variants), len(cluster_to_split))

    problems = check_structural_validity(clean_kgs, variants)
    if problems:
        logger.warning("Found %d structural problem(s) — see report for details.", len(problems))
    else:
        logger.info("Structural validity: OK — no degenerate graphs, no dangling relation references.")

    class_balance = check_class_balance(variants)
    logger.info("Class balance: %s", class_balance)

    split_x_type = build_split_x_type_table(cluster_to_split, variants)

    spot_check = spot_check_sample(clean_kgs, variants, args.spot_check_n, args.seed)
    logger.info("Spot-check sample (%d variants):", len(spot_check))
    for s in spot_check:
        logger.info(
            "  cluster %s / %s (sev=%s): %de/%dr -> %de/%dr  meta=%s",
            s["cluster_idx"], s["corruption_type"], s["severity"],
            s["clean_entities"], s["clean_relations"],
            s["corrupted_entities"], s["corrupted_relations"],
            s["corruption_metadata"],
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_markdown_report(output_path, len(clean_kgs), split_sizes, class_balance, split_x_type, problems, spot_check)

    output_json_path = Path(args.output_json)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "clean_count": len(clean_kgs),
            "split_sizes": split_sizes,
            "class_balance": class_balance,
            "split_x_type": split_x_type,
            "problem_count": len(problems),
            "problems": problems,
            "spot_check": spot_check,
        }, f, indent=2)

    logger.info("\nReports written to %s and %s", output_path, output_json_path)
    if problems:
        logger.warning("Dataset has %d structural problem(s) — review before training.", len(problems))
        sys.exit(1)


if __name__ == "__main__":
    main()
