"""The validator port, and how its report is put into words for the other agents."""

from __future__ import annotations

from typing import Protocol

from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph
from kg_agentic_extraction.models.validation import ValidationReport


class GraphValidator(Protocol):
    """Scores a candidate graph's structure. Reads the graph only — no documents."""

    def validate(self, graph: KnowledgeGraph, *, iteration: int) -> ValidationReport: ...


def grader_hints(report: ValidationReport | None) -> list[dict[str, object]]:
    """The flagged elements as template context for the grader's prompt."""
    if report is None:
        return []
    return [
        {
            "kind": f.kind,
            "element_id": f.element_id,
            "description": f.description,
            "score": round(f.score, 2),
        }
        for f in report.flagged
    ]


def repair_notes(report: ValidationReport, iteration: int) -> str:
    """
    Repair instructions for a vetoed round. Flags are framed as things to
    re-verify, not known errors: the validator never read the documents.
    """
    lines = [
        f"# Grader report — iteration {iteration}",
        "",
        "The grader found no issues, but an automatic structural check scored this graph as "
        f"likely inconsistent (consistency {report.consistency:.2f}). Re-verify each element "
        "below against the documents. Keep it if the documents support it; remove or correct "
        "it if they do not. An entity flagged here may also be missing edges the documents state.",
        "",
    ]
    for f in report.flagged:
        lines.append(f"- **{f.kind}** `{f.element_id}` — {f.description} (suspicion {f.score:.2f})")
    if not report.flagged:
        lines.append(
            "- No single element stood out; re-check the graph's coverage and connectivity."
        )
    return "\n".join(lines)
