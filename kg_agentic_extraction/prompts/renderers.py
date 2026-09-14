"""
Model → text renderers.

Separated from `registry.py` because these render *our* objects into text,
whereas the registry renders text templates for the *model*. The grader's
Markdown report is produced here: the agent returns a typed `GraderReport`, and
this module is the only place that decides what the `.md` artifact looks like.
"""

from __future__ import annotations

from kg_agentic_extraction.models.grading import GraderIssue, GraderReport, Severity
from kg_agentic_extraction.models.knowledge_graph import KnowledgeGraph

_SEVERITY_MARK = {Severity.HIGH: "🔴", Severity.MEDIUM: "🟡", Severity.LOW: "⚪"}


def report_to_markdown(report: GraderReport, *, iteration: int | None = None) -> str:
    """
    Render a `GraderReport` as the Markdown issue report.

    This is both the human-facing artifact and the text handed back to the
    extractor for repair, so it stays terse and instruction-shaped — no praise,
    no restating what the graph got right.
    """
    heading = "# Grader report" + (f" — iteration {iteration}" if iteration is not None else "")
    lines = [heading, ""]

    if report.summary:
        lines += [report.summary.strip(), ""]

    if report.converged:
        lines += ["## Issues", "", "**None.** The graph is faithful to the source documents.", ""]
        return "\n".join(lines)

    counts = report.count_by_type()
    lines += [
        f"## Issues ({len(report.issues)})",
        "",
        "| Type | Count |",
        "|---|---|",
        *(f"| {t.value} | {n} |" for t, n in sorted(counts.items(), key=lambda kv: kv[0].value)),
        "",
    ]

    for severity in (Severity.HIGH, Severity.MEDIUM, Severity.LOW):
        bucket = [i for i in report.issues if i.severity is severity]
        if not bucket:
            continue
        lines += [f"### {_SEVERITY_MARK[severity]} {severity.value.title()} severity", ""]
        lines += [_issue_to_markdown(issue) for issue in bucket]
        lines.append("")

    return "\n".join(lines)


def _issue_to_markdown(issue: GraderIssue) -> str:
    target = f" `{issue.element_id}`" if issue.element_id else ""
    fix = f"\n  - **Fix:** {issue.suggested_fix.strip()}" if issue.suggested_fix.strip() else ""
    return f"- **{issue.issue_type.value}**{target} — {issue.description.strip()}{fix}"


def format_documents(documents: list[str], *, limit: int | None = None) -> str:
    """
    Number the source documents so the model can cite them by index.

    Evidence spans reference documents by 0-based index, so the headers written
    here define what `EvidenceSpan.document_index` means — keep the two in sync.
    """
    selected = documents if limit is None else documents[:limit]
    return "\n\n---\n\n".join(
        f"[DOCUMENT {i}]\n{doc.strip()}" for i, doc in enumerate(selected) if doc.strip()
    )


def graph_to_json(graph: KnowledgeGraph, *, indent: int = 2) -> str:
    """Serialize a graph for embedding in a prompt."""
    return graph.model_dump_json(indent=indent)
