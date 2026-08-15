"""
The grader's output contract.

The grader is asked for Markdown, but Markdown is a *rendering*, not the
contract. The agent emits this typed report; `prompts/renderers.py` turns it
into the `.md` artifact that gets saved and fed back to the extractor.

Keeping the model authoritative is what makes the loop testable: convergence
is `not report.issues`, evaluated on a Pydantic object, never by pattern-matching
prose that an LLM happened to format a particular way.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class IssueType(StrEnum):
    """What kind of defect the grader found."""

    HALLUCINATION = "HALLUCINATION"
    MISSING_ENTITY = "MISSING_ENTITY"
    MISSING_RELATION = "MISSING_RELATION"
    CONTRADICTION = "CONTRADICTION"
    WEAK_EVIDENCE = "WEAK_EVIDENCE"
    WRONG_TYPE = "WRONG_TYPE"
    DANGLING_REFERENCE = "DANGLING_REFERENCE"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class GraderIssue(BaseModel):
    """
    One defect, addressed to a specific graph element where possible.

    `suggested_fix` is separate from `description` because the extractor's
    repair prompt renders only the actionable half — describing the problem is
    for the human reading the .md, instructing the fix is for the next round.
    """

    issue_type: IssueType
    severity: Severity = Severity.MEDIUM
    element_id: str | None = Field(
        None, description="Entity id, or 'source|RELATION|target' for an edge."
    )
    description: str = Field(..., min_length=1, description="What is wrong.")
    suggested_fix: str = Field("", description="What the extractor should do about it.")


class GraderReport(BaseModel):
    """The verdict on one candidate graph."""

    issues: list[GraderIssue] = Field(default_factory=list)
    summary: str = Field("", description="One-paragraph overall assessment.")

    @property
    def converged(self) -> bool:
        """True when the grader found nothing left to fix."""
        return not self.issues

    def issues_at_or_above(self, severity: Severity) -> list[GraderIssue]:
        """
        Issues filtered by severity.

        Exists so a future run policy can converge on "no high-severity issues"
        instead of "no issues at all" without changing the loop's routing code.
        """
        order = {Severity.LOW: 0, Severity.MEDIUM: 1, Severity.HIGH: 2}
        threshold = order[severity]
        return [i for i in self.issues if order[i.severity] >= threshold]

    def count_by_type(self) -> dict[IssueType, int]:
        counts: dict[IssueType, int] = {}
        for issue in self.issues:
            counts[issue.issue_type] = counts.get(issue.issue_type, 0) + 1
        return counts
