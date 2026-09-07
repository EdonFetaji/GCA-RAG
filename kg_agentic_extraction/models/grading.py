"""
The grader's output contract.

Two shapes live here, and which one is in play is decided by
`KG_GRADER_PROMPT_VERSION` together with `GraderAgent`:

- `SimpleSchemaGraderReport` — the live one, paired with the v4 templates. Two
  flat fields: the model writes its own Markdown and sets `converged` itself.
  Sent as a schema, so the provider decodes it. `from_completion` reads the same
  two fields out of an unconstrained reply instead, for the regime where nothing
  is sent at all.
- `GraderReport` + `GraderIssue` — the v1-v3 contract. A list of typed issues
  with convergence *derived* (`not report.issues`), rendered to the `.md`
  artifact by `prompts/renderers.report_to_markdown`.

What both arrangements have in common is the part that matters: convergence is
read off a Pydantic object, never by pattern-matching prose that an LLM happened
to format a particular way.
"""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, Field

#: Only for `from_completion`, below — the unconstrained regime, which is not
#: what the grader runs today. What the grader answers with, alone, when it
#: finds nothing to fix: a single word rather than a JSON field, because in that
#: arrangement nothing constrains the call at all — see `TextLLMClient`.
CONVERGED_SENTINEL = "CONVERGED"

#: A reasoning model's scratch work, which is not part of the answer.
_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE)

#: The whole body wrapped in one fence — some models fence their Markdown even
#: when told not to. Only stripped when it opens the body and closes it.
_WHOLE_FENCE = re.compile(r"\A```[^\n]*\n(?P<body>.*?)\n?```\Z", re.DOTALL)


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


class SimpleSchemaGraderReport(BaseModel):
    """
    The grader's contract when it writes its own Markdown — the v4 templates.

    Why a second model rather than an edit to `GraderReport`: the graders worth
    running on a free tier decode `json_schema` mode *strictly*, and
    `GraderReport` is a list of nested objects carrying two enums. Two flat
    fields is the shape that survives such a decoder; the prose the model would
    have put in `description` and `suggested_fix` goes into `issues_markdown`
    instead, which is what the extractor's repair prompt consumed all along.

    Note the one real behavioural difference from `GraderReport`, whose
    `converged` is *derived* (`not self.issues`): here it is a field the model
    sets for itself. There is no empty list left to infer it from, so the v4
    system prompt has to state the rule explicitly. If the model contradicts
    itself — `converged=True` alongside a non-empty report — `converged` wins and
    the loop ends; a grader that cannot say whether it is done is not worth
    another full refinement round.

    `GraderReport` is deliberately kept: it and `report_to_markdown` are how the
    v1-v3 grading regime is reproduced, and swapping `GraderAgent.output_schema`
    back to it (with `KG_GRADER_PROMPT_VERSION`) is all that takes.

    `from_completion` below reads these same two fields out of an unconstrained
    reply. It is not on the live path — the agent is decoded into, not parsed
    into — and exists for the regime where the report is too long to survive
    travelling as an escaped JSON string.
    """

    converged: bool = Field(
        ...,
        description="True when nothing is left to fix and the graph can leave the loop.",
    )
    issues_markdown: str = Field(
        "",
        description="The Markdown issue report, written by the model. Empty when converged.",
    )

    @classmethod
    def from_completion(cls, text: str) -> SimpleSchemaGraderReport:
        """
        Read one unconstrained completion as a report.

        Unused while the grader runs under a schema; this is the entry point for
        the regime where it does not, paired with `Agent.run_completion`.

        There, the grader answers with Markdown, or with `CONVERGED` and nothing else.
        That sentinel is matched against the *entire* normalized body and never
        as a substring: a report is allowed to discuss convergence in prose, and
        a run must not end because it did.

        Everything ambiguous therefore resolves to *not* converged, which costs
        one more refinement round. The opposite mistake accepts a graph the
        grader was still complaining about, and there is no round after that.
        """
        body = _THINK_BLOCK.sub("", text).strip()
        fenced = _WHOLE_FENCE.match(body)
        if fenced:
            body = fenced.group("body").strip()

        # Emphasis and headings are the model dressing the sentinel up —
        # `**CONVERGED**` and `## CONVERGED` are the same answer as `CONVERGED`.
        signal = body.strip("*_#`. \t\n").upper()

        if signal == CONVERGED_SENTINEL:
            return cls(converged=True, issues_markdown="")
        if not body:
            # Nothing to instruct the extractor with. Sending it into a repair
            # round with an empty prompt costs a full pass and changes nothing,
            # every round until the iteration cap.
            return cls(converged=True, issues_markdown="")
        return cls(converged=False, issues_markdown=body)
