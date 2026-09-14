"""
Reading the grader's unconstrained completion.

Nothing constrains this call any more, so `from_completion` is the only thing
standing between the model's prose and the loop's stopping decision. Two
properties matter, and they pull in opposite directions:

- A bare `CONVERGED` ends the run, however the model dressed it up.
- Nothing *else* ends the run — least of all a report that happens to use the
  word — because a false convergence accepts a graph the grader was still
  complaining about, and there is no round after that.
"""

from __future__ import annotations

import pytest

from kg_agentic_extraction.models.grading import CONVERGED_SENTINEL, SimpleSchemaGraderReport

REPORT = """One relation is mistyped.

## High severity

- **WRONG_TYPE** `e1|LOCATED_IN|Ohio` — the quote says Doe is the governor.
  - **Fix:** retype to `GOVERNOR_OF`.
"""


def parse(text: str) -> SimpleSchemaGraderReport:
    return SimpleSchemaGraderReport.from_completion(text)


# ── The sentinel ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "completion",
    [
        CONVERGED_SENTINEL,
        "  CONVERGED\n\n",
        "converged",
        "**CONVERGED**",
        "## CONVERGED",
        "`CONVERGED`",
        "CONVERGED.",
    ],
    ids=["bare", "padded", "lowercase", "bold", "heading", "code", "period"],
)
def test_the_sentinel_ends_the_run_however_it_is_dressed(completion):
    """The model was told to answer with one word; it will not always obey exactly."""
    report = parse(completion)

    assert report.converged is True
    assert report.issues_markdown == ""


def test_a_fenced_sentinel_still_converges():
    assert parse("```\nCONVERGED\n```").converged is True


def test_a_thinking_block_before_the_sentinel_is_discarded():
    """A reasoning model's scratch work is not part of the answer."""
    assert parse("<think>Checked every quote. All fine.</think>\nCONVERGED").converged is True


# ── Everything that must *not* converge ───────────────────────────────


def test_a_report_mentioning_convergence_does_not_converge():
    """
    The sentinel is matched against the whole body, never as a substring. A
    grader is allowed to discuss convergence in prose while still complaining.
    """
    completion = "## High severity\n\n- The graph has not CONVERGED on one name for Doe.\n"
    report = parse(completion)

    assert report.converged is False
    assert "not CONVERGED" in report.issues_markdown


def test_a_report_is_kept_verbatim():
    report = parse(REPORT)

    assert report.converged is False
    assert report.issues_markdown == REPORT.strip()


def test_a_fenced_report_keeps_its_content():
    """Models fence Markdown even when told not to; the fence is not the report."""
    report = parse(f"```markdown\n{REPORT}```")

    assert report.converged is False
    assert report.issues_markdown.startswith("One relation is mistyped.")
    assert "```" not in report.issues_markdown


def test_a_report_after_a_thinking_block_survives():
    report = parse(f"<think>weighing severity</think>\n{REPORT}")

    assert report.converged is False
    assert "**Fix:** retype to `GOVERNOR_OF`." in report.issues_markdown


# ── The empty answer ──────────────────────────────────────────────────


@pytest.mark.parametrize("completion", ["", "   \n\t ", "<think>nothing to add</think>"])
def test_an_empty_answer_converges(completion):
    """
    Not because empty means faithful, but because it cannot mean anything else
    here: an empty report sends the extractor into a repair round with no
    instructions, and would do so again every round until the iteration cap.

    The client raises on a genuinely empty completion before this is reached
    (see `LangChainToolLoopMixin.complete`), so this is the second net.
    """
    report = parse(completion)

    assert report.converged is True
    assert report.issues_markdown == ""
