"""
Run timing — the formatter, and the wiring that gets a duration onto the result.

The pipeline itself is stubbed with a fake compiled graph. `run_pipeline` only
ever calls `.invoke()` on what it is handed, so a graph that sleeps a known
amount is enough to prove the clock is wrapped around the right call — and it
keeps the test offline and instant.
"""

from __future__ import annotations

import json
import time

import pytest

from kg_agentic_extraction.runner import format_duration, run_pipeline


class _FakeGraph:
    """A compiled graph that sleeps, then returns a minimal final state."""

    def __init__(self, *, sleep: float = 0.0, boom: Exception | None = None) -> None:
        self._sleep = sleep
        self._boom = boom

    def invoke(self, state):  # noqa: ARG002
        time.sleep(self._sleep)
        if self._boom:
            raise self._boom
        return {"iteration": 2, "converged": True, "errors": []}


def test_result_carries_the_elapsed_time():
    result = run_pipeline(["doc"], graph=_FakeGraph(sleep=0.05))
    assert result.elapsed_seconds >= 0.05
    # Generous ceiling: this asserts the clock measures the call rather than
    # something process-wide, not that the machine is fast.
    assert result.elapsed_seconds < 5


def test_elapsed_time_reaches_the_json():
    payload = json.loads(run_pipeline(["doc"], graph=_FakeGraph()).to_json())
    assert "elapsed_seconds" in payload
    assert isinstance(payload["elapsed_seconds"], float)


def test_a_failed_run_is_still_timed(caplog):
    """The run whose duration you most want is the one that died on a timeout."""
    with caplog.at_level("INFO"), pytest.raises(TimeoutError):
        run_pipeline(["doc"], graph=_FakeGraph(sleep=0.05, boom=TimeoutError("provider")))
    assert any("pipeline failed after" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0.0s"),
        (4.24, "4.2s"),
        (59.9, "59.9s"),
        (60.0, "1m 00.0s"),
        (67.4, "1m 07.4s"),
        (3599.0, "59m 59.0s"),
        (3600.0, "1h 00m 00s"),
        (3727.5, "1h 02m 07s"),
    ],
)
def test_duration_formatting(seconds, expected):
    """Zero-padded so a column of runs stays aligned when you compare providers."""
    assert format_duration(seconds) == expected
