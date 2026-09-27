"""
Shared type aliases for the graph layer.

Kept in its own module so `nodes/` and `graph.py` agree on the node signature
without importing each other.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from kg_agentic_extraction.state import PipelineState

#: A LangGraph node: takes the whole state, returns only the keys it changed.
type NodeFn = Callable[[PipelineState], PipelineState]

#: What the loop's conditional edge may decide.
type LoopDecision = Literal["refine", "ground", "end"]

#: A conditional edge: reads state, names the next branch.
type RouterFn = Callable[[PipelineState], LoopDecision]

#: Whether the validator overrules a converged grader this round.
type VetoFn = Callable[[PipelineState], bool]
