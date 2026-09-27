"""
The structural validator's output: graph-level scores and the most suspicious
elements. Advice about the graph, never a rewrite of it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FlaggedElement(BaseModel):
    """One entity or relation the validator scored as likely defective."""

    kind: Literal["entity", "relation"]
    element_id: str = Field(
        ..., description="Entity id, or 'source|RELATION_TYPE|target' for a relation."
    )
    description: str = Field(
        ..., description="Human-readable form, e.g. 'Rick Scott -[GOVERNOR_OF]-> Florida'."
    )
    score: float = Field(
        ..., ge=0.0, le=1.0, description="Validator's probability that the element is defective."
    )

    model_config = {"frozen": True}


class ValidationReport(BaseModel):
    """The validator's verdict on one candidate graph, from one loop iteration."""

    iteration: int = Field(..., ge=0)
    scores: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Graph-level head scores in [0, 1]: 'consistency' (probability the graph is "
            "clean) and one probability per defect kind."
        ),
    )
    flagged: list[FlaggedElement] = Field(
        default_factory=list, description="Most suspicious first."
    )

    @property
    def consistency(self) -> float:
        """Probability the graph is clean. 1.0 when the validator produced no score."""
        return self.scores.get("consistency", 1.0)
