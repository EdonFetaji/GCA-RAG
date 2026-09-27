"""
validation — structural checks on a candidate graph, run inside the loop.

`GNNValidator` (validation/gnn.py) is not imported here, so runs with the
validator off never import torch.
"""

from kg_agentic_extraction.validation.base import GraphValidator, grader_hints, repair_notes

__all__ = ["GraphValidator", "grader_hints", "repair_notes"]
