"""
Entity-type constants for extractor_agent.

This used to be its own hardcoded set that had quietly drifted from a second
enum elsewhere in the repo (two ontologies, same project). Re-exporting keeps
one source of truth; that source is now
kg_agentic_extraction/models/ontology.py — see docs/adr/0002-agentic-pipeline.md,
which supersedes ADR 0001.
"""
from kg_agentic_extraction.models.ontology import EntityType

VALID_ENTITY_TYPES = {t.value for t in EntityType}
