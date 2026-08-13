"""
Entity-type constants for extractor_agent.

This used to be its own hardcoded set that had quietly drifted from
extraction/schemas.py's EntityType enum (two different ontologies in the
same repo). Re-exporting from extraction.schemas makes that the single
source of truth — see docs/adr/0001-canonical-extraction-path.md.
"""
from extraction.schemas import EntityType

VALID_ENTITY_TYPES = {t.value for t in EntityType}
