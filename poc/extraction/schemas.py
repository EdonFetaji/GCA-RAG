"""
Pydantic schemas for the KG extraction pipeline.

Covers all five extraction methods:
1. Ontology-constrained extraction
2. Evidence-traced extraction (with verbatim quotes)
3. Two-agent finder/grader system
4. Multi-document evidence accumulation
5. Full pipeline combining all methods
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Ontology Definitions ────────────────────────────────────────────────────


class EntityType(str, Enum):
    """
    Canonical entity ontology for the whole repo.

    This used to diverge from extractor_agent/constants.py's
    {PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, OTHER} set. Reconciled
    here (see docs/adr/0001-canonical-extraction-path.md): the original
    DISEASE/TREATMENT/SYMPTOM/METRIC types were vestigial for a project
    targeting Multi-News (general news, not a medical corpus), so they're
    dropped in favor of PRODUCT/OTHER from extractor_agent's simpler set.
    extractor_agent/constants.py now imports VALID_ENTITY_TYPES from here
    instead of keeping its own separate list, so this is the one ontology
    used everywhere extraction happens.
    """
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION"
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"
    DATE = "DATE"
    PRODUCT = "PRODUCT"
    OTHER = "OTHER"


class RelationType(str, Enum):
    LOCATED_IN = "LOCATED_IN"
    AFFILIATED_WITH = "AFFILIATED_WITH"
    ANNOUNCED = "ANNOUNCED"
    ACQUIRED = "ACQUIRED"
    PARTICIPATED_IN = "PARTICIPATED_IN"
    RESULTED_IN = "RESULTED_IN"
    CAUSES = "CAUSES"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    RELATED_TO = "RELATED_TO"


class OntologyConfig(BaseModel):
    """User-defined ontology that constrains what the AI is allowed to extract."""
    entity_types: list[str] = Field(
        default=[t.value for t in EntityType],
        description="Allowed entity types the AI can extract.",
    )
    relation_types: list[str] = Field(
        default=[r.value for r in RelationType],
        description="Allowed relation types the AI can extract.",
    )
    domain_context: str = Field(
        default="general news articles",
        description="Short domain description to guide extraction focus.",
    )


# ── Core KG Data Models ─────────────────────────────────────────────────────


class EvidenceSpan(BaseModel):
    """A verbatim quote from a source document that supports a fact."""
    document_index: int = Field(..., description="Index of the source document (0-based).")
    quote: str = Field(..., description="Exact verbatim sentence from the document.")
    section: Optional[str] = Field(None, description="Section of the document (if detected).")


class Entity(BaseModel):
    """An entity extracted from the document cluster."""
    id: str
    name: str
    type: str
    document_frequency: int = Field(1, description="How many documents mention this entity.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Verbatim quotes proving this entity exists in source documents.",
    )


class Relation(BaseModel):
    """A relation (edge) between two entities."""
    source: str = Field(..., description="Source entity id.")
    target: str = Field(..., description="Target entity id.")
    relation_type: str
    support_count: int = Field(1, description="How many documents support this relation.")
    source_documents: list[int] = Field(default_factory=list)
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    evidence: list[EvidenceSpan] = Field(
        default_factory=list,
        description="Verbatim quotes proving this relation.",
    )


class ReasoningPath(BaseModel):
    """A chain of relations forming a logical reasoning path (A→B→C)."""
    path: list[str] = Field(..., description="Ordered list of entity ids forming the chain.")
    description: str = Field(..., description="Natural language description of the inference.")
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class KnowledgeGraph(BaseModel):
    """The complete knowledge graph extracted from a document cluster."""
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    reasoning_paths: list[ReasoningPath] = Field(default_factory=list)


# ── Grader Models (Two-Agent System) ────────────────────────────────────────


class GraderIssue(BaseModel):
    """A single issue found by the Grader agent."""
    issue_type: str = Field(
        ...,
        description="Type of issue: HALLUCINATION | MISSING_ENTITY | MISSING_RELATION | CONTRADICTION | WEAK_EVIDENCE | WRONG_TYPE",
    )
    element_id: Optional[str] = Field(None, description="ID of the entity or relation with the issue.")
    description: str = Field(..., description="What is wrong and how to fix it.")
    severity: str = Field("medium", description="low | medium | high")


class GraderReport(BaseModel):
    """Full report produced by the Grader agent."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score.")
    issues: list[GraderIssue] = Field(default_factory=list)
    passes_threshold: bool = Field(False, description="Whether the KG is good enough.")
    feedback_summary: str = Field("", description="Concise summary for the Finder to act on.")


# ── API Request / Response Models ────────────────────────────────────────────


class ExtractionRequest(BaseModel):
    """Request body for all extraction endpoints."""
    dataset: str = Field(default="multi_news", description="HuggingFace dataset name.")
    cluster_num: int = Field(default=0, ge=0, description="Cluster index to extract.")
    ontology: Optional[OntologyConfig] = Field(
        default=None,
        description="Custom ontology. If None, a sensible default is used.",
    )
    max_grader_iterations: int = Field(
        default=2, ge=0, le=5,
        description="Max finder↔grader loops (0 = no grading).",
    )
    grader_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Minimum grader score to accept the KG.",
    )


class ExtractionResponse(BaseModel):
    """Response returned from extraction endpoints."""
    method: str = Field(..., description="Which extraction method was used.")
    cluster_index: int
    num_documents: int
    knowledge_graph: KnowledgeGraph
    grader_reports: list[GraderReport] = Field(default_factory=list)
    iterations_used: int = Field(0)
    final_score: Optional[float] = None
    metadata: dict = Field(default_factory=dict)
