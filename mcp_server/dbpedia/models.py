"""
Tool response contracts.

FastMCP derives each tool's output JSON schema from its return annotation, so
these models are what the calling model actually sees described. Field
descriptions are therefore written for the model, not for us.

Every result model carries a `note`. Tools never raise: an unreachable endpoint
or a malformed argument comes back as an empty result plus an explanation,
because a tool that raises stalls the agent loop mid-graph, whereas an empty
result with a reason lets the model try a different tool.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class _ToolResult(BaseModel):
    """Base for every tool response."""

    note: str = Field(
        "",
        description=(
            "Empty on success. Otherwise explains why the result is empty or "
            "partial — a failed lookup service, a rejected argument."
        ),
    )


# ── Entity tools ──────────────────────────────────────────────────────


class SpotlightCandidate(BaseModel):
    """One resource DBpedia Spotlight proposes for a mention."""

    uri: str = Field(..., description="Full DBpedia resource URI.")
    label: str = Field("", description="Human-readable name of the resource.")
    surface_form: str = Field("", description="The text span Spotlight matched.")
    offset: int = Field(0, description="Character offset of the span in the context.")
    final_score: float = Field(
        0.0, description="Spotlight's overall ranking score. Higher is a better fit."
    )
    contextual_score: float = Field(
        0.0, description="How well the surrounding context supports this resource."
    )
    support: int = Field(0, description="Number of Wikipedia inlinks. A popularity prior.")
    types: list[str] = Field(default_factory=list, description="Types Spotlight reports.")


class SpotlightResult(_ToolResult):
    mention: str = Field(..., description="The mention that was being linked.")
    candidates: list[SpotlightCandidate] = Field(
        default_factory=list, description="Best first. Empty if Spotlight matched nothing."
    )


class ResourceHit(BaseModel):
    """One resource matching a label search."""

    uri: str
    label: str = ""
    comment: str = Field("", description="First sentences of the resource's abstract.")
    types: list[str] = Field(default_factory=list, description="dbo: classes of this resource.")
    ref_count: int = Field(0, description="Wikipedia inlink count. A popularity prior.")
    score: float = Field(0.0, description="Search-engine relevance score, not a probability.")
    source: str = Field("lookup", description="Which index produced this hit: lookup or sparql.")


class ResourceSearchResult(_ToolResult):
    label: str
    expected_types: list[str] = Field(
        default_factory=list, description="Normalized class URIs the search was filtered by."
    )
    results: list[ResourceHit] = Field(default_factory=list)


class PredicateUsage(BaseModel):
    """A predicate this resource actually uses, with one example value."""

    predicate: str
    label: str = ""
    example_value: str = ""


class ResourceProfile(_ToolResult):
    """Everything needed to confirm or reject a candidate resource."""

    uri: str
    found: bool = Field(False, description="False when the URI has no triples in DBpedia.")
    label: str = ""
    abstract: str = Field("", description="Truncated English abstract.")
    types: list[str] = Field(default_factory=list, description="dbo: classes, most specific first.")
    wikidata_uri: str = Field("", description="owl:sameAs link to Wikidata, when present.")
    redirects_to: str = Field(
        "", description="Set when this URI is a redirect. Ground to the target instead."
    )
    is_disambiguation: bool = Field(
        False,
        description=(
            "True when the URI is a disambiguation page. Never ground to one — "
            "search again with more context."
        ),
    )
    sample_predicates: list[PredicateUsage] = Field(
        default_factory=list, description="A sample of outgoing dbo: predicates."
    )


class ClassHit(BaseModel):
    uri: str
    label: str = ""
    comment: str = ""
    superclasses: list[str] = Field(default_factory=list)


class ClassSearchResult(_ToolResult):
    label: str
    results: list[ClassHit] = Field(default_factory=list)


# ── Relation tools ────────────────────────────────────────────────────


class PropertyHit(BaseModel):
    """One ontology property matching a relation phrase."""

    uri: str
    label: str = ""
    comment: str = ""
    domain: str = Field("", description="rdfs:domain — the class the subject should belong to.")
    range: str = Field(
        "", description="rdfs:range — the class (or datatype) the object should belong to."
    )
    kind: str = Field("object", description="object or datatype.")


class PropertySearchResult(_ToolResult):
    relation_text: str
    results: list[PropertyHit] = Field(default_factory=list)


class PropertyProfile(_ToolResult):
    uri: str
    found: bool = False
    label: str = ""
    comment: str = ""
    domain: str = ""
    range: str = ""
    kind: str = Field("", description="object, datatype, or empty if undeclared.")
    sub_property_of: list[str] = Field(default_factory=list)
    equivalent_properties: list[str] = Field(default_factory=list)
    usage_count: int = Field(
        0, description="Triples using this property, capped. 0 suggests it is unused in practice."
    )


class PredicateLink(BaseModel):
    """A predicate DBpedia actually asserts between the two resources."""

    predicate: str
    label: str = ""
    direction: str = Field(
        ...,
        description=(
            "forward = subject→object as asked; reverse = the triple exists the "
            "other way round, so the local relation may be inverted."
        ),
    )


class PredicatesBetweenResult(_ToolResult):
    subject_uri: str
    object_uri: str
    predicates: list[PredicateLink] = Field(
        default_factory=list,
        description="Empty means DBpedia asserts no direct edge between the two.",
    )
