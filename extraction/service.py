"""
Extraction service — implements the five KG-extraction methods.

Each public function takes raw documents + config and returns a fully
populated ``ExtractionResponse``.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from extraction.schemas import (
    Entity,
    EvidenceSpan,
    ExtractionRequest,
    ExtractionResponse,
    GraderIssue,
    GraderReport,
    KnowledgeGraph,
    OntologyConfig,
    ReasoningPath,
    Relation,
)
from utils.dataset_utils import load_single_cluster
from utils.llm_utils import call_llm
from utils.prompt_utils import (
    build_evidence_prompt,
    build_grader_prompt,
    build_ontology_prompt,
    build_reasoning_paths_prompt,
    build_refinement_prompt,
)

logger = logging.getLogger(__name__)


# ── JSON Parsing Helpers ─────────────────────────────────────────────────────


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _safe_parse_json(text: str) -> dict:
    """Parse JSON from LLM output, stripping code fences first."""
    return json.loads(_strip_fences(text))


def call_llm_json(
    prompt: str,
    *,
    max_retries: int = 3,
    _call: callable = None,
) -> dict:
    """
    Call the LLM and guarantee the response is valid JSON.

    Retry logic
    -----------
    1. Send the original prompt to the LLM.
    2. Try to parse the response as JSON.
    3. If parsing fails, send a *repair* prompt that includes the broken
       output and the exact error message, asking the LLM to fix/complete it.
    4. Repeat up to ``max_retries`` times.
    5. If all retries are exhausted, raise the last ``json.JSONDecodeError``.

    Parameters
    ----------
    prompt : str
        The original prompt to send.
    max_retries : int
        How many repair attempts to allow (default 2).
    _call : callable, optional
        Override for the underlying LLM call (useful for testing).
        Defaults to ``call_llm`` from ``utils.llm_utils``.

    Returns
    -------
    dict
        The parsed JSON object.
    """
    llm = _call or call_llm

    raw_text = llm(prompt)
    last_error: Exception | None = None

    # Attempt 0 — try the original response
    try:
        return _safe_parse_json(raw_text)
    except (json.JSONDecodeError, ValueError) as exc:
        last_error = exc
        logger.warning(
            "LLM returned invalid JSON (attempt 0): %s", exc,
        )

    # Retry loop — ask the LLM to repair its own output
    for attempt in range(1, max_retries + 1):
        repair_prompt = (
            "The following text was supposed to be valid JSON but it is "
            "malformed, truncated, or wrapped in markdown fences.\n\n"
            f"--- ERROR ---\n{last_error}\n\n"
            f"--- BROKEN OUTPUT ---\n{raw_text}\n\n"
            "Please return ONLY the corrected, complete, valid JSON. "
            "Do NOT wrap it in markdown code fences. "
            "Do NOT add any commentary. "
            "If the JSON was truncated, finish constructing the "
            "knowledge graph so that every opened bracket is closed "
            "and every relation references an existing entity id."
        )

        logger.info("JSON repair attempt %d / %d …", attempt, max_retries)
        raw_text = llm(repair_prompt)

        try:
            return _safe_parse_json(raw_text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            logger.warning(
                "LLM repair attempt %d still invalid: %s", attempt, exc,
            )

    # All retries exhausted
    raise last_error  # type: ignore[misc]


def _parse_kg(raw: dict) -> KnowledgeGraph:
    """Convert raw LLM JSON dict into a validated KnowledgeGraph model."""
    entities: list[Entity] = []
    for e in raw.get("entities", []):
        evidence = [
            EvidenceSpan(
                document_index=ev.get("document_index", 0),
                quote=ev.get("quote", ""),
                section=ev.get("section"),
            )
            for ev in e.get("evidence", [])
        ]
        entities.append(
            Entity(
                id=e["id"],
                name=e["name"],
                type=e["type"],
                document_frequency=e.get("document_frequency", 1),
                confidence=e.get("confidence", 1.0),
                evidence=evidence,
            )
        )

    relations: list[Relation] = []
    for r in raw.get("relations", []):
        evidence = [
            EvidenceSpan(
                document_index=ev.get("document_index", 0),
                quote=ev.get("quote", ""),
                section=ev.get("section"),
            )
            for ev in r.get("evidence", [])
        ]
        relations.append(
            Relation(
                source=r["source"],
                target=r["target"],
                relation_type=r["relation_type"],
                support_count=r.get("support_count", 1),
                source_documents=r.get("source_documents", []),
                confidence=r.get("confidence", 1.0),
                evidence=evidence,
            )
        )

    reasoning_paths: list[ReasoningPath] = []
    for rp in raw.get("reasoning_paths", []):
        reasoning_paths.append(
            ReasoningPath(
                path=rp["path"],
                description=rp["description"],
                confidence=rp.get("confidence", 1.0),
            )
        )

    return KnowledgeGraph(
        entities=entities,
        relations=relations,
        reasoning_paths=reasoning_paths,
    )


def _parse_grader_report(raw: dict) -> GraderReport:
    """Convert raw LLM JSON dict into a validated GraderReport."""
    issues = [
        GraderIssue(
            issue_type=iss.get("issue_type", "UNKNOWN"),
            element_id=iss.get("element_id"),
            description=iss.get("description", ""),
            severity=iss.get("severity", "medium"),
        )
        for iss in raw.get("issues", [])
    ]
    return GraderReport(
        overall_score=raw.get("overall_score", 0.0),
        issues=issues,
        passes_threshold=raw.get("passes_threshold", False),
        feedback_summary=raw.get("feedback_summary", ""),
    )


def _get_ontology(req: ExtractionRequest) -> OntologyConfig:
    """Return the ontology from the request, or sensible defaults."""
    return req.ontology or OntologyConfig()


# ── 1. Ontology-Constrained Extraction ──────────────────────────────────────


def ontology_extraction(
    documents: list[str],
    req: ExtractionRequest,
) -> tuple[KnowledgeGraph, str]:
    """
    Method 1 — Constrained by a predefined ontology (rulebook).

    Returns the KG *and* the raw JSON string (needed by the grader).
    """
    ontology = _get_ontology(req)
    prompt = build_ontology_prompt(
        documents=documents,
        entity_types=ontology.entity_types,
        relation_types=ontology.relation_types,
        domain_context=ontology.domain_context,
    )
    logger.info("Calling LLM for ontology-constrained extraction …")
    data = call_llm_json(prompt)
    return _parse_kg(data), json.dumps(data, indent=2)


# ── 2. Evidence-Traced Extraction ───────────────────────────────────────────


def evidence_extraction(
    documents: list[str],
    req: ExtractionRequest,
) -> tuple[KnowledgeGraph, str]:
    """
    Method 2 — Every fact must carry a verbatim quote as proof.
    """
    ontology = _get_ontology(req)
    prompt = build_evidence_prompt(
        documents=documents,
        entity_types=ontology.entity_types,
        relation_types=ontology.relation_types,
        domain_context=ontology.domain_context,
    )
    logger.info("Calling LLM for evidence-traced extraction …")
    data = call_llm_json(prompt)
    return _parse_kg(data), json.dumps(data, indent=2)


# ── 3. Two-Agent System (Finder ↔ Grader Loop) ─────────────────────────────


def _grade_kg(
    kg_json: str,
    documents: list[str],
    ontology: OntologyConfig,
) -> tuple[GraderReport, str]:
    """Run the Grader agent on a KG and return its report + raw text."""
    prompt = build_grader_prompt(
        kg_json=kg_json,
        documents=documents,
        entity_types=ontology.entity_types,
        relation_types=ontology.relation_types,
    )
    logger.info("Calling LLM (Grader) …")
    data = call_llm_json(prompt)
    return _parse_grader_report(data), json.dumps(data, indent=2)


def _refine_kg(
    documents: list[str],
    previous_kg_json: str,
    grader_feedback: str,
    ontology: OntologyConfig,
) -> tuple[KnowledgeGraph, str]:
    """Run the Finder agent with grader feedback to produce an improved KG."""
    prompt = build_refinement_prompt(
        documents=documents,
        previous_kg_json=previous_kg_json,
        grader_feedback=grader_feedback,
        entity_types=ontology.entity_types,
        relation_types=ontology.relation_types,
        domain_context=ontology.domain_context,
    )
    logger.info("Calling LLM (Finder — refinement) …")
    data = call_llm_json(prompt)
    return _parse_kg(data), json.dumps(data, indent=2)


def two_agent_extraction(
    documents: list[str],
    req: ExtractionRequest,
) -> tuple[KnowledgeGraph, list[GraderReport], int, str]:
    """
    Method 3 — Finder + Grader self-correction loop.

    1. Finder produces initial KG (using evidence-traced extraction).
    2. Grader checks it.
    3. If score < threshold, Finder refines. Repeat up to max iterations.

    Returns (final_kg, grader_reports, iterations_used, last_kg_json).
    """
    ontology = _get_ontology(req)

    # Initial extraction (Finder — round 0)
    kg, kg_json = evidence_extraction(documents, req)
    grader_reports: list[GraderReport] = []

    for iteration in range(1, req.max_grader_iterations + 1):
        # Grade current KG
        report, _ = _grade_kg(kg_json, documents, ontology)
        grader_reports.append(report)
        logger.info(
            "Grader iteration %d — score %.2f, passes=%s",
            iteration,
            report.overall_score,
            report.passes_threshold,
        )

        if report.overall_score >= req.grader_threshold:
            return kg, grader_reports, iteration, kg_json

        # Refine
        kg, kg_json = _refine_kg(
            documents=documents,
            previous_kg_json=kg_json,
            grader_feedback=report.feedback_summary,
            ontology=ontology,
        )

    # Final grade after last refinement
    report, _ = _grade_kg(kg_json, documents, ontology)
    grader_reports.append(report)

    return kg, grader_reports, req.max_grader_iterations, kg_json


# ── 4. Multi-Document Evidence Accumulation ─────────────────────────────────


def _merge_kg(base: KnowledgeGraph, addition: KnowledgeGraph) -> KnowledgeGraph:
    """
    Merge ``addition`` into ``base`` using evidence accumulation.

    If an entity/relation already exists (by name/key), we add the new
    evidence spans rather than replacing.
    """
    # ---- Entities ----------------------------------------------------------
    entity_map: dict[str, Entity] = {e.name.lower(): e for e in base.entities}
    for ent in addition.entities:
        key = ent.name.lower()
        if key in entity_map:
            existing = entity_map[key]
            # Accumulate evidence
            existing_quotes = {ev.quote for ev in existing.evidence}
            for ev in ent.evidence:
                if ev.quote not in existing_quotes:
                    existing.evidence.append(ev)
            existing.document_frequency = max(
                existing.document_frequency, ent.document_frequency
            )
            existing.confidence = max(existing.confidence, ent.confidence)
        else:
            entity_map[key] = ent.model_copy()

    # ---- Relations ---------------------------------------------------------
    rel_map: dict[str, Relation] = {}
    for r in base.relations:
        key = f"{r.source}|{r.relation_type}|{r.target}"
        rel_map[key] = r

    for r in addition.relations:
        key = f"{r.source}|{r.relation_type}|{r.target}"
        if key in rel_map:
            existing = rel_map[key]
            existing_quotes = {ev.quote for ev in existing.evidence}
            for ev in r.evidence:
                if ev.quote not in existing_quotes:
                    existing.evidence.append(ev)
            for di in r.source_documents:
                if di not in existing.source_documents:
                    existing.source_documents.append(di)
            existing.support_count = max(existing.support_count, r.support_count)
            existing.confidence = max(existing.confidence, r.confidence)
        else:
            rel_map[key] = r.model_copy()

    return KnowledgeGraph(
        entities=list(entity_map.values()),
        relations=list(rel_map.values()),
        reasoning_paths=base.reasoning_paths + addition.reasoning_paths,
    )


def accumulate_extraction(
    documents: list[str],
    req: ExtractionRequest,
) -> tuple[KnowledgeGraph, str]:
    """
    Method 4 — Evidence Accumulation across multiple documents.

    Process each document individually, then merge KGs using evidence
    accumulation (never discard, only enrich).
    """
    ontology = _get_ontology(req)
    accumulated = KnowledgeGraph()

    for i, doc in enumerate(documents[:5]):
        doc_text = doc.strip()
        if not doc_text:
            continue

        logger.info("Accumulation — extracting from document %d …", i)
        prompt = build_evidence_prompt(
            documents=[doc_text],
            entity_types=ontology.entity_types,
            relation_types=ontology.relation_types,
            domain_context=ontology.domain_context,
        )
        data = call_llm_json(prompt)
        doc_kg = _parse_kg(data)

        accumulated = _merge_kg(accumulated, doc_kg)

    # Serialize the accumulated KG for downstream use
    kg_json = json.dumps(accumulated.model_dump(), indent=2)
    return accumulated, kg_json


# ── 5. Full Pipeline (all methods combined) ─────────────────────────────────


def full_pipeline(
    documents: list[str],
    req: ExtractionRequest,
) -> ExtractionResponse:
    """
    Method 5 — Full pipeline combining all methods:

    1. Evidence-accumulated extraction across documents.
    2. Finder ↔ Grader self-correction loop.
    3. Reasoning-path discovery.
    """
    ontology = _get_ontology(req)

    # Step 1 — accumulate across documents
    logger.info("Full pipeline — accumulation phase …")
    kg, kg_json = accumulate_extraction(documents, req)

    # Step 2 — grader loop on the accumulated KG
    grader_reports: list[GraderReport] = []
    iterations = 0

    for iteration in range(1, req.max_grader_iterations + 1):
        report, _ = _grade_kg(kg_json, documents, ontology)
        grader_reports.append(report)
        iterations = iteration
        logger.info(
            "Full pipeline grader iteration %d — score %.2f",
            iteration,
            report.overall_score,
        )

        if report.overall_score >= req.grader_threshold:
            break

        kg, kg_json = _refine_kg(
            documents=documents,
            previous_kg_json=kg_json,
            grader_feedback=report.feedback_summary,
            ontology=ontology,
        )

    # Step 3 — discover reasoning paths
    logger.info("Full pipeline — reasoning paths …")
    rp_prompt = build_reasoning_paths_prompt(kg_json)
    rp_data = call_llm_json(rp_prompt)
    for rp in rp_data.get("reasoning_paths", []):
        kg.reasoning_paths.append(
            ReasoningPath(
                path=rp["path"],
                description=rp["description"],
                confidence=rp.get("confidence", 1.0),
            )
        )

    final_score = grader_reports[-1].overall_score if grader_reports else None

    return ExtractionResponse(
        method="full_pipeline",
        cluster_index=req.cluster_num,
        num_documents=len(documents),
        knowledge_graph=kg,
        grader_reports=grader_reports,
        iterations_used=iterations,
        final_score=final_score,
        metadata={
            "ontology_entity_types": ontology.entity_types,
            "ontology_relation_types": ontology.relation_types,
            "domain_context": ontology.domain_context,
        },
    )
