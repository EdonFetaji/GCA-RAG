"""
FastAPI router — exposes each KG-extraction method as a testable endpoint.

Endpoints
---------
GET  /extract/health              → liveness check
POST /extract/ontology            → Method 1: ontology-constrained extraction
POST /extract/evidence            → Method 2: evidence-traced extraction
POST /extract/two-agent           → Method 3: finder ↔ grader self-correction
POST /extract/accumulate          → Method 4: multi-document evidence accumulation
POST /extract/full-pipeline       → Method 5: full pipeline (all combined)
GET  /extract/kg                  → legacy simple extraction (backward compat)
"""

from __future__ import annotations

import json
import logging
import traceback

from fastapi import FastAPI, HTTPException

from extraction.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    KnowledgeGraph,
    OntologyConfig,
)
from extraction.service import (
    accumulate_extraction,
    evidence_extraction,
    full_pipeline,
    ontology_extraction,
    two_agent_extraction,
)
from utils.dataset_utils import load_single_cluster

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="GCA-RAG KG Extraction API",
    description="Test endpoints for prompt-based knowledge-graph extraction methods.",
    version="0.2.0",
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _load_docs(req: ExtractionRequest) -> list[str]:
    """Load the document cluster specified in the request."""
    documents, _ = load_single_cluster(cluster_idx=req.cluster_num)
    return [d.strip() for d in documents if d.strip()]


# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/extract/health")
async def health():
    return {"status": "ok"}


@app.post("/extract/ontology", response_model=ExtractionResponse)
async def extract_ontology(req: ExtractionRequest):
    """
    **Method 1 — Ontology-Constrained Extraction**

    The AI is given a strict "rulebook" of allowed entity & relation types.
    It may only extract facts that fit the schema — nothing else.
    """
    try:
        documents = _load_docs(req)
        kg, _ = ontology_extraction(documents, req)
        return ExtractionResponse(
            method="ontology_constrained",
            cluster_index=req.cluster_num,
            num_documents=len(documents),
            knowledge_graph=kg,
        )
    except Exception as exc:
        logger.error("ontology extraction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract/evidence", response_model=ExtractionResponse)
async def extract_evidence(req: ExtractionRequest):
    """
    **Method 2 — Evidence-Traced Extraction**

    Every entity and relation MUST carry a verbatim quote from the source
    documents. This makes the KG fully auditable and reduces hallucinations.
    """
    try:
        documents = _load_docs(req)
        kg, _ = evidence_extraction(documents, req)
        return ExtractionResponse(
            method="evidence_traced",
            cluster_index=req.cluster_num,
            num_documents=len(documents),
            knowledge_graph=kg,
        )
    except Exception as exc:
        logger.error("evidence extraction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract/two-agent", response_model=ExtractionResponse)
async def extract_two_agent(req: ExtractionRequest):
    """
    **Method 3 — Two-Agent (Finder ↔ Grader) System**

    AI #1 (Finder) extracts the KG.  AI #2 (Grader) audits it for
    hallucinations, missing facts, and contradictions.  If the score is
    below the threshold the Finder is asked to fix the issues — this loops
    until the KG passes or max iterations are reached.
    """
    try:
        documents = _load_docs(req)
        kg, reports, iters, _ = two_agent_extraction(documents, req)
        final_score = reports[-1].overall_score if reports else None
        return ExtractionResponse(
            method="two_agent",
            cluster_index=req.cluster_num,
            num_documents=len(documents),
            knowledge_graph=kg,
            grader_reports=reports,
            iterations_used=iters,
            final_score=final_score,
        )
    except Exception as exc:
        logger.error("two-agent extraction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract/accumulate", response_model=ExtractionResponse)
async def extract_accumulate(req: ExtractionRequest):
    """
    **Method 4 — Multi-Document Evidence Accumulation**

    Each document is processed individually.  When a later document mentions
    a fact already found earlier, the new evidence is *added* to the existing
    node/edge rather than replacing it — building a richer, multi-sourced KG.
    """
    try:
        documents = _load_docs(req)
        kg, _ = accumulate_extraction(documents, req)
        return ExtractionResponse(
            method="evidence_accumulation",
            cluster_index=req.cluster_num,
            num_documents=len(documents),
            knowledge_graph=kg,
        )
    except Exception as exc:
        logger.error("accumulation extraction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/extract/full-pipeline", response_model=ExtractionResponse)
async def extract_full_pipeline(req: ExtractionRequest):
    """
    **Method 5 — Full Pipeline (all methods combined)**

    1. Evidence-accumulated extraction across all documents.
    2. Finder ↔ Grader self-correction loop.
    3. Reasoning-path discovery (multi-hop chains).
    """
    try:
        documents = _load_docs(req)
        return full_pipeline(documents, req)
    except Exception as exc:
        logger.error("full pipeline failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))


# ── Legacy endpoint (backward compat) ───────────────────────────────────────


@app.get("/extract/kg")
async def extract_kg_legacy(dataset: str = "multi_news", cluster_num: int = 0):
    """Legacy endpoint kept for backward compatibility."""
    req = ExtractionRequest(dataset=dataset, cluster_num=cluster_num)
    try:
        documents = _load_docs(req)
        kg, _ = ontology_extraction(documents, req)
        return {
            "cluster_index": cluster_num,
            "num_documents": len(documents),
            "knowledge_graph": kg.model_dump(),
        }
    except Exception as exc:
        logger.error("legacy extraction failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(exc))

