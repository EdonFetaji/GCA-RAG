"""
Prompt builders for the five KG-extraction methods.

Each builder returns a plain string prompt ready to send to `call_llm()`.
"""

from __future__ import annotations

from typing import Optional


# ── Helpers ──────────────────────────────────────────────────────────────────


def _format_documents(documents: list[str], limit: int = 5) -> str:
    """Join documents with index headers so the LLM can reference them."""
    parts: list[str] = []
    for i, doc in enumerate(documents[:limit]):
        parts.append(f"[DOCUMENT {i}]\n{doc.strip()}")
    return "\n\n---\n\n".join(parts)


# ── 1. Ontology-Constrained Extraction ──────────────────────────────────────


def build_ontology_prompt(
    documents: list[str],
    entity_types: list[str],
    relation_types: list[str],
    domain_context: str = "general news articles",
) -> str:
    """
    Step 1 — Give the AI a "Rulebook" (Ontology).

    The LLM is constrained to ONLY use the provided entity/relation types.
    This prevents creative drift and keeps the KG schema consistent.
    """
    docs_text = _format_documents(documents)
    entity_list = ", ".join(entity_types)
    relation_list = ", ".join(relation_types)

    return f"""You are a precise knowledge-graph extractor for {domain_context}.

## ONTOLOGY RULEBOOK — you MUST follow these constraints:
ALLOWED ENTITY TYPES (use ONLY these): {entity_list}
ALLOWED RELATION TYPES (use ONLY these): {relation_list}

If a fact does not fit any allowed type, SKIP it. Never invent new types.

## TASK
Read the documents below and extract a knowledge graph.

Output ONLY valid JSON (no markdown, no code fences, no commentary):
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "<one of the allowed entity types>",
      "document_frequency": 3,
      "confidence": 0.95
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "<one of the allowed relation types>",
      "support_count": 2,
      "source_documents": [0, 1],
      "confidence": 0.85
    }}
  ]
}}

CRITICAL RULES:
- Every relation source/target MUST reference an entity id from the entities list.
- document_frequency = how many documents mention this entity.
- support_count = how many documents support this relation.
- confidence = how certain you are (0.0–1.0).
- Do NOT include any entity or relation whose type is not in the rulebook.

## DOCUMENTS
{docs_text}

JSON:"""


# ── 2. Evidence-Traced Extraction (Show Your Work) ──────────────────────────


def build_evidence_prompt(
    documents: list[str],
    entity_types: list[str],
    relation_types: list[str],
    domain_context: str = "general news articles",
) -> str:
    """
    Step 2 — Make the AI "Show Its Work" (Evidence Traceability).

    For every entity and relation the LLM must provide the exact verbatim
    sentence(s) from the source documents as proof.  This makes the KG
    auditable and dramatically reduces hallucinations.
    """
    docs_text = _format_documents(documents)
    entity_list = ", ".join(entity_types)
    relation_list = ", ".join(relation_types)

    return f"""You are an evidence-grounded knowledge-graph extractor for {domain_context}.

## ONTOLOGY RULEBOOK
ALLOWED ENTITY TYPES: {entity_list}
ALLOWED RELATION TYPES: {relation_list}

## CRITICAL REQUIREMENT — EVIDENCE TRACEABILITY
For EVERY entity and EVERY relation you extract, you MUST provide at least one
verbatim quote copied exactly from the source document that proves it.
If you cannot find a direct quote, do NOT include the fact.

## OUTPUT FORMAT — valid JSON only (no markdown, no code fences):
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "<allowed type>",
      "document_frequency": 2,
      "confidence": 0.95,
      "evidence": [
        {{
          "document_index": 0,
          "quote": "Exact sentence copied from Document 0 proving this entity."
        }}
      ]
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "<allowed type>",
      "support_count": 2,
      "source_documents": [0, 1],
      "confidence": 0.85,
      "evidence": [
        {{
          "document_index": 0,
          "quote": "Exact sentence proving this relation."
        }},
        {{
          "document_index": 1,
          "quote": "Another sentence from Doc 1 corroborating."
        }}
      ]
    }}
  ]
}}

RULES:
- Every entity and relation MUST have at least one evidence quote.
- Quotes must be EXACT substrings from the documents — no paraphrasing.
- document_index is the 0-based index shown in [DOCUMENT N] headers.
- Only use allowed entity/relation types.

## DOCUMENTS
{docs_text}

JSON:"""


# ── 3a. Grader Prompt (Two-Agent System) ────────────────────────────────────


def build_grader_prompt(
    kg_json: str,
    documents: list[str],
    entity_types: list[str],
    relation_types: list[str],
) -> str:
    """
    Step 3 — AI #2 (The Grader).

    Receives the Finder's KG and the original documents.
    Must check for hallucinations, missing facts, contradictions, and
    produce an actionable feedback report.
    """
    docs_text = _format_documents(documents)

    return f"""You are a strict quality-assurance grader for knowledge graphs.

## YOUR TASK
A Finder AI has extracted the knowledge graph below from the source documents.
Your job is to grade it and list every issue you find.

## ALLOWED TYPES
Entity types: {", ".join(entity_types)}
Relation types: {", ".join(relation_types)}

## CHECKS TO PERFORM
1. **HALLUCINATION** — Is any entity or relation NOT supported by the documents?
   Verify each evidence quote actually appears in the cited document.
2. **MISSING_ENTITY** — Are there important entities in the documents that were missed?
3. **MISSING_RELATION** — Are there important relations that were missed?
4. **CONTRADICTION** — Do any two relations contradict each other?
5. **WEAK_EVIDENCE** — Is any evidence quote too vague or a paraphrase rather than verbatim?
6. **WRONG_TYPE** — Does any entity/relation use a type outside the allowed list?

## OUTPUT FORMAT — valid JSON only:
{{
  "overall_score": 0.75,
  "passes_threshold": false,
  "issues": [
    {{
      "issue_type": "HALLUCINATION|MISSING_ENTITY|MISSING_RELATION|CONTRADICTION|WEAK_EVIDENCE|WRONG_TYPE",
      "element_id": "entity_3 or null",
      "description": "Explain the problem and suggest a fix.",
      "severity": "low|medium|high"
    }}
  ],
  "feedback_summary": "Concise paragraph telling the Finder exactly what to fix."
}}

## EXTRACTED KNOWLEDGE GRAPH (from the Finder)
{kg_json}

## SOURCE DOCUMENTS
{docs_text}

JSON:"""


# ── 3b. Refinement Prompt (Finder reacts to Grader feedback) ────────────────


def build_refinement_prompt(
    documents: list[str],
    previous_kg_json: str,
    grader_feedback: str,
    entity_types: list[str],
    relation_types: list[str],
    domain_context: str = "general news articles",
) -> str:
    """
    Step 3 continued — The Finder receives the Grader's feedback and
    produces an improved KG.
    """
    docs_text = _format_documents(documents)
    entity_list = ", ".join(entity_types)
    relation_list = ", ".join(relation_types)

    return f"""You are an evidence-grounded knowledge-graph extractor for {domain_context}.

## PREVIOUS ATTEMPT
You previously extracted this knowledge graph:
{previous_kg_json}

## GRADER FEEDBACK
A quality-assurance reviewer found these problems:
{grader_feedback}

## YOUR TASK
Fix every issue the grader raised. Produce a CORRECTED knowledge graph.
- Remove any hallucinated facts.
- Add any missing entities or relations the grader identified.
- Replace weak/paraphrased evidence with exact verbatim quotes.
- Fix any wrong entity or relation types.
- Keep everything that the grader did NOT flag.

## ONTOLOGY RULEBOOK
ALLOWED ENTITY TYPES: {entity_list}
ALLOWED RELATION TYPES: {relation_list}

## OUTPUT FORMAT — same JSON schema as before (with evidence quotes):
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "<allowed type>",
      "document_frequency": 2,
      "confidence": 0.95,
      "evidence": [
        {{"document_index": 0, "quote": "Exact quote."}}
      ]
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "<allowed type>",
      "support_count": 2,
      "source_documents": [0, 1],
      "confidence": 0.85,
      "evidence": [
        {{"document_index": 0, "quote": "Exact quote."}}
      ]
    }}
  ]
}}

## SOURCE DOCUMENTS
{docs_text}

JSON:"""


# ── 4. Section-Aware Extraction ─────────────────────────────────────────────


def build_section_prompt(
    section_text: str,
    section_name: str,
    entity_types: list[str],
    relation_types: list[str],
    document_index: int = 0,
) -> str:
    """
    Step 3 from the checklist — Break it into sections.

    Instead of feeding the AI the whole document, extract from one
    section at a time with tailored instructions.
    """
    entity_list = ", ".join(entity_types)
    relation_list = ", ".join(relation_types)

    section_guidance = {
        "abstract": "Focus on the main topic, key claims, and high-level entities.",
        "introduction": "Focus on background context, problem statement, and motivations.",
        "methods": "Focus on techniques, tools, datasets, and procedures used.",
        "results": "Focus on findings, measurements, comparisons, and outcomes.",
        "discussion": "Focus on interpretations, implications, limitations, and future work.",
        "conclusion": "Focus on key takeaways and final claims.",
    }
    guidance = section_guidance.get(
        section_name.lower(),
        "Extract all relevant entities and relations from this section.",
    )

    return f"""You are extracting facts from the **{section_name}** section of a document.

GUIDANCE FOR THIS SECTION: {guidance}

ALLOWED ENTITY TYPES: {entity_list}
ALLOWED RELATION TYPES: {relation_list}

Output ONLY valid JSON:
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "<allowed type>",
      "confidence": 0.95,
      "evidence": [
        {{"document_index": {document_index}, "quote": "Exact quote from this section.", "section": "{section_name}"}}
      ]
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "<allowed type>",
      "confidence": 0.85,
      "evidence": [
        {{"document_index": {document_index}, "quote": "Exact quote.", "section": "{section_name}"}}
      ]
    }}
  ]
}}

## SECTION TEXT ({section_name})
{section_text}

JSON:"""


# ── 5. Reasoning Paths (Connect the Dots) ───────────────────────────────────


def build_reasoning_paths_prompt(kg_json: str) -> str:
    """
    Step 4 from the checklist — Connect the dots.

    Given an already-extracted KG, ask the LLM to find multi-hop
    reasoning chains (A → B → C) that reveal deeper insights.
    """
    return f"""You are an analytical reasoning engine.

Given the knowledge graph below, find reasoning paths — chains of
relations that reveal deeper insights (e.g., "A causes B, and B causes C,
therefore A may lead to C").

## KNOWLEDGE GRAPH
{kg_json}

## OUTPUT FORMAT — valid JSON only:
{{
  "reasoning_paths": [
    {{
      "path": ["entity_1", "entity_3", "entity_7"],
      "description": "Natural language explanation of this chain of reasoning.",
      "confidence": 0.8
    }}
  ]
}}

RULES:
- Each path must contain at least 3 entity ids that form a connected chain.
- The description must explain the causal or logical link.
- confidence reflects how strongly the chain is supported by the KG evidence.
- Find at least 3 paths if possible.

JSON:"""


# ── Legacy wrapper (backwards compatibility) ────────────────────────────────


def build_extraction_prompt(documents: list[str]) -> str:
    """Original simple extraction prompt (kept for backward compat)."""
    docs_text = "\n\n---DOCUMENT---\n\n".join(documents[:5])

    return f"""You are extracting a knowledge graph from a cluster of news articles about the same topic.

Extract:
1. **Entities**: Key people, organizations, locations, events, concepts mentioned across documents
2. **Relations**: How these entities relate to each other

Output ONLY valid JSON in this exact format (no markdown, no code fences, no commentary):
{{
  "entities": [
    {{
      "id": "entity_1",
      "name": "Entity Name",
      "type": "PERSON|ORGANIZATION|LOCATION|EVENT|CONCEPT",
      "document_frequency": 3,
      "confidence": 0.95
    }}
  ],
  "relations": [
    {{
      "source": "entity_1",
      "target": "entity_2",
      "relation_type": "verb phrase describing relation",
      "support_count": 2,
      "source_documents": [0, 1],
      "confidence": 0.85
    }}
  ]
}}

CRITICAL RULES:
- Every relation's source and target MUST reference an entity id from the entities list
- Use past tense verbs for relations (e.g., "announced", "acquired", "launched")
- document_frequency = how many documents mention this entity
- support_count = how many documents support this relation
- confidence = how certain you are this entity/relation is correct (0.0 to 1.0)

Documents:
{docs_text}

Extract the knowledge graph as JSON:"""