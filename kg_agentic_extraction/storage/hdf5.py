"""
HDF5 persistence for extracted knowledge graphs.

Layout
------
The graph is written as flat, columnar datasets rather than a serialized JSON
blob, so a reader can pull just the entity table (or just the quotes) without
parsing the whole file — which is the only reason to use HDF5 over JSON here.

    /                          attrs: cluster_index, n_entities, n_relations,
                                      converged, iterations, model, provider,
                                      created_utc, schema_version
    /entities                  compound: id, name, type, document_frequency, confidence
    /relations                 compound: source, target, relation_type,
                                         support_count, confidence
    /relation_source_documents vlen int32, row-aligned with /relations
    /entity_evidence           compound: element_index, element_id,
                                         document_index, quote
    /relation_evidence         compound: element_index, element_key,
                                         document_index, quote

Evidence is a separate table rather than a nested field because HDF5 compound
types are fixed-width: an entity may carry zero or five quotes, so inlining them
would mean padding every row to the worst case. `element_index` points back to
the row in /entities or /relations; `element_id` / `element_key` repeat the
identity so the evidence table is independently readable.

Strings are variable-length UTF-8, and the quote-bearing tables are gzipped —
evidence spans are verbatim sentences duplicated across near-identical source
documents, so they compress heavily.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from kg_agentic_extraction.models.knowledge_graph import (
    Entity,
    EvidenceSpan,
    KnowledgeGraph,
    Relation,
)

logger = logging.getLogger(__name__)

#: Bump when the on-disk layout changes incompatibly.
SCHEMA_VERSION = 1

_STR = h5py.string_dtype(encoding="utf-8")

_ENTITY_DTYPE = np.dtype(
    [
        ("id", _STR),
        ("name", _STR),
        ("type", _STR),
        ("document_frequency", np.int32),
        # float64, not float32: float32 cannot represent common confidence
        # values (0.95 -> 0.949999988) exactly, which breaks round-trip fidelity.
        ("confidence", np.float64),
    ]
)

_RELATION_DTYPE = np.dtype(
    [
        ("source", _STR),
        ("target", _STR),
        ("relation_type", _STR),
        ("support_count", np.int32),
        ("confidence", np.float64),
    ]
)

_EVIDENCE_DTYPE = np.dtype(
    [
        ("element_index", np.int32),
        ("element_id", _STR),
        ("document_index", np.int32),
        ("quote", _STR),
    ]
)

_VLEN_INT = h5py.vlen_dtype(np.int32)


def graph_filename(cluster_index: int | None, *, stem: str = "cluster") -> str:
    """
    Filename for one cluster's graph.

    `cluster_index` is None when the pipeline was run on an ad-hoc document set
    rather than a dataset cluster (`--file`), in which case there is no index to
    embed and the file is named `<stem>.h5`.
    """
    return f"{stem}_{cluster_index}.h5" if cluster_index is not None else f"{stem}.h5"


def save_knowledge_graph(
    graph: KnowledgeGraph,
    path: str | Path,
    *,
    cluster_index: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """
    Write `graph` to `path` in HDF5.

    Parameters
    ----------
    graph
        The extracted knowledge graph.
    path
        Destination `.h5` file. Parent directories are created.
    cluster_index
        Recorded as a root attribute. `-1` is stored when None, since HDF5
        attributes have no null.
    metadata
        Extra root attributes (model, provider, convergence, …). Values are
        coerced to str unless already int/float/bool.

    Returns
    -------
    Path
        The file that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary file and swap, so an interrupted write cannot leave a
    # half-written .h5 that later looks readable.
    tmp = path.with_suffix(path.suffix + ".tmp")

    with h5py.File(tmp, "w") as f:
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["cluster_index"] = -1 if cluster_index is None else int(cluster_index)
        f.attrs["n_entities"] = len(graph.entities)
        f.attrs["n_relations"] = len(graph.relations)
        f.attrs["created_utc"] = datetime.now(UTC).isoformat()
        for key, value in (metadata or {}).items():
            f.attrs[key] = value if isinstance(value, (int, float, bool)) else str(value)

        _write_entities(f, graph.entities)
        _write_relations(f, graph.relations)

    tmp.replace(path)
    logger.info(
        "saved graph → %s (%d entities, %d relations)",
        path,
        len(graph.entities),
        len(graph.relations),
    )
    return path


def _write_entities(f: h5py.File, entities: list[Entity]) -> None:
    rows = np.array(
        [(e.id, e.name, e.type.value, e.document_frequency, e.confidence) for e in entities],
        dtype=_ENTITY_DTYPE,
    )
    f.create_dataset("entities", data=rows, compression="gzip")
    _write_evidence(
        f,
        "entity_evidence",
        [(i, e.id, e.evidence) for i, e in enumerate(entities)],
    )


def _write_relations(f: h5py.File, relations: list[Relation]) -> None:
    rows = np.array(
        [
            (r.source, r.target, r.relation_type.value, r.support_count, r.confidence)
            for r in relations
        ],
        dtype=_RELATION_DTYPE,
    )
    f.create_dataset("relations", data=rows, compression="gzip")

    # Ragged per-relation document lists, row-aligned with /relations.
    src_docs = f.create_dataset("relation_source_documents", (len(relations),), dtype=_VLEN_INT)
    for i, r in enumerate(relations):
        src_docs[i] = np.array(r.source_documents, dtype=np.int32)

    _write_evidence(
        f,
        "relation_evidence",
        [(i, r.key, r.evidence) for i, r in enumerate(relations)],
    )


def _write_evidence(
    f: h5py.File,
    name: str,
    items: list[tuple[int, str, list[EvidenceSpan]]],
) -> None:
    rows = np.array(
        [
            (index, element_id, span.document_index, span.quote)
            for index, element_id, spans in items
            for span in spans
        ],
        dtype=_EVIDENCE_DTYPE,
    )
    f.create_dataset(name, data=rows, compression="gzip")


def load_knowledge_graph(path: str | Path) -> tuple[KnowledgeGraph, dict[str, Any]]:
    """
    Read back a graph written by `save_knowledge_graph`.

    Returns the graph and the root attributes. Provided so a saved file can be
    verified as a genuine round-trip rather than assumed correct because the
    write did not raise.
    """
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = {k: _decode(v) for k, v in f.attrs.items()}

        evidence_by_index = _read_evidence(f, "entity_evidence")
        entities = [
            Entity(
                id=_decode(row["id"]),
                name=_decode(row["name"]),
                type=_decode(row["type"]),
                document_frequency=int(row["document_frequency"]),
                confidence=float(row["confidence"]),
                evidence=evidence_by_index.get(i, []),
            )
            for i, row in enumerate(f["entities"][()])
        ]

        rel_evidence = _read_evidence(f, "relation_evidence")
        src_docs = f["relation_source_documents"][()]
        relations = [
            Relation(
                source=_decode(row["source"]),
                target=_decode(row["target"]),
                relation_type=_decode(row["relation_type"]),
                support_count=int(row["support_count"]),
                source_documents=[int(d) for d in src_docs[i]],
                confidence=float(row["confidence"]),
                evidence=rel_evidence.get(i, []),
            )
            for i, row in enumerate(f["relations"][()])
        ]

    return KnowledgeGraph(entities=entities, relations=relations), attrs


def _read_evidence(f: h5py.File, name: str) -> dict[int, list[EvidenceSpan]]:
    out: dict[int, list[EvidenceSpan]] = {}
    for row in f[name][()]:
        out.setdefault(int(row["element_index"]), []).append(
            EvidenceSpan(
                document_index=int(row["document_index"]),
                quote=_decode(row["quote"]),
            )
        )
    return out


def _decode(value: Any) -> Any:
    """HDF5 hands back variable-length strings as bytes."""
    return value.decode("utf-8") if isinstance(value, bytes) else value
