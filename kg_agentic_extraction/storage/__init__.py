"""
storage — persistence for extracted graphs.

Depends only on `models/`, never on agents or the graph, so a saved file can be
read back by anything (Track 3's GNN training, analysis notebooks) without
importing the pipeline.
"""

from kg_agentic_extraction.storage.hdf5 import (
    SCHEMA_VERSION,
    graph_filename,
    load_knowledge_graph,
    save_knowledge_graph,
)

__all__ = [
    "SCHEMA_VERSION",
    "graph_filename",
    "load_knowledge_graph",
    "save_knowledge_graph",
]
