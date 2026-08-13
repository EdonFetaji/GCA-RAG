"""
extractor_agent — modular knowledge-graph extraction pipeline.

Steps: extract_entities -> validate_schema -> normalize_entities ->
extract_relations -> build_graph_object, orchestrated by
`extractor_agent.run_extractor_pipeline`.

Making this an explicit package (rather than relying on Python's implicit
namespace-package behavior) is what makes
`from extractor_agent.extractor_agent import run_extractor_pipeline` a
reliable import from the project root.
"""
