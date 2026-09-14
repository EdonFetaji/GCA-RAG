"""
poc — frozen reference implementations from the pre-LangGraph generation.

Nothing here is on the active path. These modules are kept because they
document how the extraction problem was first approached (and because
`generate_training_data.py` still generates Track 2 data through
`poc.extraction.service`), but new work belongs in `kg_agentic_extraction/`.

Nothing outside this package should import from it, with the single
documented exception of `generate_training_data.py`. See `poc/README.md`.
"""
