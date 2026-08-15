"""
Package entrypoint, so `python -m kg_agentic_extraction` works.

    uv run python -m kg_agentic_extraction --cluster 0 --report report.md
"""

import sys

from kg_agentic_extraction.runner import main

if __name__ == "__main__":
    sys.exit(main())
