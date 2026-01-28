#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/local_dbs/crossref_scitex.py
"""CrossRef-SciTeX: Minimal API for crossref-local.

Usage:
    >>> from scitex.scholar.local_dbs import crossref_scitex
    >>> results = crossref_scitex.search("machine learning")
    >>> work = crossref_scitex.get("10.1038/nature12373")
"""

try:
    from crossref_local import (
        SearchResult,
        # Classes
        Work,
        count,
        get,
        info,
        # Core functions
        search,
    )
except ImportError as e:
    raise ImportError(
        "crossref-local not installed. Install with: pip install crossref-local"
    ) from e

__all__ = ["search", "get", "count", "info", "Work", "SearchResult"]

# EOF
