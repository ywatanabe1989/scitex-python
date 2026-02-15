#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/local_dbs/openalex_scitex.py
"""OpenAlex-SciTeX: Minimal API for openalex-local.

Usage:
    >>> from scitex.scholar.local_dbs import openalex_scitex
    >>> results = openalex_scitex.search("machine learning")
    >>> work = openalex_scitex.get("10.1038/nature12373")
"""

try:
    from openalex_local import (  # Classes; Core functions
        SearchResult,
        Work,
        count,
        get,
        info,
        search,
    )
except ImportError as e:
    raise ImportError(
        "openalex-local not installed. Install with: pip install openalex-local"
    ) from e

__all__ = ["search", "get", "count", "info", "Work", "SearchResult"]

# EOF
