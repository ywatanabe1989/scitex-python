#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/scholar/local_dbs/__init__.py
"""Local database integrations for scitex.scholar.

This package provides access to local scholarly databases:
- crossref_scitex: CrossRef database (167M+ papers via crossref-local)
- openalex_scitex: OpenAlex database (284M+ works via openalex-local)

Both modules delegate directly to their respective external packages
without any additional logic.

Usage:
    >>> from scitex.scholar.local_dbs import crossref_scitex
    >>> results = crossref_scitex.search("machine learning")

    >>> from scitex.scholar.local_dbs import openalex_scitex
    >>> results = openalex_scitex.search("neural networks")
"""

from __future__ import annotations

from . import crossref_scitex, openalex_scitex

__all__ = [
    "crossref_scitex",
    "openalex_scitex",
]


# EOF
