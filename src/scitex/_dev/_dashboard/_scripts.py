#!/usr/bin/env python3
# Timestamp: 2026-02-05
# File: scitex/_dev/_dashboard/_scripts.py

"""JavaScript for the dashboard.

This module re-exports get_javascript() from the modular _scripts/ package.
The JavaScript has been split into:
- _scripts/_core.py: Fetch, cache, refresh functions
- _scripts/_filters.py: Filter rendering
- _scripts/_cards.py: Package card rendering with source badges
- _scripts/_render.py: Main data rendering
- _scripts/_utils.py: Export, copy, toggle utilities
"""

from ._scripts import get_javascript

__all__ = ["get_javascript"]


# EOF
