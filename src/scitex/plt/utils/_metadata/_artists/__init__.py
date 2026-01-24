#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_artists/__init__.py

"""
Artist extraction for figure metadata.

This package splits the large _extract_artists function into logical modules:
- _base: Common utilities and context setup
- _lines: Line2D artist extraction (including boxplot/violin/stem semantics)
- _collections: Collection artist extraction (scatter, hexbin, violin bodies)
- _patches: Patch artist extraction (bar, pie, histogram)
- _images: Image artist extraction
- _text: Text artist extraction
"""

from ._extract import _extract_artists

# Backward compatibility alias
_extract_traces = _extract_artists

__all__ = ["_extract_artists", "_extract_traces"]


# EOF
