#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/_plot/__init__.py

"""Plot kind - Data visualization with encoding.

A plot bundle contains data (payload/data.csv) and an encoding
specification that maps data columns to visual channels (x, y, color, etc.).

Structure:
- payload/data.csv: Source data
- canonical/encoding.json: Data-to-visual mappings
- canonical/theme.json: Visual styling
"""

from ._dataclasses import Encoding, Theme
from ._backend._render import render_traces

__all__ = [
    "Encoding",
    "Theme",
    "render_traces",
]

# EOF
