#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_extractors/__init__.py

"""Plot type extractors for FTS bundle creation.

Each extractor handles data extraction and encoding building for a specific plot type.
"""

# Line plots
from ._extract_line import build_line_traces, extract_line_data

# Scatter plots (PathCollection)
from ._extract_scatter import build_scatter_traces, extract_scatter_data

# Bar charts (Rectangle patches)
from ._extract_bar import build_bar_traces, count_valid_bars, extract_bar_data

__all__ = [
    # Line
    "extract_line_data",
    "build_line_traces",
    # Scatter
    "extract_scatter_data",
    "build_scatter_traces",
    # Bar
    "extract_bar_data",
    "count_valid_bars",
    "build_bar_traces",
]

# EOF
