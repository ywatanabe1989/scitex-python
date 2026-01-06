#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/__init__.py

"""FTS Kind-specific modules.

Bundle kinds:
- figure: Composite container (has children, no payload)
- plot: Data visualization (has payload/data.csv)
- table: Tabular data (has payload/table.csv)
- stats: Statistical results (has payload/stats.json)
- text: Text annotation (no payload)
- shape: Shape annotation (no payload)
- image: Embedded image (has payload/image.*)
"""

from ._figure import render_composite
from ._plot import Encoding, Theme, render_traces
from ._stats import Stats
from ._table import export_to_latex
from ._text import render_text
from ._shape import render_shape
from ._image import render_image, load_image

__all__ = [
    # Figure (composite)
    "render_composite",
    # Plot
    "Encoding",
    "Theme",
    "render_traces",
    # Stats
    "Stats",
    # Table
    "export_to_latex",
    # Text
    "render_text",
    # Shape
    "render_shape",
    # Image
    "render_image",
    "load_image",
]

# EOF
