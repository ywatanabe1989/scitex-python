#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/__init__.py

"""Core WebEditor package for Flask-based figure editing.

This package provides the WebEditor class and supporting modules for
browser-based figure editing functionality.
"""

from ._bbox_extraction import extract_bboxes_from_metadata
from ._editor import WebEditor
from ._export_helpers import compose_panels_to_figure, export_composed_figure

# Backward compatibility alias
_extract_bboxes_from_metadata = extract_bboxes_from_metadata

__all__ = [
    "WebEditor",
    "extract_bboxes_from_metadata",
    "export_composed_figure",
    "compose_panels_to_figure",
    "_extract_bboxes_from_metadata",
]


# EOF
