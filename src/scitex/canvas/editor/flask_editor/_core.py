#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core.py

"""Core WebEditor class for Flask-based figure editing.

This module re-exports the WebEditor class and helper functions from the
_core package for backward compatibility. The actual implementation is
in the _core/ subpackage.
"""

from ._core import (
    WebEditor,
    _extract_bboxes_from_metadata,
    compose_panels_to_figure,
    export_composed_figure,
    extract_bboxes_from_metadata,
)

__all__ = [
    "WebEditor",
    "extract_bboxes_from_metadata",
    "_extract_bboxes_from_metadata",
    "export_composed_figure",
    "compose_panels_to_figure",
]


# EOF
