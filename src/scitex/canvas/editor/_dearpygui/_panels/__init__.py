#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_panels/__init__.py

"""
UI panel creation for DearPyGui editor.

Subpackage for organizing control panel sections.
"""

from ._control import create_control_panel
from ._preview import create_preview_panel

__all__ = ["create_preview_panel", "create_control_panel"]


# EOF
