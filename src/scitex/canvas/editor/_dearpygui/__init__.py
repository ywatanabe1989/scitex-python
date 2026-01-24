#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/__init__.py

"""
DearPyGui editor package.

Modular GPU-accelerated figure editor using DearPyGui.

Modules:
- _state: EditorState dataclass for all editor state
- _utils: Utility functions (checkerboard, constants)
- _rendering: Figure rendering and highlight drawing
- _plotting: CSV data plotting
- _selection: Element selection and hover detection
- _handlers: Event handlers and callbacks
- _panels: UI panel creation
"""

from ._editor import DearPyGuiEditor

__all__ = ["DearPyGuiEditor"]


# EOF
