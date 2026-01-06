#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/_flask_editor.py
"""Web-based figure editor using Flask.

This module re-exports from the flask_editor package for backward compatibility.
The actual implementation is in the flask_editor/ subpackage.
"""

from .flask_editor import (
    WebEditor,
    find_available_port,
    kill_process_on_port,
    check_port_available,
    render_preview_with_bboxes,
    plot_from_csv,
)

# Legacy aliases for backward compatibility
_find_available_port = find_available_port
_kill_process_on_port = kill_process_on_port


__all__ = [
    "WebEditor",
    "find_available_port",
    "kill_process_on_port",
    "check_port_available",
    "render_preview_with_bboxes",
    "plot_from_csv",
    # Legacy aliases
    "_find_available_port",
    "_kill_process_on_port",
]


# EOF
