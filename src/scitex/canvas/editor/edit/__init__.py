#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/edit/__init__.py

"""Edit module for launching visual figure editors.

This module provides the main edit() function and supporting utilities for:
- Backend detection and selection (flask, dearpygui, qt, tkinter, mpl)
- Path resolution for figure files (.json, .csv, .png)
- Bundle resolution for .plot and .figure formats
- Panel data loading for multi-panel figures
- Manual override handling (.manual.json)
"""

from .backend_detector import detect_best_backend, print_available_backends
from .bundle_resolver import (
    resolve_figure_bundle,
    resolve_layered_plot_bundle,
    resolve_plot_bundle,
)
from .editor_launcher import edit
from .manual_handler import compute_file_hash, save_manual_overrides
from .panel_loader import load_panel_data
from .path_resolver import resolve_figure_paths

__all__ = [
    # Main entry point
    "edit",
    # Backend detection
    "detect_best_backend",
    "print_available_backends",
    # Path resolution
    "resolve_figure_paths",
    # Bundle resolution
    "resolve_plot_bundle",
    "resolve_figure_bundle",
    "resolve_layered_plot_bundle",
    # Panel loading
    "load_panel_data",
    # Manual overrides
    "compute_file_hash",
    "save_manual_overrides",
]


# EOF
