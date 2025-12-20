#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/editor/cui/__init__.py

"""CUI module for launching visual figure editors."""

from ._backend_detector import detect_best_backend, print_available_backends
from ._bundle_resolver import (
    resolve_figz_bundle,
    resolve_layered_pltz_bundle,
    resolve_pltz_bundle,
    resolve_stx_bundle,
)
from ._editor_launcher import edit
from ._manual_handler import compute_file_hash, save_manual_overrides
from ._panel_loader import load_panel_data
from ._path_resolver import resolve_figure_paths

__all__ = [
    "edit",
    "detect_best_backend",
    "print_available_backends",
    "resolve_figure_paths",
    "resolve_pltz_bundle",
    "resolve_figz_bundle",
    "resolve_layered_pltz_bundle",
    "resolve_stx_bundle",
    "load_panel_data",
    "compute_file_hash",
    "save_manual_overrides",
]

# EOF
