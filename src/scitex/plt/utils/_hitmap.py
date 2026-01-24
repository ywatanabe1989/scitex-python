#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap.py

"""
Hit map generation utilities for interactive element selection.

This module provides functions to generate hit maps for matplotlib figures,
enabling pixel-perfect element selection in web editors and interactive tools.

Supported methods:
1. ID Colors: Single render with unique colors per element (~89ms)
2. Export Path Data: Extract geometry for client-side hit testing (~192ms)

Based on experimental results (see FIGZ_PLTZ_STATSZ.md):
- ID Colors is 33x faster than sequential per-element rendering
- Export Path Data supports reshape/zoom operations

Reserved colors:
- Black (#000000, ID=0): Background/no element
- Dark gray (#010101, ID=65793): Non-selectable axes elements (spines, labels, ticks)

This module re-exports all functions from the _hitmap package for backward
compatibility. The actual implementation is in the _hitmap/ subpackage.
"""

# Re-export all public API from the _hitmap package
from ._hitmap import (
    HITMAP_AXES_COLOR,
    HITMAP_BACKGROUND_COLOR,
    _apply_id_color,
    _id_to_rgb,
    _prepare_hitmap_figure,
    _restore_figure_props,
    _rgb_to_id_lookup,
    _to_native,
    apply_hitmap_colors,
    detect_logical_groups,
    extract_path_data,
    extract_selectable_regions,
    generate_hitmap_id_colors,
    generate_hitmap_with_bbox_tight,
    get_all_artists,
    get_all_artists_with_groups,
    query_hitmap_neighborhood,
    restore_original_colors,
    save_hitmap_png,
)

__all__ = [
    "get_all_artists",
    "get_all_artists_with_groups",
    "detect_logical_groups",
    "generate_hitmap_id_colors",
    "extract_path_data",
    "extract_selectable_regions",
    "query_hitmap_neighborhood",
    "save_hitmap_png",
    "apply_hitmap_colors",
    "restore_original_colors",
    "generate_hitmap_with_bbox_tight",
    "HITMAP_BACKGROUND_COLOR",
    "HITMAP_AXES_COLOR",
    "_rgb_to_id_lookup",
    "_to_native",
    "_id_to_rgb",
    "_apply_id_color",
    "_prepare_hitmap_figure",
    "_restore_figure_props",
]


# EOF
