#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_hitmap/__init__.py

"""
Hit map generation utilities for interactive element selection.

This package provides functions to generate hit maps for matplotlib figures,
enabling pixel-perfect element selection in web editors and interactive tools.

Supported methods:
1. ID Colors: Single render with unique colors per element (~89ms)
2. Export Path Data: Extract geometry for client-side hit testing (~192ms)

Reserved colors:
- Black (#000000, ID=0): Background/no element
- Dark gray (#010101, ID=65793): Non-selectable axes elements (spines, labels, ticks)
"""

from ._artist_extraction import (
    detect_logical_groups,
    get_all_artists,
    get_all_artists_with_groups,
)
from ._color_application import (
    apply_hitmap_colors,
    apply_id_color,
    prepare_hitmap_figure,
    restore_figure_props,
    restore_original_colors,
)
from ._color_conversion import id_to_rgb, rgb_to_id, rgb_to_id_lookup
from ._constants import HITMAP_AXES_COLOR, HITMAP_BACKGROUND_COLOR, to_native
from ._hitmap_core import generate_hitmap_id_colors, generate_hitmap_with_bbox_tight
from ._path_extraction import extract_path_data, extract_selectable_regions
from ._query import query_hitmap_neighborhood, save_hitmap_png

# Backward compatibility aliases
_to_native = to_native
_id_to_rgb = id_to_rgb
_rgb_to_id = rgb_to_id
_rgb_to_id_lookup = rgb_to_id_lookup
_apply_id_color = apply_id_color
_prepare_hitmap_figure = prepare_hitmap_figure
_restore_figure_props = restore_figure_props

__all__ = [
    # Constants
    "HITMAP_BACKGROUND_COLOR",
    "HITMAP_AXES_COLOR",
    # Artist extraction
    "get_all_artists",
    "get_all_artists_with_groups",
    "detect_logical_groups",
    # Color conversion
    "id_to_rgb",
    "rgb_to_id",
    "rgb_to_id_lookup",
    # Core hitmap generation
    "generate_hitmap_id_colors",
    "generate_hitmap_with_bbox_tight",
    # Path extraction
    "extract_path_data",
    "extract_selectable_regions",
    # Query and save
    "query_hitmap_neighborhood",
    "save_hitmap_png",
    # Color application
    "apply_hitmap_colors",
    "restore_original_colors",
    # Backward compatibility
    "_to_native",
    "_id_to_rgb",
    "_rgb_to_id",
    "_rgb_to_id_lookup",
    "_apply_id_color",
    "_prepare_hitmap_figure",
    "_restore_figure_props",
]


# EOF
