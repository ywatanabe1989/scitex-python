#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/__init__.py

"""
Figure metadata collection package.

This package provides utilities to collect comprehensive metadata from matplotlib
figures and axes for embedding in saved images.

Public API
----------
collect_figure_metadata : function
    Main function to collect all metadata from a figure
collect_recipe_metadata : function
    Collect metadata with reconstruction recipe
assert_csv_json_consistency : function
    Assert CSV columns match JSON metadata
verify_csv_json_consistency : function
    Verify CSV-JSON consistency and return detailed results
export_editable_figure : function
    Export figure with geometry data for interactive editing (schema v0.3)
"""

# Import public API from core module
from ._core import collect_figure_metadata
from ._data_linkage import (
    assert_csv_json_consistency,
    verify_csv_json_consistency,
    collect_recipe_metadata,
)
from ._geometry_extraction import (
    extract_axes_bbox_px,
    data_to_axes_px,
    extract_line_geometry,
    extract_scatter_geometry,
    extract_polygon_geometry,
    extract_rectangle_geometry,
    extract_bar_group_geometry,
    extract_text_geometry,
    extract_image_geometry,
)
from ._editable_export import export_editable_figure

__all__ = [
    "collect_figure_metadata",
    "collect_recipe_metadata",
    "assert_csv_json_consistency",
    "verify_csv_json_consistency",
    "export_editable_figure",
    # Geometry extraction
    "extract_axes_bbox_px",
    "data_to_axes_px",
    "extract_line_geometry",
    "extract_scatter_geometry",
    "extract_polygon_geometry",
    "extract_rectangle_geometry",
    "extract_bar_group_geometry",
    "extract_text_geometry",
    "extract_image_geometry",
]
