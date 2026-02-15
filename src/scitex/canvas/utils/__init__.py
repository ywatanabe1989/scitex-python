#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/utils/__init__.py
"""
Utilities for scitex.canvas.

Includes validation functions and default templates for
common publication formats.
"""

from ._defaults import (  # Constants; Template functions
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    CELL_DOUBLE_COLUMN_MM,
    CELL_SINGLE_COLUMN_MM,
    NATURE_DOUBLE_COLUMN_MM,
    NATURE_FULL_PAGE_MM,
    NATURE_SINGLE_COLUMN_MM,
    PNAS_DOUBLE_COLUMN_MM,
    PNAS_SINGLE_COLUMN_MM,
    SCIENCE_DOUBLE_COLUMN_MM,
    SCIENCE_SINGLE_COLUMN_MM,
    TEMPLATES,
    get_a4_figure,
    get_nature_double_column,
    get_nature_single_column,
    get_presentation_slide,
    get_science_single_column,
    get_square_figure,
    get_template,
    list_templates,
)
from ._validate import (
    check_schema_version,
    validate_axes_layout,
    validate_color,
    validate_json_structure,
    validate_plot_data,
)

__all__ = [
    # Validation
    "validate_json_structure",
    "validate_plot_data",
    "check_schema_version",
    "validate_color",
    "validate_axes_layout",
    # Constants
    "NATURE_SINGLE_COLUMN_MM",
    "NATURE_DOUBLE_COLUMN_MM",
    "NATURE_FULL_PAGE_MM",
    "SCIENCE_SINGLE_COLUMN_MM",
    "SCIENCE_DOUBLE_COLUMN_MM",
    "CELL_SINGLE_COLUMN_MM",
    "CELL_DOUBLE_COLUMN_MM",
    "PNAS_SINGLE_COLUMN_MM",
    "PNAS_DOUBLE_COLUMN_MM",
    "A4_WIDTH_MM",
    "A4_HEIGHT_MM",
    # Templates
    "get_nature_single_column",
    "get_nature_double_column",
    "get_science_single_column",
    "get_a4_figure",
    "get_square_figure",
    "get_presentation_slide",
    "get_template",
    "list_templates",
    "TEMPLATES",
]

# EOF
