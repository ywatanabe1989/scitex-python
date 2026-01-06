#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/__init__.py

"""Utilities for FTS figures - layout, defaults, validation."""

# Constants
from ._const_sizes import (
    A4_HEIGHT_MM,
    A4_WIDTH_MM,
    CELL_DOUBLE_COLUMN_MM,
    CELL_SINGLE_COLUMN_MM,
    DEFAULT_MARGIN_MM,
    DEFAULT_SPACING_MM,
    NATURE_DOUBLE_COLUMN_MM,
    NATURE_FULL_PAGE_MM,
    NATURE_SINGLE_COLUMN_MM,
    PNAS_DOUBLE_COLUMN_MM,
    PNAS_SINGLE_COLUMN_MM,
    SCIENCE_DOUBLE_COLUMN_MM,
    SCIENCE_SINGLE_COLUMN_MM,
)

# Templates
from ._get_template import (
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

# Normalization utilities
from ._normalize import (
    DEFAULT_POSITION,
    DEFAULT_SIZE,
    Position,
    Size,
    normalize_position,
    normalize_size,
)

# Coordinate conversion
from ._convert_coords import to_absolute, to_relative

# Bounds calculation
from ._calc_bounds import (
    Bounds,
    content_bounds,
    element_bounds,
    validate_within_bounds,
)

# Auto layout
from ._auto_layout import auto_crop_layout, auto_layout_grid

# Layout visualization
from ._plot_layout import BLUEPRINT_STYLE, plot_auto_crop_comparison, plot_layout

# Validation
from ._validate import (
    check_schema_version,
    validate_axes_layout,
    validate_color,
    validate_json_structure,
    validate_plot_data,
)

__all__ = [
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
    "DEFAULT_MARGIN_MM",
    "DEFAULT_SPACING_MM",
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
    # Type aliases
    "Position",
    "Size",
    "Bounds",
    "DEFAULT_POSITION",
    "DEFAULT_SIZE",
    # Normalization
    "normalize_position",
    "normalize_size",
    # Coordinate conversion
    "to_absolute",
    "to_relative",
    # Bounds calculation
    "element_bounds",
    "content_bounds",
    "validate_within_bounds",
    # Auto layout
    "auto_layout_grid",
    "auto_crop_layout",
    # Layout visualization
    "plot_layout",
    "plot_auto_crop_comparison",
    "BLUEPRINT_STYLE",
    # Validation
    "validate_json_structure",
    "validate_plot_data",
    "check_schema_version",
    "validate_color",
    "validate_axes_layout",
]

# EOF
