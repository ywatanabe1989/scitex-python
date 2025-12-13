#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/schema/__init__.py
# Time-stamp: "2024-12-09 08:15:00 (ywatanabe)"
"""
SciTeX Schema Module - Central source of truth for cross-module data structures.

This module provides standardized schemas for data that crosses module boundaries:
- Figure specifications (vis ↔ writer ↔ cloud)
- Statistical results (stats ↔ plt ↔ vis)
- Canvas metadata (vis ↔ io ↔ cloud)

Design Principles:
- Anything serialized to JSON/disk → defined here
- Anything shared across modules → defined here
- Internal module implementations → stay per-module

Usage:
    from scitex.schema import FigureSpec, StatResult, CanvasSpec
    from scitex.schema import validate_figure, validate_stat_result
"""

# Schema version for all cross-module specs
SCHEMA_VERSION = "0.1.0"

# =============================================================================
# Figure Schemas (re-exported from vis.model for central access)
# =============================================================================
from scitex.fig.model import (
    # Core models
    FigureModel as FigureSpec,
    AxesModel as AxesSpec,
    PlotModel as PlotSpec,
    AnnotationModel as AnnotationSpec,
    GuideModel as GuideSpec,
    # Style models
    PlotStyle,
    AxesStyle,
    GuideStyle,
    TextStyle,
    # Style utilities
    copy_plot_style,
    copy_axes_style,
    copy_guide_style,
    copy_text_style,
    apply_style_to_plots,
)

# =============================================================================
# Statistical Result Schemas (SOURCE OF TRUTH)
# =============================================================================
from scitex.schema._stats import (
    # Type aliases
    PositionMode,
    UnitType,
    SymbolStyle,
    # Classes
    StatResult,
    StatPositioning,
    StatStyling,
    Position,
    # Convenience function
    create_stat_result,
)

# =============================================================================
# Canvas Schemas
# =============================================================================
from scitex.schema._canvas import (
    CanvasSpec,
    PanelSpec,
    CanvasAnnotationSpec,
    CanvasTitleSpec,
    CanvasCaptionSpec,
    CanvasMetadataSpec,
    DataFileSpec,
)

# =============================================================================
# Validation Functions
# =============================================================================
from scitex.schema._validation import (
    validate_figure,
    validate_axes,
    validate_plot,
    validate_stat_result,
    validate_canvas,
    validate_color,
    ValidationError,
)

# =============================================================================
# Public API
# =============================================================================
__all__ = [
    # Version
    "SCHEMA_VERSION",
    # Figure specs
    "FigureSpec",
    "AxesSpec",
    "PlotSpec",
    "AnnotationSpec",
    "GuideSpec",
    # Style specs
    "PlotStyle",
    "AxesStyle",
    "GuideStyle",
    "TextStyle",
    # Style utilities
    "copy_plot_style",
    "copy_axes_style",
    "copy_guide_style",
    "copy_text_style",
    "apply_style_to_plots",
    # Stats type aliases
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # Stats specs
    "StatResult",
    "StatPositioning",
    "StatStyling",
    "Position",
    "create_stat_result",
    # Canvas specs
    "CanvasSpec",
    "PanelSpec",
    "CanvasAnnotationSpec",
    "CanvasTitleSpec",
    "CanvasCaptionSpec",
    "CanvasMetadataSpec",
    "DataFileSpec",
    # Validation
    "validate_figure",
    "validate_axes",
    "validate_plot",
    "validate_stat_result",
    "validate_canvas",
    "validate_color",
    "ValidationError",
]


# EOF
