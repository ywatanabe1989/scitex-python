#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/schema/__init__.py
# Timestamp: "2025-12-13 (ywatanabe)"
"""
SciTeX Schema Module - Central source of truth for cross-module data structures.

This module provides standardized schemas for data that crosses module boundaries:
- Plot specifications (plt ↔ fig ↔ cloud) - NEW layered architecture
- Figure specifications (fig ↔ writer ↔ cloud)
- Statistical results (stats ↔ plt ↔ fig)
- Canvas metadata (fig ↔ io ↔ cloud)

Design Principles:
- Anything serialized to JSON/disk → defined here
- Anything shared across modules → defined here
- Internal module implementations → stay per-module
- Canonical units: ratio (0-1) for axes bbox, mm for panel size
- px data is ALWAYS derived/cached, never source of truth

Usage:
    # New layered plot schema (recommended)
    from scitex.schema import PltzSpec, PltzStyle, PltzGeometry

    # Legacy figure model
    from scitex.schema import FigureSpec, StatResult, CanvasSpec
    from scitex.schema import validate_figure, validate_stat_result
"""

# Schema version for all cross-module specs
SCHEMA_VERSION = "0.2.0"

# =============================================================================
# Plot Schemas - NEW Layered Architecture (SOURCE OF TRUTH for .pltz files)
# =============================================================================
from scitex.schema._plot import (
    # Version constants
    PLOT_SPEC_VERSION,
    PLOT_STYLE_VERSION,
    PLOT_GEOMETRY_VERSION,
    # DPI fallback for legacy data
    DPI_FALLBACK,
    # Type aliases
    TraceType,
    CoordinateSpace,
    LegendLocation,
    # Bbox classes
    BboxRatio,
    BboxPx,
    # Spec classes (canonical) - prefixed with Pltz to avoid collision
    TraceSpec as PltzTraceSpec,
    AxesLimits as PltzAxesLimits,
    AxesLabels as PltzAxesLabels,
    AxesSpecItem as PltzAxesItem,
    DataSourceSpec as PltzDataSource,
    PlotSpec as PltzSpec,
    # Style classes
    TraceStyleSpec as PltzTraceStyle,
    ThemeSpec as PltzTheme,
    FontSpec as PltzFont,
    SizeSpec as PltzSize,
    LegendSpec as PltzLegendSpec,
    PlotStyle as PltzStyle,
    # Geometry classes (cache)
    RenderedArtist as PltzRenderedArtist,
    RenderedAxes as PltzRenderedAxes,
    HitRegionEntry as PltzHitRegion,
    SelectableRegion as PltzSelectableRegion,
    PlotGeometry as PltzGeometry,
    # Manifest
    RenderManifest as PltzRenderManifest,
)

# =============================================================================
# Figure Schemas (re-exported from fig.model for central access)
# These are for figure COMPOSITION (multi-panel), not individual plots
# =============================================================================
from scitex.fig.model import (
    # Core models
    FigureModel as FigureSpec,
    AxesModel as FigAxesSpec,
    PlotModel as FigPlotSpec,
    AnnotationModel as AnnotationSpec,
    GuideModel as GuideSpec,
    # Style models
    PlotStyle as FigPlotStyle,
    AxesStyle as FigAxesStyle,
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
    # ==========================================================================
    # NEW: Plot Schemas (layered architecture for .pltz files)
    # ==========================================================================
    # Version constants
    "PLOT_SPEC_VERSION",
    "PLOT_STYLE_VERSION",
    "PLOT_GEOMETRY_VERSION",
    # Type aliases
    "TraceType",
    "CoordinateSpace",
    "LegendLocation",
    # Bbox classes
    "BboxRatio",
    "BboxPx",
    # Spec classes (canonical)
    "PltzTraceSpec",
    "PltzAxesLimits",
    "PltzAxesLabels",
    "PltzAxesItem",
    "PltzDataSource",
    "PltzSpec",
    # Style classes
    "PltzTraceStyle",
    "PltzTheme",
    "PltzFont",
    "PltzSize",
    "PltzLegendSpec",
    "PltzStyle",
    # Geometry classes (cache)
    "PltzRenderedArtist",
    "PltzRenderedAxes",
    "PltzHitRegion",
    "PltzSelectableRegion",
    "PltzGeometry",
    # Manifest
    "PltzRenderManifest",
    # ==========================================================================
    # Figure specs (for multi-panel composition)
    # ==========================================================================
    "FigureSpec",
    "FigAxesSpec",
    "FigPlotSpec",
    "AnnotationSpec",
    "GuideSpec",
    # Style specs
    "FigPlotStyle",
    "FigAxesStyle",
    "GuideStyle",
    "TextStyle",
    # Style utilities
    "copy_plot_style",
    "copy_axes_style",
    "copy_guide_style",
    "copy_text_style",
    "apply_style_to_plots",
    # ==========================================================================
    # Stats type aliases
    # ==========================================================================
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    # Stats specs
    "StatResult",
    "StatPositioning",
    "StatStyling",
    "Position",
    "create_stat_result",
    # ==========================================================================
    # Canvas specs
    # ==========================================================================
    "CanvasSpec",
    "PanelSpec",
    "CanvasAnnotationSpec",
    "CanvasTitleSpec",
    "CanvasCaptionSpec",
    "CanvasMetadataSpec",
    "DataFileSpec",
    # ==========================================================================
    # Validation
    # ==========================================================================
    "validate_figure",
    "validate_axes",
    "validate_plot",
    "validate_stat_result",
    "validate_canvas",
    "validate_color",
    "ValidationError",
]


# EOF
