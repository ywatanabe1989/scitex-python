#!/usr/bin/env python3
# File: ./src/scitex/schema/__init__.py
# Timestamp: "2025-12-17 (ywatanabe)"
"""
SciTeX Schema Module - Central source of truth for cross-module data structures.

This module provides standardized schemas for data that crosses module boundaries:
- Plot specifications (plt ↔ fig ↔ cloud) - Layered architecture for .pltz bundles
- Figure specifications (fig ↔ writer ↔ cloud) - For .figz bundles
- Statistical results (stats ↔ plt ↔ fig)

Design Principles:
- Anything serialized to JSON/disk → defined here
- Anything shared across modules → defined here
- Internal module implementations → stay per-module
- Canonical units: ratio (0-1) for axes bbox, mm for panel size
- px data is ALWAYS derived/cached, never source of truth

Usage:
    # Layered plot schema for .pltz bundles
    from scitex.schema import PltzSpec, PltzStyle, PltzGeometry

    # Figure model for .figz bundles
    from scitex.schema import FigureSpec, StatResult
    from scitex.schema import validate_figure, validate_stat_result
"""

# Schema version for all cross-module specs
SCHEMA_VERSION = "0.2.0"

# =============================================================================
# Plot Schemas - NEW Layered Architecture (SOURCE OF TRUTH for .pltz files)
# =============================================================================
from scitex.fig.model import (
    AnnotationModel as AnnotationSpec,
)
from scitex.fig.model import (
    AxesModel as FigAxesSpec,
)
from scitex.fig.model import (
    AxesStyle as FigAxesStyle,
)

# =============================================================================
# Figure Schemas (re-exported from fig.model for central access)
# These are for figure COMPOSITION (multi-panel), not individual plots
# =============================================================================
from scitex.fig.model import (
    # Core models
    FigureModel as FigureSpec,
)
from scitex.fig.model import (
    GuideModel as GuideSpec,
)
from scitex.fig.model import (
    GuideStyle,
    TextStyle,
    apply_style_to_plots,
    copy_axes_style,
    copy_guide_style,
    # Style utilities
    copy_plot_style,
    copy_text_style,
)
from scitex.fig.model import (
    PlotModel as FigPlotSpec,
)
from scitex.fig.model import (
    # Style models
    PlotStyle as FigPlotStyle,
)

# =============================================================================
# Encoding Schemas - Data to Visual Mapping (NEW)
# =============================================================================
from scitex.schema._encoding import (
    ENCODING_VERSION,
    ChannelBinding,
    PlotEncoding,
    TraceEncoding,
)

# =============================================================================
# Figure Elements Schemas - Title, Caption, Panel Labels (NEW)
# =============================================================================
from scitex.schema._figure_elements import (
    FIGURE_ELEMENTS_VERSION,
    Caption,
    FigureTitle,
    PanelInfo,
    PanelLabels,
    generate_caption,
    generate_caption_latex,
    generate_caption_markdown,
)
from scitex.schema._plot import (
    # DPI fallback for legacy data
    DPI_FALLBACK,
    PLOT_GEOMETRY_VERSION,
    # Version constants
    PLOT_SPEC_VERSION,
    PLOT_STYLE_VERSION,
    BboxPx,
    # Bbox classes
    BboxRatio,
    CoordinateSpace,
    LegendLocation,
    # Type aliases
    TraceType,
)
from scitex.schema._plot import (
    AxesLabels as PltzAxesLabels,
)
from scitex.schema._plot import (
    AxesLimits as PltzAxesLimits,
)
from scitex.schema._plot import (
    AxesSpecItem as PltzAxesItem,
)
from scitex.schema._plot import (
    DataSourceSpec as PltzDataSource,
)
from scitex.schema._plot import (
    FontSpec as PltzFont,
)
from scitex.schema._plot import (
    HitRegionEntry as PltzHitRegion,
)
from scitex.schema._plot import (
    LegendSpec as PltzLegendSpec,
)
from scitex.schema._plot import (
    PlotGeometry as PltzGeometry,
)
from scitex.schema._plot import (
    PlotSpec as PltzSpec,
)
from scitex.schema._plot import (
    PlotStyle as PltzStyle,
)
from scitex.schema._plot import (
    # Geometry classes (cache)
    RenderedArtist as PltzRenderedArtist,
)
from scitex.schema._plot import (
    RenderedAxes as PltzRenderedAxes,
)
from scitex.schema._plot import (
    # Manifest
    RenderManifest as PltzRenderManifest,
)
from scitex.schema._plot import (
    SelectableRegion as PltzSelectableRegion,
)
from scitex.schema._plot import (
    SizeSpec as PltzSize,
)
from scitex.schema._plot import (
    ThemeSpec as PltzTheme,
)
from scitex.schema._plot import (
    # Spec classes (canonical) - prefixed with Pltz to avoid collision
    TraceSpec as PltzTraceSpec,
)
from scitex.schema._plot import (
    # Style classes
    TraceStyleSpec as PltzTraceStyle,
)

# =============================================================================
# Statistical Result Schemas (SOURCE OF TRUTH)
# =============================================================================
from scitex.schema._stats import (
    Position,
    # Type aliases
    PositionMode,
    StatPositioning,
    # Classes
    StatResult,
    StatStyling,
    SymbolStyle,
    UnitType,
    # Convenience function
    create_stat_result,
)

# =============================================================================
# Theme Schemas - Pure Aesthetics (NEW)
# =============================================================================
from scitex.schema._theme import (
    THEME_VERSION,
    ColorScheme,
    LegendStyle,
    LineDefaults,
    MarkerDefaults,
    PlotTheme,
    Typography,
)
from scitex.schema._theme import (
    TraceStyle as ThemeTraceStyle,
)

# =============================================================================
# Validation Functions
# =============================================================================
from scitex.schema._validation import (
    ValidationError,
    validate_axes,
    validate_color,
    validate_figure,
    validate_plot,
    validate_stat_result,
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
    # Encoding Schemas (Data to Visual Mapping)
    # ==========================================================================
    "ENCODING_VERSION",
    "ChannelBinding",
    "TraceEncoding",
    "PlotEncoding",
    # ==========================================================================
    # Theme Schemas (Pure Aesthetics)
    # ==========================================================================
    "THEME_VERSION",
    "ColorScheme",
    "Typography",
    "LineDefaults",
    "MarkerDefaults",
    "ThemeTraceStyle",
    "LegendStyle",
    "PlotTheme",
    # ==========================================================================
    # Figure Elements (Title, Caption, Panel Labels)
    # ==========================================================================
    "FIGURE_ELEMENTS_VERSION",
    "FigureTitle",
    "Caption",
    "PanelLabels",
    "PanelInfo",
    "generate_caption",
    "generate_caption_latex",
    "generate_caption_markdown",
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
    # Validation
    # ==========================================================================
    "validate_figure",
    "validate_axes",
    "validate_plot",
    "validate_stat_result",
    "validate_color",
    "ValidationError",
]


# EOF
