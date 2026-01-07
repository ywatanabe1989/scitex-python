#!/usr/bin/env python3
# File: ./src/scitex/schema/__init__.py
# Timestamp: "2025-12-20 (ywatanabe)"
"""
SciTeX Schema Module - DEPRECATED

This module is scheduled for removal. All schemas are being consolidated
in scitex.io.bundle as the single source of truth.

For new code, import from scitex.io.bundle:
    from scitex.io.bundle import FTS, Node, Encoding, Theme, Stats

This module exists temporarily for backward compatibility.
"""

import warnings

warnings.warn(
    "scitex.schema is deprecated. Import from scitex.io.bundle instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Schema version
SCHEMA_VERSION = "0.2.0"

# =============================================================================
# Re-export from scitex.io.bundle (new single source of truth)
# =============================================================================
try:
    from scitex.io.bundle import (  # Core bundle classes; Encoding and Theme; Stats
        FTS,
        BBox,
        DataInfo,
        Encoding,
        Node,
        SizeMM,
        Stats,
        Theme,
    )
except ImportError:
    # FTS not fully configured yet
    FTS = None
    Node = None
    Encoding = None
    Theme = None
    Stats = None

# =============================================================================
# Independent schemas (still defined here temporarily)
# These will be migrated to scitex.io.bundle in subsequent phases
# =============================================================================

# Stats schema (GUI-focused version - to be merged with FTS Stats)
# Encoding schema
from scitex.schema._encoding import (
    ENCODING_VERSION,
    ChannelBinding,
    PlotEncoding,
    TraceEncoding,
)

# Figure elements
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

# Plot schema
from scitex.schema._plot import (
    DPI_FALLBACK,
    PLOT_GEOMETRY_VERSION,
    PLOT_SPEC_VERSION,
    PLOT_STYLE_VERSION,
    AxesLabels,
    AxesLimits,
    AxesSpecItem,
    BboxPx,
    BboxRatio,
    CoordinateSpace,
    DataSourceSpec,
    FontSpec,
    HitRegionEntry,
    LegendLocation,
    LegendSpec,
    PlotGeometry,
    PlotSpec,
    PlotStyle,
    RenderedArtist,
    RenderedAxes,
    RenderManifest,
    SelectableRegion,
    SizeSpec,
    ThemeSpec,
    TraceSpec,
    TraceStyleSpec,
    TraceType,
)
from scitex.schema._stats import (
    Position,
    PositionMode,
    StatPositioning,
    StatResult,
    StatStyling,
    SymbolStyle,
    UnitType,
    create_stat_result,
)

# Theme schema
from scitex.schema._theme import (
    THEME_VERSION,
    ColorScheme,
    LegendStyle,
    LineDefaults,
    MarkerDefaults,
    PlotTheme,
)
from scitex.schema._theme import TraceStyle as ThemeTraceStyle
from scitex.schema._theme import Typography

# Validation (temporarily keep here)
from scitex.schema._validation import (
    ValidationError,
    validate_axes,
    validate_color,
    validate_figure,
    validate_plot,
    validate_stat_result,
)

# =============================================================================
# Public API (maintaining backward compatibility)
# =============================================================================
__all__ = [
    # Version
    "SCHEMA_VERSION",
    # FTS re-exports
    "FTS",
    "Node",
    "BBox",
    "SizeMM",
    "DataInfo",
    "Encoding",
    "Theme",
    "Stats",
    # Plot specs
    "PLOT_SPEC_VERSION",
    "PLOT_STYLE_VERSION",
    "PLOT_GEOMETRY_VERSION",
    "TraceType",
    "CoordinateSpace",
    "LegendLocation",
    "BboxRatio",
    "BboxPx",
    "TraceSpec",
    "AxesLimits",
    "AxesLabels",
    "AxesSpecItem",
    "DataSourceSpec",
    "PlotSpec",
    "TraceStyleSpec",
    "ThemeSpec",
    "FontSpec",
    "SizeSpec",
    "LegendSpec",
    "PlotStyle",
    "RenderedArtist",
    "RenderedAxes",
    "HitRegionEntry",
    "SelectableRegion",
    "PlotGeometry",
    "RenderManifest",
    # Encoding
    "ENCODING_VERSION",
    "ChannelBinding",
    "TraceEncoding",
    "PlotEncoding",
    # Theme
    "THEME_VERSION",
    "ColorScheme",
    "Typography",
    "LineDefaults",
    "MarkerDefaults",
    "ThemeTraceStyle",
    "LegendStyle",
    "PlotTheme",
    # Figure elements
    "FIGURE_ELEMENTS_VERSION",
    "FigureTitle",
    "Caption",
    "PanelLabels",
    "PanelInfo",
    "generate_caption",
    "generate_caption_latex",
    "generate_caption_markdown",
    # Stats
    "PositionMode",
    "UnitType",
    "SymbolStyle",
    "StatResult",
    "StatPositioning",
    "StatStyling",
    "Position",
    "create_stat_result",
    # Validation
    "validate_figure",
    "validate_axes",
    "validate_plot",
    "validate_stat_result",
    "validate_color",
    "ValidationError",
]

# EOF
