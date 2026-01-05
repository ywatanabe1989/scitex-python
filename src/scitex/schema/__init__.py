#!/usr/bin/env python3
# File: ./src/scitex/schema/__init__.py
# Timestamp: "2025-12-20 (ywatanabe)"
"""
SciTeX Schema Module - DEPRECATED

This module is scheduled for removal. All schemas are being consolidated
in scitex.fts as the single source of truth.

For new code, import from scitex.fts:
    from scitex.fts import FTS, Node, Encoding, Theme, Stats

This module exists temporarily for backward compatibility.
"""

import warnings

warnings.warn(
    "scitex.schema is deprecated. Import from scitex.fts instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Schema version
SCHEMA_VERSION = "0.2.0"

# =============================================================================
# Re-export from scitex.fts (new single source of truth)
# =============================================================================
try:
    from scitex.fts import (
        # Core bundle classes
        FTS,
        Node,
        BBox,
        SizeMM,
        DataInfo,
        # Encoding and Theme
        Encoding,
        Theme,
        # Stats
        Stats,
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
# These will be migrated to scitex.fts in subsequent phases
# =============================================================================

# Stats schema (GUI-focused version - to be merged with FTS Stats)
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

# Theme schema
from scitex.schema._theme import (
    THEME_VERSION,
    ColorScheme,
    LegendStyle,
    LineDefaults,
    MarkerDefaults,
    PlotTheme,
    Typography,
)
from scitex.schema._theme import TraceStyle as ThemeTraceStyle

# Plot schema
from scitex.schema._plot import (
    DPI_FALLBACK,
    PLOT_GEOMETRY_VERSION,
    PLOT_SPEC_VERSION,
    PLOT_STYLE_VERSION,
    BboxPx,
    BboxRatio,
    CoordinateSpace,
    LegendLocation,
    TraceType,
)
from scitex.schema._plot import AxesLabels as PltzAxesLabels
from scitex.schema._plot import AxesLimits as PltzAxesLimits
from scitex.schema._plot import AxesSpecItem as PltzAxesItem
from scitex.schema._plot import DataSourceSpec as PltzDataSource
from scitex.schema._plot import FontSpec as PltzFont
from scitex.schema._plot import HitRegionEntry as PltzHitRegion
from scitex.schema._plot import LegendSpec as PltzLegendSpec
from scitex.schema._plot import PlotGeometry as PltzGeometry
from scitex.schema._plot import PlotSpec as PltzSpec
from scitex.schema._plot import PlotStyle as PltzStyle
from scitex.schema._plot import RenderedArtist as PltzRenderedArtist
from scitex.schema._plot import RenderedAxes as PltzRenderedAxes
from scitex.schema._plot import RenderManifest as PltzRenderManifest
from scitex.schema._plot import SelectableRegion as PltzSelectableRegion
from scitex.schema._plot import SizeSpec as PltzSize
from scitex.schema._plot import ThemeSpec as PltzTheme
from scitex.schema._plot import TraceSpec as PltzTraceSpec
from scitex.schema._plot import TraceStyleSpec as PltzTraceStyle

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
    "PltzTraceSpec",
    "PltzAxesLimits",
    "PltzAxesLabels",
    "PltzAxesItem",
    "PltzDataSource",
    "PltzSpec",
    "PltzTraceStyle",
    "PltzTheme",
    "PltzFont",
    "PltzSize",
    "PltzLegendSpec",
    "PltzStyle",
    "PltzRenderedArtist",
    "PltzRenderedAxes",
    "PltzHitRegion",
    "PltzSelectableRegion",
    "PltzGeometry",
    "PltzRenderManifest",
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
