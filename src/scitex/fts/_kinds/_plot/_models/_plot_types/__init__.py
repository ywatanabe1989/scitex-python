#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/_plot_types/__init__.py

"""Plot type configurations and registry."""

from typing import List, Type

# Line plots
from ._line import LinePlotConfig, PlotLineConfig, ShadedLineConfig

# Scatter plots
from ._scatter import ScatterPlotConfig

# Bar plots
from ._bar import BarHPlotConfig, BarPlotConfig

# Histograms
from ._histogram import HistPlotConfig

# Box plots
from ._box import BoxConfig, BoxPlotConfig

# Violin plots
from ._violin import ViolinConfig, ViolinPlotConfig

# Error bars and fill
from ._errorbar import ErrorbarPlotConfig, FillBetweenConfig, MeanStdConfig

# Image plots
from ._image import ContourConfig, HeatmapConfig, ImshowConfig

# Distribution plots
from ._distribution import ECDFConfig, KDEConfig

# Seaborn plots
from ._seaborn import (
    SeabornBarplotConfig,
    SeabornBoxplotConfig,
    SeabornHistplotConfig,
    SeabornKDEplotConfig,
    SeabornLineplotConfig,
    SeabornScatterplotConfig,
    SeabornStripplotConfig,
    SeabornViolinplotConfig,
)

# =============================================================================
# PLOT TYPE REGISTRY
# =============================================================================

PLOT_TYPE_CONFIGS = {
    # Matplotlib basic
    "line": LinePlotConfig,
    "scatter": ScatterPlotConfig,
    "bar": BarPlotConfig,
    "barh": BarHPlotConfig,
    "hist": HistPlotConfig,
    "boxplot": BoxPlotConfig,
    "errorbar": ErrorbarPlotConfig,
    "fill_between": FillBetweenConfig,
    "imshow": ImshowConfig,
    "contour": ContourConfig,
    "contourf": ContourConfig,
    "violinplot": ViolinPlotConfig,
    # Custom scitex
    "heatmap": HeatmapConfig,
    "stx_line": PlotLineConfig,
    "stx_shaded_line": ShadedLineConfig,
    "stx_violin": ViolinConfig,
    "stx_ecdf": ECDFConfig,
    "stx_box": BoxConfig,
    "stx_mean_std": MeanStdConfig,
    "stx_kde": KDEConfig,
    # Seaborn
    "sns_boxplot": SeabornBoxplotConfig,
    "sns_violinplot": SeabornViolinplotConfig,
    "sns_scatterplot": SeabornScatterplotConfig,
    "sns_lineplot": SeabornLineplotConfig,
    "sns_histplot": SeabornHistplotConfig,
    "sns_barplot": SeabornBarplotConfig,
    "sns_stripplot": SeabornStripplotConfig,
    "sns_kdeplot": SeabornKDEplotConfig,
}


def get_plot_config_class(plot_type: str) -> Type:
    """Get the configuration dataclass for a plot type.

    Args:
        plot_type: Plot type name

    Returns:
        Configuration dataclass for the plot type

    Raises:
        ValueError: If plot type is not supported
    """
    if plot_type not in PLOT_TYPE_CONFIGS:
        raise ValueError(
            f"Unknown plot type: {plot_type}. "
            f"Supported: {list(PLOT_TYPE_CONFIGS.keys())}"
        )
    return PLOT_TYPE_CONFIGS[plot_type]


def list_plot_types() -> List[str]:
    """List all supported plot types."""
    return list(PLOT_TYPE_CONFIGS.keys())


__all__ = [
    # Registry
    "PLOT_TYPE_CONFIGS",
    "get_plot_config_class",
    "list_plot_types",
    # Line plots
    "LinePlotConfig",
    "PlotLineConfig",
    "ShadedLineConfig",
    # Scatter
    "ScatterPlotConfig",
    # Bar
    "BarPlotConfig",
    "BarHPlotConfig",
    # Histogram
    "HistPlotConfig",
    # Box
    "BoxPlotConfig",
    "BoxConfig",
    # Violin
    "ViolinPlotConfig",
    "ViolinConfig",
    # Error/Fill
    "ErrorbarPlotConfig",
    "FillBetweenConfig",
    "MeanStdConfig",
    # Image
    "ImshowConfig",
    "ContourConfig",
    "HeatmapConfig",
    # Distribution
    "ECDFConfig",
    "KDEConfig",
    # Seaborn
    "SeabornBoxplotConfig",
    "SeabornViolinplotConfig",
    "SeabornScatterplotConfig",
    "SeabornLineplotConfig",
    "SeabornHistplotConfig",
    "SeabornBarplotConfig",
    "SeabornStripplotConfig",
    "SeabornKDEplotConfig",
]

# EOF
