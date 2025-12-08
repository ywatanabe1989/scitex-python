#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/model/plot_types.py
"""
Specialized plot type configurations with proper properties.

This module provides dataclasses for each plot type with their specific
properties for:
- Backend validation
- UI property panels
- File round-trip
- Diff analysis
"""

from typing import Optional, List, Union, Any
from dataclasses import dataclass, field


# ============================================================================
# MATPLOTLIB BASIC PLOTS
# ============================================================================


@dataclass
class LinePlotConfig:
    """Line plot configuration."""

    # Data (required)
    x: List[float]
    y: List[float]

    # Style
    color: Optional[str] = None
    linewidth: Optional[float] = None  # or linewidth_mm for scitex
    linestyle: Optional[str] = "-"  # "-", "--", "-.", ":"
    marker: Optional[str] = None  # "o", "s", "^", etc.
    markersize: Optional[float] = None
    alpha: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID for tracking
    id: Optional[str] = None


@dataclass
class ScatterPlotConfig:
    """Scatter plot configuration."""

    # Data (required)
    x: List[float]
    y: List[float]

    # Style
    color: Optional[str] = None
    size_mm: Optional[float] = None  # scitex-specific: marker size in mm
    s: Optional[float] = None  # matplotlib: marker size in points²
    marker: Optional[str] = "o"
    alpha: Optional[float] = None
    cmap: Optional[str] = None  # For color-mapped scatter

    # Color data (optional)
    c: Optional[Union[str, List[float]]] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class BarPlotConfig:
    """Bar plot configuration."""

    # Data (required)
    x: Union[List[float], List[str]]  # Can be categorical
    height: List[float]  # or y

    # Style
    width: Optional[float] = 0.8
    color: Optional[str] = None
    alpha: Optional[float] = None
    edge_thickness_mm: Optional[float] = None  # scitex-specific
    edgecolor: Optional[str] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class BarHPlotConfig:
    """Horizontal bar plot configuration."""

    # Data (required)
    y: Union[List[float], List[str]]  # Categories
    width: List[float]  # Bar widths

    # Style
    height: Optional[float] = 0.8  # Bar height (thickness)
    color: Optional[str] = None
    alpha: Optional[float] = None
    edge_thickness_mm: Optional[float] = None
    edgecolor: Optional[str] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class HistPlotConfig:
    """Histogram configuration."""

    # Data (required)
    x: List[float]

    # Histogram parameters
    bins: Union[int, List[float]] = 10
    density: bool = False
    cumulative: bool = False
    histtype: str = "bar"  # "bar", "barstacked", "step", "stepfilled"

    # Style
    color: Optional[str] = None
    alpha: Optional[float] = None
    edgecolor: Optional[str] = None

    # Range
    range: Optional[tuple] = None  # (min, max)

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class BoxPlotConfig:
    """Box plot configuration."""

    # Data (required)
    data: Union[List[List[float]], List[float]]  # Multiple groups or single

    # Box parameters
    labels: Optional[List[str]] = None
    positions: Optional[List[float]] = None
    widths: Optional[float] = None

    # Style
    linewidth_mm: Optional[float] = None  # scitex-specific
    showfliers: bool = True  # Show outliers
    showmeans: bool = False

    # ID
    id: Optional[str] = None


@dataclass
class ErrorbarPlotConfig:
    """Error bar plot configuration."""

    # Data (required)
    x: List[float]
    y: List[float]

    # Error data
    xerr: Optional[Union[float, List[float]]] = None
    yerr: Optional[Union[float, List[float]]] = None

    # Style
    fmt: str = "o-"  # Format string
    color: Optional[str] = None
    capsize: Optional[float] = None
    capthick: Optional[float] = None
    thickness_mm: Optional[float] = None  # scitex-specific: error bar thickness
    cap_width_mm: Optional[float] = None  # scitex-specific: cap width
    alpha: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class FillBetweenConfig:
    """Fill between configuration."""

    # Data (required)
    x: List[float]
    y1: List[float]
    y2: List[float]

    # Style
    color: Optional[str] = None
    alpha: Optional[float] = 0.3
    linewidth: Optional[float] = None
    edgecolor: Optional[str] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class ImshowConfig:
    """Image display configuration."""

    # Data (required)
    img: Any  # 2D array

    # Display parameters
    cmap: Optional[str] = "viridis"
    aspect: str = "auto"  # "auto", "equal", or numeric
    interpolation: Optional[str] = None  # "nearest", "bilinear", etc.

    # Value range
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Origin
    origin: str = "upper"  # "upper" or "lower"

    # ID
    id: Optional[str] = None


@dataclass
class ContourConfig:
    """Contour plot configuration."""

    # Data (required)
    x: List[float]
    y: List[float]
    z: Any  # 2D array

    # Contour parameters
    levels: Optional[Union[int, List[float]]] = None
    filled: bool = False  # False for contour, True for contourf

    # Style
    cmap: Optional[str] = None
    colors: Optional[str] = None
    linewidths: Optional[float] = None
    alpha: Optional[float] = None

    # Value range
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # ID
    id: Optional[str] = None


@dataclass
class ViolinPlotConfig:
    """Violin plot configuration."""

    # Data (required)
    data: List[List[float]]  # Multiple groups

    # Violin parameters
    positions: Optional[List[float]] = None
    widths: Optional[float] = 0.5
    showmeans: bool = False
    showmedians: bool = False
    showextrema: bool = True

    # ID
    id: Optional[str] = None


# ============================================================================
# CUSTOM SCITEX PLOTS
# ============================================================================


@dataclass
class HeatmapConfig:
    """Heatmap configuration (scitex.plt.ax.stx_heatmap)."""

    # Data (required)
    data: Any  # 2D array

    # Labels
    x_labels: Optional[List[str]] = None
    y_labels: Optional[List[str]] = None

    # Colorbar
    cbar_label: Optional[str] = None
    cmap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None

    # Annotations
    show_annot: bool = False
    value_format: str = "{x:.2f}"

    # ID
    id: Optional[str] = None


@dataclass
class PlotLineConfig:
    """Plot line configuration (scitex.plt.ax.stx_line)."""

    # Data (required - single array)
    y: List[float]
    x: Optional[List[float]] = None  # If None, uses indices

    # Style
    color: Optional[str] = None
    linewidth_mm: Optional[float] = None
    linestyle: Optional[str] = "-"

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class ShadedLineConfig:
    """Shaded line configuration (scitex.plt.ax.stx_shaded_line)."""

    # Data (required)
    x: List[float]
    y_lower: List[float]
    y_middle: List[float]
    y_upper: List[float]

    # Style
    color: Optional[str] = None
    alpha: Optional[float] = 0.3
    linewidth_mm: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class ViolinConfig:
    """Violin plot configuration (scitex.plt.ax.stx_violin)."""

    # Data (required)
    data: List[List[float]]
    labels: Optional[List[str]] = None

    # Style
    colors: Optional[List[str]] = None

    # ID
    id: Optional[str] = None


@dataclass
class ECDFConfig:
    """ECDF configuration (scitex.plt.ax.stx_ecdf)."""

    # Data (required)
    data: List[float]

    # Style
    color: Optional[str] = None
    linewidth_mm: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class BoxConfig:
    """Box plot configuration (scitex.plt.ax.stx_box)."""

    # Data (required)
    data: List[float]

    # Style
    color: Optional[str] = None
    linewidth_mm: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class MeanStdConfig:
    """Mean±Std configuration (scitex.plt.ax.stx_mean_std)."""

    # Data (required)
    y_mean: List[float]
    xx: Optional[List[float]] = None  # X values
    sd: Union[float, List[float]] = 1.0  # Standard deviation

    # Style
    color: Optional[str] = None
    alpha: Optional[float] = 0.3

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class KDEConfig:
    """KDE configuration (scitex.plt.ax.stx_kde)."""

    # Data (required)
    data: List[float]

    # KDE parameters
    bw_method: Optional[str] = None  # Bandwidth method

    # Style
    color: Optional[str] = None
    linewidth_mm: Optional[float] = None

    # Label
    label: Optional[str] = None

    # ID
    id: Optional[str] = None


# ============================================================================
# SEABORN PLOTS
# ============================================================================


@dataclass
class SeabornBoxplotConfig:
    """Seaborn boxplot configuration (sns.boxplot via scitex)."""

    # Data specification
    x: Optional[str] = None  # Column name
    y: Optional[str] = None  # Column name
    data: Optional[Any] = None  # DataFrame

    # Style
    hue: Optional[str] = None
    palette: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornViolinplotConfig:
    """Seaborn violinplot configuration."""

    # Data specification
    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    palette: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornScatterplotConfig:
    """Seaborn scatterplot configuration."""

    # Data specification
    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    size: Optional[str] = None
    style: Optional[str] = None
    palette: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornLineplotConfig:
    """Seaborn lineplot configuration."""

    # Data specification
    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    style: Optional[str] = None
    palette: Optional[str] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornHistplotConfig:
    """Seaborn histplot configuration."""

    # Data specification
    x: Optional[str] = None
    data: Optional[Any] = None

    # Histogram parameters
    hue: Optional[str] = None
    bins: Union[int, str] = "auto"
    kde: bool = False
    alpha: Optional[float] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornBarplotConfig:
    """Seaborn barplot configuration."""

    # Data specification
    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    palette: Optional[str] = None
    estimator: str = "mean"  # "mean", "median", "sum", etc.

    # ID
    id: Optional[str] = None


@dataclass
class SeabornStripplotConfig:
    """Seaborn stripplot configuration."""

    # Data specification
    x: Optional[str] = None
    y: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    alpha: Optional[float] = None

    # ID
    id: Optional[str] = None


@dataclass
class SeabornKDEplotConfig:
    """Seaborn KDE plot configuration."""

    # Data specification
    x: Optional[str] = None
    data: Optional[Any] = None

    # Style
    hue: Optional[str] = None
    fill: bool = False

    # Note: ID parameter may have issues with seaborn wrapper


# ============================================================================
# PLOT TYPE REGISTRY
# ============================================================================

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
    "contourf": ContourConfig,  # Same config, filled=True
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


def get_plot_config_class(plot_type: str):
    """
    Get the configuration dataclass for a plot type.

    Parameters
    ----------
    plot_type : str
        Plot type name

    Returns
    -------
    type
        Configuration dataclass for the plot type

    Raises
    ------
    ValueError
        If plot type is not supported

    Examples
    --------
    >>> config_class = get_plot_config_class("line")
    >>> config = config_class(x=[0,1,2], y=[0,1,4], color="blue")
    """
    if plot_type not in PLOT_TYPE_CONFIGS:
        raise ValueError(
            f"Unknown plot type: {plot_type}. "
            f"Supported: {list(PLOT_TYPE_CONFIGS.keys())}"
        )

    return PLOT_TYPE_CONFIGS[plot_type]


def list_plot_types() -> List[str]:
    """
    List all supported plot types.

    Returns
    -------
    List[str]
        List of plot type names

    Examples
    --------
    >>> plot_types = list_plot_types()
    >>> "line" in plot_types
    True
    """
    return list(PLOT_TYPE_CONFIGS.keys())


# EOF
