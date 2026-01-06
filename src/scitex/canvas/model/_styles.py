#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/model/styles.py
"""
Style dataclasses for scitex.canvas objects.

Separating style properties from data makes:
- UI property panels easy to auto-generate
- Style copy/paste straightforward
- Batch style application simple
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field, asdict


@dataclass
class PlotStyle:
    """Style properties for plots."""

    # Color and transparency
    color: Optional[str] = None
    alpha: Optional[float] = None

    # Line properties
    linewidth: Optional[float] = None
    linestyle: Optional[str] = None

    # Marker properties
    marker: Optional[str] = None
    markersize: Optional[float] = None

    # Fill properties (for fill_between, etc.)
    fill_color: Optional[str] = None
    fill_alpha: Optional[float] = None

    # Heatmap/image properties
    cmap: Optional[str] = None
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    interpolation: Optional[str] = None

    # Error bar properties
    capsize: Optional[float] = None
    xerr: Optional[float] = None
    yerr: Optional[float] = None

    # Bar properties
    width: Optional[float] = None

    # Histogram properties
    bins: Optional[int] = None
    density: bool = False
    cumulative: bool = False

    # Layering
    zorder: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlotStyle":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class AxesStyle:
    """Style properties for axes."""

    # Background
    facecolor: Optional[str] = None

    # Grid
    grid: bool = False
    grid_alpha: Optional[float] = 0.3
    grid_linestyle: Optional[str] = "--"
    grid_linewidth: Optional[float] = 0.5

    # Spines visibility
    spines_visible: Dict[str, bool] = field(
        default_factory=lambda: {
            "top": True,
            "right": True,
            "bottom": True,
            "left": True,
        }
    )

    # Font sizes
    label_fontsize: Optional[float] = None
    title_fontsize: Optional[float] = None
    tick_fontsize: Optional[float] = None

    # Legend
    legend: bool = False
    legend_loc: str = "best"
    legend_fontsize: Optional[float] = None
    legend_frameon: bool = True

    # Aspect
    aspect: Optional[str] = None  # "auto", "equal", or numeric

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AxesStyle":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class GuideStyle:
    """Style properties for guides (reference lines/spans)."""

    color: Optional[str] = None
    alpha: Optional[float] = None
    linestyle: Optional[str] = "--"
    linewidth: Optional[float] = None
    zorder: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GuideStyle":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class TextStyle:
    """Style properties for text/annotations."""

    fontsize: Optional[float] = None
    fontweight: Optional[str] = None  # "normal", "bold"
    color: Optional[str] = None
    ha: str = "center"  # Horizontal alignment
    va: str = "center"  # Vertical alignment
    rotation: Optional[float] = None
    zorder: Optional[int] = None

    # Bounding box (for text background)
    bbox: Optional[Dict[str, Any]] = None

    # Arrow properties (for annotate)
    arrowprops: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextStyle":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


# ============================================================================
# Style Copy/Paste Helpers
# ============================================================================


def copy_plot_style(src_plot, dst_plot) -> None:
    """
    Copy style from one plot to another.

    Parameters
    ----------
    src_plot : PlotModel
        Source plot
    dst_plot : PlotModel
        Destination plot
    """
    dst_plot.style = PlotStyle.from_dict(src_plot.style.to_dict())


def copy_axes_style(src_axes, dst_axes) -> None:
    """
    Copy style from one axes to another.

    Parameters
    ----------
    src_axes : AxesModel
        Source axes
    dst_axes : AxesModel
        Destination axes
    """
    dst_axes.style = AxesStyle.from_dict(src_axes.style.to_dict())


def copy_guide_style(src_guide, dst_guide) -> None:
    """
    Copy style from one guide to another.

    Parameters
    ----------
    src_guide : GuideModel
        Source guide
    dst_guide : GuideModel
        Destination guide
    """
    dst_guide.style = GuideStyle.from_dict(src_guide.style.to_dict())


def copy_text_style(src_annotation, dst_annotation) -> None:
    """
    Copy style from one annotation to another.

    Parameters
    ----------
    src_annotation : AnnotationModel
        Source annotation
    dst_annotation : AnnotationModel
        Destination annotation
    """
    dst_annotation.style = TextStyle.from_dict(src_annotation.style.to_dict())


def apply_style_to_plots(style: PlotStyle, plots: list) -> None:
    """
    Apply the same style to multiple plots.

    Parameters
    ----------
    style : PlotStyle
        Style to apply
    plots : list
        List of PlotModel objects

    Examples
    --------
    >>> blue_style = PlotStyle(color="blue", linewidth=2, alpha=0.8)
    >>> apply_style_to_plots(blue_style, my_plots)
    """
    style_dict = style.to_dict()
    for plot in plots:
        plot.style = PlotStyle.from_dict(style_dict)


# EOF
