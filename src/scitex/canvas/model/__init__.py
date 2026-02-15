#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/model/__init__.py
"""
JSON models for scitex.canvas figure specifications.

This module provides dataclass-based models for representing
publication-quality figures as JSON structures.
"""

from ._annotations import AnnotationModel
from ._axes import AxesModel
from ._figure import FigureModel
from ._guides import GuideModel
from ._plot import PlotModel
from ._plot_types import (  # Registry; Matplotlib basic; Custom scitex; Seaborn
    PLOT_TYPE_CONFIGS,
    BarHPlotConfig,
    BarPlotConfig,
    BoxConfig,
    BoxPlotConfig,
    ContourConfig,
    ECDFConfig,
    ErrorbarPlotConfig,
    FillBetweenConfig,
    HeatmapConfig,
    HistPlotConfig,
    ImshowConfig,
    KDEConfig,
    LinePlotConfig,
    MeanStdConfig,
    PlotLineConfig,
    ScatterPlotConfig,
    SeabornBarplotConfig,
    SeabornBoxplotConfig,
    SeabornHistplotConfig,
    SeabornKDEplotConfig,
    SeabornLineplotConfig,
    SeabornScatterplotConfig,
    SeabornStripplotConfig,
    SeabornViolinplotConfig,
    ShadedLineConfig,
    ViolinConfig,
    ViolinPlotConfig,
    get_plot_config_class,
    list_plot_types,
)
from ._styles import (
    AxesStyle,
    GuideStyle,
    PlotStyle,
    TextStyle,
    apply_style_to_plots,
    copy_axes_style,
    copy_guide_style,
    copy_plot_style,
    copy_text_style,
)

__all__ = [
    # Core models
    "FigureModel",
    "AxesModel",
    "PlotModel",
    "GuideModel",
    "AnnotationModel",
    # Style models
    "PlotStyle",
    "AxesStyle",
    "GuideStyle",
    "TextStyle",
    # Style helpers
    "copy_plot_style",
    "copy_axes_style",
    "copy_guide_style",
    "copy_text_style",
    "apply_style_to_plots",
    # Registry
    "PLOT_TYPE_CONFIGS",
    "get_plot_config_class",
    "list_plot_types",
    # Matplotlib basic
    "LinePlotConfig",
    "ScatterPlotConfig",
    "BarPlotConfig",
    "BarHPlotConfig",
    "HistPlotConfig",
    "BoxPlotConfig",
    "ErrorbarPlotConfig",
    "FillBetweenConfig",
    "ImshowConfig",
    "ContourConfig",
    "ViolinPlotConfig",
    # Custom scitex
    "HeatmapConfig",
    "PlotLineConfig",
    "ShadedLineConfig",
    "ViolinConfig",
    "ECDFConfig",
    "BoxConfig",
    "MeanStdConfig",
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
