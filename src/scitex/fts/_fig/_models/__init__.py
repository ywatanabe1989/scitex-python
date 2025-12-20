#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_models/__init__.py

"""JSON models for FTS figure specifications."""

from ._Annotations import AnnotationModel
from ._Axes import AxesModel
from ._Figure import FigureModel
from ._Guides import GuideModel
from ._Plot import PlotModel
from ._plot_types import (
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
from ._Styles import (
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
]

# EOF
