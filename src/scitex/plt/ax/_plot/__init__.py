#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 20:12:19 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/plt/ax/_plot/__init__.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/ax/_plot/__init__.py"
__DIR__ = os.path.dirname(__FILE__)

from ._stx_scatter_hist import stx_scatter_hist
from ._stx_heatmap import stx_heatmap
from ._plot_circular_hist import plot_circular_hist
from ._stx_conf_mat import stx_conf_mat
from ._plot_cube import plot_cube
from ._stx_ecdf import stx_ecdf
from ._stx_fillv import stx_fillv
from ._stx_violin import stx_violin, sns_plot_violin
from ._stx_image import stx_image
from ._stx_joyplot import stx_joyplot
from ._stx_raster import stx_raster
from ._stx_rectangle import stx_rectangle
from ._stx_shaded_line import stx_shaded_line, _plot_single_shaded_line
from ._plot_statistical_shaded_line import (
    stx_line,
    stx_mean_std,
    stx_mean_ci,
    stx_median_iqr,
)
from ._add_fitted_line import add_fitted_line

from scitex.decorators import deprecated

# Backward-compatible aliases for renamed functions with deprecation warnings
@deprecated(reason="Use stx_line instead", forward_to="scitex.plt.ax._plot.stx_line")
def plot_line(*args, **kwargs):
    pass

@deprecated(reason="Use stx_mean_std instead", forward_to="scitex.plt.ax._plot.stx_mean_std")
def plot_mean_std(*args, **kwargs):
    pass

@deprecated(reason="Use stx_mean_ci instead", forward_to="scitex.plt.ax._plot.stx_mean_ci")
def plot_mean_ci(*args, **kwargs):
    pass

@deprecated(reason="Use stx_median_iqr instead", forward_to="scitex.plt.ax._plot.stx_median_iqr")
def plot_median_iqr(*args, **kwargs):
    pass

__all__ = [
    "stx_scatter_hist",
    "stx_heatmap",
    "plot_circular_hist",
    "stx_conf_mat",
    "plot_cube",
    "stx_ecdf",
    "stx_fillv",
    "stx_violin",
    "sns_plot_violin",
    "stx_image",
    "stx_joyplot",
    "stx_raster",
    "stx_rectangle",
    "stx_shaded_line",
    "_plot_single_shaded_line",
    "stx_line",
    "stx_mean_std",
    "stx_mean_ci",
    "stx_median_iqr",
    "add_fitted_line",
    # Backward-compatible aliases
    "plot_line",
    "plot_mean_std",
    "plot_mean_ci",
    "plot_median_iqr",
]

# EOF
