#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/__init__.py

"""
Development plotting utilities with three API layers.

All plotters follow signature: (plt, rng, ax=None) -> (fig, ax)

API Layers:
-----------
stx_*  : SciTeX canonical (ArrayLike input, tracked)
sns_*  : Seaborn-style (DataFrame input, tracked)
mpl_*  : Matplotlib-style (raw args, tracked)

Usage:
    from scitex.dev.plt import PLOTTERS_STX, PLOTTERS_SNS, PLOTTERS_MPL

    @stx.session
    def main(plt=stx.INJECTED, rng_manager=stx.INJECTED):
        rng = rng_manager("demo")

        # stx_* API (ArrayLike)
        for name, plotter in PLOTTERS_STX.items():
            fig, ax = plotter(plt, rng)

        # sns_* API (DataFrame)
        for name, plotter in PLOTTERS_SNS.items():
            fig, ax = plotter(plt, rng)

        # mpl_* API (matplotlib-style)
        for name, plotter in PLOTTERS_MPL.items():
            fig, ax = plotter(plt, rng)
"""

# =============================================================================
# stx_* API Layer - ArrayLike input (25 plotters)
# =============================================================================
from .plot_stx_line import plot_stx_line
from .plot_stx_mean_std import plot_stx_mean_std
from .plot_stx_mean_ci import plot_stx_mean_ci
from .plot_stx_median_iqr import plot_stx_median_iqr
from .plot_stx_shaded_line import plot_stx_shaded_line
from .plot_stx_box import plot_stx_box
from .plot_stx_violin import plot_stx_violin
from .plot_stx_scatter import plot_stx_scatter
from .plot_stx_bar import plot_stx_bar
from .plot_stx_barh import plot_stx_barh
from .plot_stx_errorbar import plot_stx_errorbar
from .plot_stx_fill_between import plot_stx_fill_between
from .plot_stx_kde import plot_stx_kde
from .plot_stx_ecdf import plot_stx_ecdf
from .plot_stx_heatmap import plot_stx_heatmap
from .plot_stx_image import plot_stx_image
from .plot_stx_imshow import plot_stx_imshow
from .plot_stx_contour import plot_stx_contour
from .plot_stx_raster import plot_stx_raster
from .plot_stx_conf_mat import plot_stx_conf_mat
from .plot_stx_joyplot import plot_stx_joyplot
from .plot_stx_rectangle import plot_stx_rectangle
from .plot_stx_fillv import plot_stx_fillv
from .plot_stx_boxplot import plot_stx_boxplot
from .plot_stx_violinplot import plot_stx_violinplot

PLOTTERS_STX = {
    "stx_line": plot_stx_line,
    "stx_mean_std": plot_stx_mean_std,
    "stx_mean_ci": plot_stx_mean_ci,
    "stx_median_iqr": plot_stx_median_iqr,
    "stx_shaded_line": plot_stx_shaded_line,
    "stx_box": plot_stx_box,
    "stx_violin": plot_stx_violin,
    "stx_scatter": plot_stx_scatter,
    "stx_bar": plot_stx_bar,
    "stx_barh": plot_stx_barh,
    "stx_errorbar": plot_stx_errorbar,
    "stx_fill_between": plot_stx_fill_between,
    "stx_kde": plot_stx_kde,
    "stx_ecdf": plot_stx_ecdf,
    "stx_heatmap": plot_stx_heatmap,
    "stx_image": plot_stx_image,
    "stx_imshow": plot_stx_imshow,
    "stx_contour": plot_stx_contour,
    "stx_raster": plot_stx_raster,
    "stx_conf_mat": plot_stx_conf_mat,
    "stx_joyplot": plot_stx_joyplot,
    "stx_rectangle": plot_stx_rectangle,
    "stx_fillv": plot_stx_fillv,
    "stx_boxplot": plot_stx_boxplot,
    "stx_violinplot": plot_stx_violinplot,
}

# =============================================================================
# sns_* API Layer - DataFrame input (10 plotters)
# =============================================================================
from .plot_sns_boxplot import plot_sns_boxplot
from .plot_sns_violinplot import plot_sns_violinplot
from .plot_sns_barplot import plot_sns_barplot
from .plot_sns_histplot import plot_sns_histplot
from .plot_sns_kdeplot import plot_sns_kdeplot
from .plot_sns_scatterplot import plot_sns_scatterplot
from .plot_sns_lineplot import plot_sns_lineplot
from .plot_sns_swarmplot import plot_sns_swarmplot
from .plot_sns_stripplot import plot_sns_stripplot
from .plot_sns_heatmap import plot_sns_heatmap

PLOTTERS_SNS = {
    "sns_boxplot": plot_sns_boxplot,
    "sns_violinplot": plot_sns_violinplot,
    "sns_barplot": plot_sns_barplot,
    "sns_histplot": plot_sns_histplot,
    "sns_kdeplot": plot_sns_kdeplot,
    "sns_scatterplot": plot_sns_scatterplot,
    "sns_lineplot": plot_sns_lineplot,
    "sns_swarmplot": plot_sns_swarmplot,
    "sns_stripplot": plot_sns_stripplot,
    "sns_heatmap": plot_sns_heatmap,
}

# =============================================================================
# mpl_* API Layer - Matplotlib-style input (26 plotters)
# =============================================================================
from .plot_mpl_plot import plot_mpl_plot
from .plot_mpl_scatter import plot_mpl_scatter
from .plot_mpl_bar import plot_mpl_bar
from .plot_mpl_barh import plot_mpl_barh
from .plot_mpl_hist import plot_mpl_hist
from .plot_mpl_hist2d import plot_mpl_hist2d
from .plot_mpl_hexbin import plot_mpl_hexbin
from .plot_mpl_boxplot import plot_mpl_boxplot
from .plot_mpl_violinplot import plot_mpl_violinplot
from .plot_mpl_errorbar import plot_mpl_errorbar
from .plot_mpl_step import plot_mpl_step
from .plot_mpl_stem import plot_mpl_stem
from .plot_mpl_fill import plot_mpl_fill
from .plot_mpl_fill_between import plot_mpl_fill_between
from .plot_mpl_stackplot import plot_mpl_stackplot
from .plot_mpl_contour import plot_mpl_contour
from .plot_mpl_contourf import plot_mpl_contourf
from .plot_mpl_imshow import plot_mpl_imshow
from .plot_mpl_pcolormesh import plot_mpl_pcolormesh
from .plot_mpl_pie import plot_mpl_pie
from .plot_mpl_eventplot import plot_mpl_eventplot
from .plot_mpl_quiver import plot_mpl_quiver
from .plot_mpl_axhline import plot_mpl_axhline
from .plot_mpl_axvline import plot_mpl_axvline
from .plot_mpl_axhspan import plot_mpl_axhspan
from .plot_mpl_axvspan import plot_mpl_axvspan

PLOTTERS_MPL = {
    "mpl_plot": plot_mpl_plot,
    "mpl_scatter": plot_mpl_scatter,
    "mpl_bar": plot_mpl_bar,
    "mpl_barh": plot_mpl_barh,
    "mpl_hist": plot_mpl_hist,
    "mpl_hist2d": plot_mpl_hist2d,
    "mpl_hexbin": plot_mpl_hexbin,
    "mpl_boxplot": plot_mpl_boxplot,
    "mpl_violinplot": plot_mpl_violinplot,
    "mpl_errorbar": plot_mpl_errorbar,
    "mpl_step": plot_mpl_step,
    "mpl_stem": plot_mpl_stem,
    "mpl_fill": plot_mpl_fill,
    "mpl_fill_between": plot_mpl_fill_between,
    "mpl_stackplot": plot_mpl_stackplot,
    "mpl_contour": plot_mpl_contour,
    "mpl_contourf": plot_mpl_contourf,
    "mpl_imshow": plot_mpl_imshow,
    "mpl_pcolormesh": plot_mpl_pcolormesh,
    "mpl_pie": plot_mpl_pie,
    "mpl_eventplot": plot_mpl_eventplot,
    "mpl_quiver": plot_mpl_quiver,
    "mpl_axhline": plot_mpl_axhline,
    "mpl_axvline": plot_mpl_axvline,
    "mpl_axhspan": plot_mpl_axhspan,
    "mpl_axvspan": plot_mpl_axvspan,
}

# =============================================================================
# Combined registry
# =============================================================================
PLOTTERS = {**PLOTTERS_STX, **PLOTTERS_SNS, **PLOTTERS_MPL}


def get_plotter(name):
    """Get a plotter function by name."""
    if name in PLOTTERS:
        return PLOTTERS[name]
    raise KeyError(f"Unknown plotter: {name}. Available: {list(PLOTTERS.keys())}")


def list_plotters():
    """List all available plotter names."""
    return list(PLOTTERS.keys())


__all__ = [
    # stx_* plotters (25)
    "plot_stx_line",
    "plot_stx_mean_std",
    "plot_stx_mean_ci",
    "plot_stx_median_iqr",
    "plot_stx_shaded_line",
    "plot_stx_box",
    "plot_stx_violin",
    "plot_stx_scatter",
    "plot_stx_bar",
    "plot_stx_barh",
    "plot_stx_errorbar",
    "plot_stx_fill_between",
    "plot_stx_kde",
    "plot_stx_ecdf",
    "plot_stx_heatmap",
    "plot_stx_image",
    "plot_stx_imshow",
    "plot_stx_contour",
    "plot_stx_raster",
    "plot_stx_conf_mat",
    "plot_stx_joyplot",
    "plot_stx_rectangle",
    "plot_stx_fillv",
    "plot_stx_boxplot",
    "plot_stx_violinplot",
    # sns_* plotters (10)
    "plot_sns_boxplot",
    "plot_sns_violinplot",
    "plot_sns_barplot",
    "plot_sns_histplot",
    "plot_sns_kdeplot",
    "plot_sns_scatterplot",
    "plot_sns_lineplot",
    "plot_sns_swarmplot",
    "plot_sns_stripplot",
    "plot_sns_heatmap",
    # mpl_* plotters (26)
    "plot_mpl_plot",
    "plot_mpl_scatter",
    "plot_mpl_bar",
    "plot_mpl_barh",
    "plot_mpl_hist",
    "plot_mpl_hist2d",
    "plot_mpl_hexbin",
    "plot_mpl_boxplot",
    "plot_mpl_violinplot",
    "plot_mpl_errorbar",
    "plot_mpl_step",
    "plot_mpl_stem",
    "plot_mpl_fill",
    "plot_mpl_fill_between",
    "plot_mpl_stackplot",
    "plot_mpl_contour",
    "plot_mpl_contourf",
    "plot_mpl_imshow",
    "plot_mpl_pcolormesh",
    "plot_mpl_pie",
    "plot_mpl_eventplot",
    "plot_mpl_quiver",
    "plot_mpl_axhline",
    "plot_mpl_axvline",
    "plot_mpl_axhspan",
    "plot_mpl_axvspan",
    # Registries
    "PLOTTERS_STX",
    "PLOTTERS_SNS",
    "PLOTTERS_MPL",
    "PLOTTERS",
    # Helper functions
    "get_plotter",
    "list_plotters",
]

# EOF
