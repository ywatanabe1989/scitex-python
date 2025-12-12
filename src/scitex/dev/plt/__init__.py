#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/__init__.py

"""
Development plotting utilities.

Reusable plotting functions for testing, demos, and development.
Each function takes (plt, rng, ax=None) and returns (fig, ax).

Usage:
    from scitex.dev.plt import plot_histogram, plot_scatter_sizes, PLOTTERS

    @stx.session
    def main(plt=stx.INJECTED, rng_manager=stx.INJECTED):
        rng = rng_manager("demo")

        # Direct function call
        fig, ax = plot_histogram(plt, rng)

        # Registry access
        for name, plotter in PLOTTERS.items():
            fig, ax = plotter(plt, rng)

        # Multi-panel usage
        fig, axes = plt.subplots(2, 2)
        for ax, name in zip(axes.flatten(), ["histogram", "scatter", "bar", "line"]):
            PLOTTERS[name](plt, rng, ax=ax)
"""

from .plot_bar_grouped import plot_bar_grouped
from .plot_bar_simple import plot_bar_simple
from .plot_bar_stacked import plot_bar_stacked
from .plot_boxplot import plot_boxplot
from .plot_complex_annotations import plot_complex_annotations
from .plot_contour import plot_contour
from .plot_errorbar import plot_errorbar
from .plot_fill_between import plot_fill_between
from .plot_heatmap import plot_heatmap
from .plot_histogram import plot_histogram
from .plot_histogram_multiple import plot_histogram_multiple
from .plot_multi_line import plot_multi_line
from .plot_multi_panel import plot_multi_panel
from .plot_pie import plot_pie
from .plot_scatter_sizes import plot_scatter_sizes
from .plot_step import plot_step
from .plot_stem import plot_stem
from .plot_step_stem import plot_step_stem  # Composite (deprecated)
from .plot_violin import plot_violin

# Registry of single-panel plotters (accept ax=None parameter)
PLOTTERS = {
    "bar_grouped": plot_bar_grouped,
    "bar_simple": plot_bar_simple,
    "bar_stacked": plot_bar_stacked,
    "boxplot": plot_boxplot,
    "complex_annotations": plot_complex_annotations,
    "contour": plot_contour,
    "errorbar": plot_errorbar,
    "fill_between": plot_fill_between,
    "heatmap": plot_heatmap,
    "histogram": plot_histogram,
    "histogram_multiple": plot_histogram_multiple,
    "line": plot_multi_line,
    "pie": plot_pie,
    "scatter": plot_scatter_sizes,
    "step": plot_step,
    "stem": plot_stem,
    "violin": plot_violin,
}

# Composite plotters (create their own multi-panel layout)
COMPOSITE_PLOTTERS = {
    "multi_panel": plot_multi_panel,
    "step_stem": plot_step_stem,  # Deprecated: use plot_step and plot_stem separately
}

# All plotters combined
ALL_PLOTTERS = {**PLOTTERS, **COMPOSITE_PLOTTERS}


def get_plotter(name):
    """Get a plotter function by name.

    Parameters
    ----------
    name : str
        Plotter name (e.g., "histogram", "scatter", "bar_simple")

    Returns
    -------
    callable
        Plotter function with signature (plt, rng, ax=None) -> (fig, ax)

    Raises
    ------
    KeyError
        If plotter name not found
    """
    if name in ALL_PLOTTERS:
        return ALL_PLOTTERS[name]
    raise KeyError(f"Unknown plotter: {name}. Available: {list(ALL_PLOTTERS.keys())}")


def list_plotters():
    """List all available plotter names.

    Returns
    -------
    list[str]
        List of plotter names
    """
    return list(ALL_PLOTTERS.keys())


__all__ = [
    # Functions
    "plot_bar_grouped",
    "plot_bar_simple",
    "plot_bar_stacked",
    "plot_boxplot",
    "plot_complex_annotations",
    "plot_contour",
    "plot_errorbar",
    "plot_fill_between",
    "plot_heatmap",
    "plot_histogram",
    "plot_histogram_multiple",
    "plot_multi_line",
    "plot_multi_panel",
    "plot_pie",
    "plot_scatter_sizes",
    "plot_step",
    "plot_stem",
    "plot_step_stem",
    "plot_violin",
    # Registry
    "PLOTTERS",
    "COMPOSITE_PLOTTERS",
    "ALL_PLOTTERS",
    "get_plotter",
    "list_plotters",
]

# EOF
