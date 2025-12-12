#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/dev/plt/__init__.py

"""
Development plotting utilities.

Reusable plotting functions for testing, demos, and development.
Each function takes (plt, rng) and returns (fig, ax).

Usage:
    from scitex.dev.plt import plot_histogram, plot_scatter_sizes

    @stx.session
    def main(plt=stx.INJECTED, rng_manager=stx.INJECTED):
        rng = rng_manager("demo")
        fig, ax = plot_histogram(plt, rng)
        ...
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
from .plot_step_stem import plot_step_stem
from .plot_violin import plot_violin

__all__ = [
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
    "plot_step_stem",
    "plot_violin",
]

# EOF
