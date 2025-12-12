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
        for ax, name in zip(axes.flatten(), ["histogram", "scatter", "bar_simple", "line"]):
            PLOTTERS[name](plt, rng, ax=ax)

        # 3D plots
        from scitex.dev.plt import PLOTTERS_3D
        fig, ax = PLOTTERS_3D["3d_surface"](plt, rng)

        # Animations
        from scitex.dev.plt import ANIMATIONS
        fig, anim = ANIMATIONS["line"](plt, rng)
        anim.save("wave.gif", writer="pillow")
"""

# Basic plotters (single-axis, accept ax=None)
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
from .plot_swarm import plot_swarm
from .plot_raster import plot_raster
from .plot_joyplot import plot_joyplot
from .plot_kde2d import plot_kde2d
from .plot_heatmap_annotated import plot_heatmap_annotated
from .plot_parallel import plot_parallel
from .plot_roc import plot_roc, plot_precision_recall

# Composite plotters (create their own multi-panel layout)
from .plot_gallery import plot_gallery, plot_gallery_quick
from .plot_clustermap import plot_clustermap
from .plot_pairplot import plot_pairplot
from .plot_jointplot import plot_jointplot

# 3D plotters
from .plot_3d import (
    plot_3d_line,
    plot_3d_scatter,
    plot_3d_surface,
    plot_3d_wireframe,
    plot_3d_bar,
    plot_3d_contour,
    plot_3d_gallery,
)

# Animation creators
from .plot_animation import (
    create_line_animation,
    create_scatter_animation,
    create_bar_animation,
    create_heatmap_animation,
    create_3d_rotation_animation,
    ANIMATIONS,
    list_animations,
    get_animation,
)

# =============================================================================
# Registry of single-panel plotters (accept ax=None parameter)
# =============================================================================
PLOTTERS = {
    # Bar plots
    "bar_grouped": plot_bar_grouped,
    "bar_simple": plot_bar_simple,
    "bar_stacked": plot_bar_stacked,
    # Statistical plots
    "boxplot": plot_boxplot,
    "violin": plot_violin,
    "swarm": plot_swarm,
    "errorbar": plot_errorbar,
    # Line plots
    "line": plot_multi_line,
    "step": plot_step,
    "stem": plot_stem,
    "fill_between": plot_fill_between,
    # Scatter plots
    "scatter": plot_scatter_sizes,
    # Histograms and density
    "histogram": plot_histogram,
    "histogram_multiple": plot_histogram_multiple,
    "joyplot": plot_joyplot,
    "kde2d": plot_kde2d,
    # Heatmaps
    "heatmap": plot_heatmap,
    "heatmap_annotated": plot_heatmap_annotated,
    "contour": plot_contour,
    # Other
    "pie": plot_pie,
    "raster": plot_raster,
    "parallel": plot_parallel,
    "complex_annotations": plot_complex_annotations,
    # Machine learning / classification
    "roc": plot_roc,
    "precision_recall": plot_precision_recall,
}

# =============================================================================
# Composite plotters (create their own multi-panel layout)
# =============================================================================
COMPOSITE_PLOTTERS = {
    "multi_panel": plot_multi_panel,
    "step_stem": plot_step_stem,  # Deprecated: use plot_step and plot_stem separately
    "gallery": plot_gallery,  # 8x6 comprehensive gallery
    "gallery_quick": plot_gallery_quick,  # 4x4 quick gallery
    "clustermap": plot_clustermap,
    "pairplot": plot_pairplot,
    "jointplot": plot_jointplot,
}

# =============================================================================
# 3D plotters (require projection="3d")
# =============================================================================
PLOTTERS_3D = {
    "3d_line": plot_3d_line,
    "3d_scatter": plot_3d_scatter,
    "3d_surface": plot_3d_surface,
    "3d_wireframe": plot_3d_wireframe,
    "3d_bar": plot_3d_bar,
    "3d_contour": plot_3d_contour,
    "3d_gallery": plot_3d_gallery,
}

# =============================================================================
# All plotters combined
# =============================================================================
ALL_PLOTTERS = {**PLOTTERS, **COMPOSITE_PLOTTERS, **PLOTTERS_3D}


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
    # Basic plotters
    "plot_bar_grouped",
    "plot_bar_simple",
    "plot_bar_stacked",
    "plot_boxplot",
    "plot_complex_annotations",
    "plot_contour",
    "plot_errorbar",
    "plot_fill_between",
    "plot_heatmap",
    "plot_heatmap_annotated",
    "plot_histogram",
    "plot_histogram_multiple",
    "plot_joyplot",
    "plot_kde2d",
    "plot_multi_line",
    "plot_multi_panel",
    "plot_parallel",
    "plot_pie",
    "plot_raster",
    "plot_scatter_sizes",
    "plot_step",
    "plot_stem",
    "plot_step_stem",
    "plot_swarm",
    "plot_violin",
    "plot_roc",
    "plot_precision_recall",
    # Composite plotters
    "plot_gallery",
    "plot_gallery_quick",
    "plot_clustermap",
    "plot_pairplot",
    "plot_jointplot",
    # 3D plotters
    "plot_3d_line",
    "plot_3d_scatter",
    "plot_3d_surface",
    "plot_3d_wireframe",
    "plot_3d_bar",
    "plot_3d_contour",
    "plot_3d_gallery",
    # Animation creators
    "create_line_animation",
    "create_scatter_animation",
    "create_bar_animation",
    "create_heatmap_animation",
    "create_3d_rotation_animation",
    # Registries
    "PLOTTERS",
    "COMPOSITE_PLOTTERS",
    "PLOTTERS_3D",
    "ALL_PLOTTERS",
    "ANIMATIONS",
    # Helper functions
    "get_plotter",
    "list_plotters",
    "get_animation",
    "list_animations",
]

# EOF
