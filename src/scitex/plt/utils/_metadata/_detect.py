#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/plt/utils/_metadata/_detect.py

"""
Plot type detection from axes content.

Detects the primary plot type by analyzing scitex history and matplotlib
artist content.
"""

from typing import Optional, Tuple


def _detect_plot_type(ax) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect the primary plot type and method from axes content.

    Checks for:
    - Lines -> "line"
    - Scatter collections -> "scatter"
    - Bar containers -> "bar"
    - Patches (histogram) -> "hist"
    - Box plot -> "boxplot"
    - Violin plot -> "violin"
    - Image -> "image"
    - Contour -> "contour"
    - KDE -> "kde"

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to analyze

    Returns
    -------
    tuple
        (plot_type, method) where method is the actual plotting function used,
        or (None, None) if unclear
    """
    # Check scitex history FIRST (most reliable for scitex plots)
    result = _detect_from_history(ax)
    if result[0] is not None:
        return result

    # Check for images (takes priority)
    if len(ax.images) > 0:
        return "image", "imshow"

    # Check for 2D density plots (hist2d, hexbin)
    result = _detect_2d_density(ax)
    if result[0] is not None:
        return result

    # Check for contours
    result = _detect_contours(ax)
    if result[0] is not None:
        return result

    # Check for bar plots and containers
    result = _detect_bars(ax)
    if result[0] is not None:
        return result

    # Check for patches (histogram, violin, pie)
    result = _detect_patches(ax)
    if result[0] is not None:
        return result

    # Check for scatter plots (PathCollection)
    result = _detect_scatter(ax)
    if result[0] is not None:
        return result

    # Check for line plots
    result = _detect_lines(ax)
    if result[0] is not None:
        return result

    return None, None


def _detect_from_history(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect plot type from scitex history."""
    if not hasattr(ax, "history") or len(ax.history) == 0:
        return None, None

    # Method to (plot_type, method) mapping
    method_map = {
        "stx_heatmap": ("heatmap", "stx_heatmap"),
        "stx_kde": ("kde", "stx_kde"),
        "stx_ecdf": ("ecdf", "stx_ecdf"),
        "stx_violin": ("violin", "stx_violin"),
        "stx_box": ("boxplot", "stx_box"),
        "boxplot": ("boxplot", "boxplot"),
        "stx_line": ("line", "stx_line"),
        "plot_scatter": ("scatter", "plot_scatter"),
        "stx_mean_std": ("line", "stx_mean_std"),
        "stx_mean_ci": ("line", "stx_mean_ci"),
        "stx_median_iqr": ("line", "stx_median_iqr"),
        "stx_shaded_line": ("line", "stx_shaded_line"),
        "sns_boxplot": ("boxplot", "sns_boxplot"),
        "sns_violinplot": ("violin", "sns_violinplot"),
        "sns_scatterplot": ("scatter", "sns_scatterplot"),
        "sns_lineplot": ("line", "sns_lineplot"),
        "sns_histplot": ("hist", "sns_histplot"),
        "sns_barplot": ("bar", "sns_barplot"),
        "sns_stripplot": ("scatter", "sns_stripplot"),
        "sns_kdeplot": ("kde", "sns_kdeplot"),
        "scatter": ("scatter", "scatter"),
        "bar": ("bar", "bar"),
        "barh": ("bar", "barh"),
        "hist": ("hist", "hist"),
        "hist2d": ("hist2d", "hist2d"),
        "hexbin": ("hexbin", "hexbin"),
        "violinplot": ("violin", "violinplot"),
        "errorbar": ("errorbar", "errorbar"),
        "fill_between": ("fill", "fill_between"),
        "fill_betweenx": ("fill", "fill_betweenx"),
        "imshow": ("image", "imshow"),
        "matshow": ("image", "matshow"),
        "contour": ("contour", "contour"),
        "contourf": ("contour", "contourf"),
        "stem": ("stem", "stem"),
        "step": ("step", "step"),
        "pie": ("pie", "pie"),
        "quiver": ("quiver", "quiver"),
        "streamplot": ("stream", "streamplot"),
        "plot": ("line", "plot"),
    }

    # Get all methods from history
    for record in ax.history.values():
        if isinstance(record, tuple) and len(record) >= 2:
            method = record[1]
            if method in method_map:
                return method_map[method]

    return None, None


def _detect_2d_density(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect 2D density plots (hist2d, hexbin)."""
    if not hasattr(ax, "collections"):
        return None, None

    for coll in ax.collections:
        coll_type = type(coll).__name__
        if "QuadMesh" in coll_type:
            return "hist2d", "hist2d"
        if "PolyCollection" in coll_type and hasattr(coll, "get_array"):
            arr = coll.get_array()
            if arr is not None and len(arr) > 0:
                return "hexbin", "hexbin"

    return None, None


def _detect_contours(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect contour plots."""
    if not hasattr(ax, "collections"):
        return None, None

    for coll in ax.collections:
        if "Contour" in type(coll).__name__:
            return "contour", "contour"

    return None, None


def _detect_bars(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect bar and boxplots from containers."""
    if len(ax.containers) == 0:
        return None, None

    if any("boxplot" in str(type(c)).lower() for c in ax.containers):
        return "boxplot", "boxplot"

    return "bar", "bar"


def _detect_patches(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect histogram, violin, pie from patches."""
    if len(ax.patches) == 0:
        return None, None

    # Check for pie chart (Wedge patches)
    if any("Wedge" in type(p).__name__ for p in ax.patches):
        return "pie", "pie"

    # If there are many rectangular patches, likely histogram
    if len(ax.patches) > 5:
        return "hist", "hist"

    # Check for violin plot
    if any("Poly" in type(p).__name__ for p in ax.patches):
        return "violin", "violinplot"

    return None, None


def _detect_scatter(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect scatter plots from PathCollection."""
    if not hasattr(ax, "collections") or len(ax.collections) == 0:
        return None, None

    for coll in ax.collections:
        if "PathCollection" in type(coll).__name__:
            return "scatter", "scatter"

    return None, None


def _detect_lines(ax) -> Tuple[Optional[str], Optional[str]]:
    """Detect line and errorbar plots."""
    if len(ax.lines) == 0:
        return None, None

    if any(hasattr(line, "_mpl_error") for line in ax.lines):
        return "errorbar", "errorbar"

    return "line", "plot"


# EOF
