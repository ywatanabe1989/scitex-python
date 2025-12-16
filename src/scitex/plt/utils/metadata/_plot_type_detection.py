#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_plot_type_detection.py

"""
Plot type detection utilities.

This module provides functions to detect the primary plot type from axes content.
"""


def _detect_plot_type(ax) -> tuple:
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
    # History format: dict with keys as IDs and values as tuples (id, method, tracked_dict, kwargs)
    if hasattr(ax, "history") and len(ax.history) > 0:
        # Get all methods from history
        methods = []
        for record in ax.history.values():
            if isinstance(record, tuple) and len(record) >= 2:
                methods.append(record[1])  # record[1] is the method name

        # Check methods in priority order (more specific first)
        for method in methods:
            if method == "stx_heatmap":
                return "heatmap", "stx_heatmap"
            elif method == "stx_kde":
                return "kde", "stx_kde"
            elif method == "stx_ecdf":
                return "ecdf", "stx_ecdf"
            elif method == "stx_violin":
                return "violin", "stx_violin"
            elif method in ("stx_box", "boxplot"):
                return "boxplot", method
            elif method == "stx_line":
                return "line", "stx_line"
            elif method == "plot_scatter":
                return "scatter", "plot_scatter"
            elif method == "stx_mean_std":
                return "line", "stx_mean_std"
            elif method == "stx_mean_ci":
                return "line", "stx_mean_ci"
            elif method == "stx_median_iqr":
                return "line", "stx_median_iqr"
            elif method == "stx_shaded_line":
                return "line", "stx_shaded_line"
            elif method == "sns_boxplot":
                return "boxplot", "sns_boxplot"
            elif method == "sns_violinplot":
                return "violin", "sns_violinplot"
            elif method == "sns_scatterplot":
                return "scatter", "sns_scatterplot"
            elif method == "sns_lineplot":
                return "line", "sns_lineplot"
            elif method == "sns_histplot":
                return "hist", "sns_histplot"
            elif method == "sns_barplot":
                return "bar", "sns_barplot"
            elif method == "sns_stripplot":
                return "scatter", "sns_stripplot"
            elif method == "sns_kdeplot":
                return "kde", "sns_kdeplot"
            elif method == "scatter":
                return "scatter", "scatter"
            elif method == "bar":
                return "bar", "bar"
            elif method == "barh":
                return "bar", "barh"
            elif method == "hist":
                return "hist", "hist"
            elif method == "hist2d":
                return "hist2d", "hist2d"
            elif method == "hexbin":
                return "hexbin", "hexbin"
            elif method == "violinplot":
                return "violin", "violinplot"
            elif method == "errorbar":
                return "errorbar", "errorbar"
            elif method == "fill_between":
                return "fill", "fill_between"
            elif method == "fill_betweenx":
                return "fill", "fill_betweenx"
            elif method == "imshow":
                return "image", "imshow"
            elif method == "matshow":
                return "image", "matshow"
            elif method == "contour":
                return "contour", "contour"
            elif method == "contourf":
                return "contour", "contourf"
            elif method == "stem":
                return "stem", "stem"
            elif method == "step":
                return "step", "step"
            elif method == "pie":
                return "pie", "pie"
            elif method == "quiver":
                return "quiver", "quiver"
            elif method == "streamplot":
                return "stream", "streamplot"
            elif method == "plot":
                return "line", "plot"
            # Note: "plot" method is handled last as a fallback since boxplot uses it internally

    # Check for images (takes priority)
    if len(ax.images) > 0:
        return "image", "imshow"

    # Check for 2D density plots (hist2d, hexbin) - QuadMesh or PolyCollection
    if hasattr(ax, "collections"):
        for coll in ax.collections:
            coll_type = type(coll).__name__
            if "QuadMesh" in coll_type:
                return "hist2d", "hist2d"
            if "PolyCollection" in coll_type and hasattr(coll, "get_array"):
                # hexbin creates PolyCollection with array data
                arr = coll.get_array()
                if arr is not None and len(arr) > 0:
                    return "hexbin", "hexbin"

    # Check for contours
    if hasattr(ax, "collections"):
        for coll in ax.collections:
            if "Contour" in type(coll).__name__:
                return "contour", "contour"

    # Check for bar plots
    if len(ax.containers) > 0:
        # Check if it's a boxplot (has multiple containers with specific structure)
        if any("boxplot" in str(type(c)).lower() for c in ax.containers):
            return "boxplot", "boxplot"
        # Otherwise assume bar plot
        return "bar", "bar"

    # Check for patches (could be histogram, violin, pie, etc.)
    if len(ax.patches) > 0:
        # Check for pie chart (Wedge patches)
        if any("Wedge" in type(p).__name__ for p in ax.patches):
            return "pie", "pie"
        # If there are many rectangular patches, likely histogram
        if len(ax.patches) > 5:
            return "hist", "hist"
        # Check for violin plot
        if any("Poly" in type(p).__name__ for p in ax.patches):
            return "violin", "violinplot"

    # Check for scatter plots (PathCollection)
    if hasattr(ax, "collections") and len(ax.collections) > 0:
        for coll in ax.collections:
            if "PathCollection" in type(coll).__name__:
                return "scatter", "scatter"

    # Check for line plots
    if len(ax.lines) > 0:
        # If there are error bars, it might be errorbar plot
        if any(hasattr(line, "_mpl_error") for line in ax.lines):
            return "errorbar", "errorbar"
        return "line", "plot"

    return None, None
