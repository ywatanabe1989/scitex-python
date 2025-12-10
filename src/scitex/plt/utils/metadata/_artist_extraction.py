#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scitex/plt/utils/metadata/_artist_extraction.py

"""
Artist extraction orchestration.

This module provides the main _extract_artists function that coordinates extraction
of all artist types from matplotlib axes.
"""

from ._plot_type_detection import _detect_plot_type
from ._line_artists import _extract_line_artists, _extract_line_collection_artists
from ._collection_artists import (
    _extract_scatter_artists,
    _extract_hist2d_hexbin_artists,
    _extract_violin_body_artists,
)
from ._patch_artists import _extract_rectangle_artists, _extract_wedge_artists
from ._image_text_artists import _extract_image_artists, _extract_text_artists


def _extract_artists(ax) -> list:
    """
    Extract artist information including properties and CSV column mapping.

    Uses matplotlib terminology: each drawable element is an Artist.
    Only includes artists that were explicitly created via scitex tracking (top-level calls),
    not internal artists created by matplotlib functions like boxplot() which internally
    call plot() multiple times.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to extract artists from

    Returns
    -------
    list
        List of artist dictionaries with:
        - id: unique identifier
        - artist_class: matplotlib class name (Line2D, PathCollection, etc.)
        - label: legend label
        - style: color, linestyle, linewidth, etc.
        - data_ref: CSV column mapping (matches columns_actual exactly)
    """
    artists = []

    # Get axes position for CSV column naming
    ax_row, ax_col = 0, 0
    if hasattr(ax, "_scitex_metadata") and "position_in_grid" in ax._scitex_metadata:
        pos = ax._scitex_metadata["position_in_grid"]
        ax_row, ax_col = pos[0], pos[1]

    # Get the raw matplotlib axes for accessing artists
    mpl_ax = ax._axis_mpl if hasattr(ax, "_axis_mpl") else ax

    # Try to find scitex wrapper for plot type detection and history access
    ax_for_detection = ax
    if not hasattr(ax, 'history') and hasattr(mpl_ax, '_scitex_wrapper'):
        ax_for_detection = mpl_ax._scitex_wrapper

    # Detect plot type
    plot_type, method = _detect_plot_type(ax_for_detection)

    # Plot types where internal line artists should be hidden
    internal_plot_types = {
        "boxplot", "violin", "hist", "bar", "image", "heatmap", "kde", "ecdf",
        "errorbar", "fill", "stem", "contour", "pie", "quiver", "stream"
    }
    skip_unlabeled = plot_type in internal_plot_types

    # Extract Line2D artists
    line_artists = _extract_line_artists(
        mpl_ax, ax_for_detection, plot_type, method, ax_row, ax_col, skip_unlabeled
    )
    artists.extend(line_artists)

    # Extract PathCollection artists (scatter points)
    scatter_artists = _extract_scatter_artists(mpl_ax, ax_row, ax_col)
    artists.extend(scatter_artists)

    # Extract Rectangle patches (bar/barh/hist)
    rectangle_artists = _extract_rectangle_artists(
        mpl_ax, ax_for_detection, plot_type, ax_row, ax_col, skip_unlabeled
    )
    artists.extend(rectangle_artists)

    # Extract Wedge patches (pie charts)
    wedge_artists = _extract_wedge_artists(mpl_ax)
    artists.extend(wedge_artists)

    # Extract QuadMesh and PolyCollection (hist2d, hexbin)
    hist2d_hexbin_artists = _extract_hist2d_hexbin_artists(mpl_ax, ax_for_detection, plot_type)
    artists.extend(hist2d_hexbin_artists)

    # Extract violin body (PolyCollection)
    violin_body_artists = _extract_violin_body_artists(mpl_ax, plot_type)
    artists.extend(violin_body_artists)

    # Extract AxesImage (imshow)
    image_artists = _extract_image_artists(mpl_ax)
    artists.extend(image_artists)

    # Extract Text artists (annotations)
    text_artists = _extract_text_artists(mpl_ax)
    artists.extend(text_artists)

    # Extract LineCollection artists (errorbar lines, etc.)
    linecollection_artists = _extract_line_collection_artists(
        mpl_ax, ax_for_detection, plot_type, method, ax_row, ax_col
    )
    artists.extend(linecollection_artists)

    return artists


# Backward compatibility alias
_extract_traces = _extract_artists
