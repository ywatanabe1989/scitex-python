#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:00:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_scatter.py

"""
Style scatter plot elements with millimeter-based control.

Default values are loaded from SCITEX_STYLE.yaml via presets.py.
"""

from typing import Optional

from scitex.plt.styles.presets import SCITEX_STYLE

# Get defaults from centralized config
_DEFAULT_SIZE_MM = SCITEX_STYLE.get("scatter_size_mm", 0.8)
_DEFAULT_EDGE_THICKNESS_MM = SCITEX_STYLE.get("marker_edge_width_mm", 0.0)


def style_scatter(
    path_collection,
    size_mm: float = None,
    edge_thickness_mm: float = None,
):
    """
    Apply consistent styling to matplotlib scatter plot elements.

    Parameters
    ----------
    path_collection : PathCollection
        Collection returned by ax.scatter()
    size_mm : float, optional
        Marker size in millimeters (default: 0.8mm)
    edge_thickness_mm : float, optional
        Edge line thickness in millimeters (default: 0.0mm = no border)

    Returns
    -------
    path_collection : PathCollection
        The styled path collection

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.NATURE_STYLE)
    >>> scatter = ax.scatter(x, y)
    >>> stx.ax.style_scatter(scatter, size_mm=0.8)

    Notes
    -----
    Matplotlib scatter uses marker size in points squared.
    We convert mm to points, then square for the area.
    By default, no border is applied (edge_thickness_mm=0).
    """
    from scitex.plt.utils import mm_to_pt

    # Use centralized defaults if not specified
    if size_mm is None:
        size_mm = _DEFAULT_SIZE_MM
    if edge_thickness_mm is None:
        edge_thickness_mm = _DEFAULT_EDGE_THICKNESS_MM

    # Convert mm to points
    size_pt = mm_to_pt(size_mm)

    # Matplotlib scatter uses area (points^2)
    # For a marker of diameter d, area = (d/2)^2 * pi
    # But matplotlib's 's' parameter is already area-like
    # So we use size_pt^2 to get the right visual size
    marker_area = size_pt**2

    # Set marker size
    path_collection.set_sizes([marker_area])

    # Set edge thickness (0 by default = no border)
    edge_width_pt = mm_to_pt(edge_thickness_mm)
    path_collection.set_linewidths(edge_width_pt)

    return path_collection


# EOF
