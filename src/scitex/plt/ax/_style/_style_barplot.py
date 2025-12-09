#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 20:00:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_barplot.py

"""
Style bar plot elements with millimeter-based control.

Default values are loaded from SCITEX_STYLE.yaml via presets.py.
"""

from typing import Optional, Union, List

from scitex.plt.styles.presets import SCITEX_STYLE

# Get defaults from centralized config
_DEFAULT_EDGE_THICKNESS_MM = SCITEX_STYLE.get("bar_edge_thickness_mm", 0.2)


def style_barplot(
    bar_container,
    edge_thickness_mm: float = None,
    edgecolor: Optional[Union[str, List[str]]] = "black",
):
    """
    Apply consistent styling to matplotlib bar plot elements.

    Parameters
    ----------
    bar_container : BarContainer
        Container returned by ax.bar() or ax.barh()
    edge_thickness_mm : float, optional
        Edge line thickness in millimeters (default: 0.2mm)
    edgecolor : str or list of str, optional
        Edge color(s) for bars. If None, uses default matplotlib colors.

    Returns
    -------
    bar_container : BarContainer
        The styled bar container

    Examples
    --------
    >>> fig, ax = stx.plt.subplots(**stx.plt.presets.NATURE_STYLE)
    >>> bars = ax.bar(x, heights)
    >>> stx.plt.ax.style_barplot(bars, edge_thickness_mm=0.2, edgecolor='black')
    """
    from scitex.plt.utils import mm_to_pt

    # Use centralized default if not specified
    if edge_thickness_mm is None:
        edge_thickness_mm = _DEFAULT_EDGE_THICKNESS_MM

    # Convert mm to points
    lw_pt = mm_to_pt(edge_thickness_mm)

    # Style each bar
    for i, bar in enumerate(bar_container):
        bar.set_linewidth(lw_pt)
        if edgecolor is not None:
            if isinstance(edgecolor, list):
                bar.set_edgecolor(edgecolor[i % len(edgecolor)])
            else:
                bar.set_edgecolor(edgecolor)

    return bar_container


# EOF
