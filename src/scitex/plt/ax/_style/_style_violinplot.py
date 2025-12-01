#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 14:15:00 (ywatanabe)"
# File: ./src/scitex/plt/ax/_style/_style_violinplot.py

"""Style violin plot elements with millimeter-based control."""

from typing import Optional, Union

from matplotlib.axes import Axes


def style_violinplot(
    ax: Union[Axes, "AxisWrapper"],
    linewidth_mm: float = 0.2,
    edge_color: str = "black",
    remove_caps: bool = True,
) -> Union[Axes, "AxisWrapper"]:
    """Apply publication-quality styling to seaborn violin plots.

    This function modifies violin plots created by seaborn.violinplot() to:
    - Add borders to the KDE (violin body) edges
    - Remove caps from the internal boxplot whiskers
    - Apply consistent line widths

    Parameters
    ----------
    ax : matplotlib.axes.Axes or AxisWrapper
        The axes containing the violin plot.
    linewidth_mm : float, default 0.2
        Line width in millimeters for violin edges and boxplot elements.
    edge_color : str, default "black"
        Color for the violin body edges.
    remove_caps : bool, default True
        Whether to remove the caps (horizontal lines) from boxplot whiskers.

    Returns
    -------
    ax : matplotlib.axes.Axes or AxisWrapper
        The axes with styled violin plot.

    Examples
    --------
    >>> import seaborn as sns
    >>> import scitex as stx
    >>> fig, ax = stx.plt.subplots()
    >>> sns.violinplot(data=df, x="group", y="value", ax=ax)
    >>> stx.plt.ax.style_violinplot(ax)
    """
    from scitex.plt.utils import mm_to_pt

    lw_pt = mm_to_pt(linewidth_mm)

    # Style violin bodies (PolyCollection)
    for collection in ax.collections:
        # Check if it's a violin body (PolyCollection with filled area)
        if hasattr(collection, 'set_edgecolor'):
            collection.set_edgecolor(edge_color)
            collection.set_linewidth(lw_pt)

    # Style internal boxplot elements (Line2D objects)
    for line in ax.lines:
        # Get line data to identify element type
        xdata = line.get_xdata()
        ydata = line.get_ydata()

        # Caps are horizontal lines (same y-value for both points)
        is_horizontal = len(ydata) == 2 and ydata[0] == ydata[1]

        if remove_caps and is_horizontal:
            # Hide caps by making them invisible
            line.set_visible(False)
        else:
            # Style other lines (whiskers, median, etc.)
            line.set_linewidth(lw_pt)

    return ax


# EOF
