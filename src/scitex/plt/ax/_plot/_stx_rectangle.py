#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 08:45:44 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/plt/ax/_plot/_plot_rectangle.py
# ----------------------------------------
import os

__FILE__ = "./src/scitex/plt/ax/_plot/_plot_rectangle.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

from matplotlib.patches import Rectangle


def stx_rectangle(ax, xx, yy, ww, hh, **kwargs):
    """Add a rectangle patch to an axes.

    Convenience function for adding rectangular patches to plots, useful for
    highlighting regions, creating box annotations, or drawing geometric shapes.
    By default, rectangles have no edge (border) for cleaner publication figures.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the rectangle to.
    xx : float
        X-coordinate of the rectangle's bottom-left corner.
    yy : float
        Y-coordinate of the rectangle's bottom-left corner.
    ww : float
        Width of the rectangle.
    hh : float
        Height of the rectangle.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib.patches.Rectangle.
        Common options include:
        - facecolor/fc : fill color
        - edgecolor/ec : edge color (default: 'none')
        - linewidth/lw : edge line width
        - alpha : transparency (0-1)
        - linestyle/ls : edge line style

    Returns
    -------
    matplotlib.axes.Axes
        The axes with the rectangle added.

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 10], [0, 10])
    >>> # Highlight a region (no border by default)
    >>> stx_rectangle(ax, 2, 3, 4, 3, facecolor='yellow', alpha=0.3)

    >>> # Draw a box with explicit edge
    >>> stx_rectangle(ax, 5, 5, 2, 2, facecolor='none', edgecolor='red', linewidth=2)

    See Also
    --------
    matplotlib.patches.Rectangle : The underlying Rectangle class
    matplotlib.axes.Axes.add_patch : Method used to add the patch
    """
    # Default to no edge for cleaner publication figures
    if "edgecolor" not in kwargs and "ec" not in kwargs:
        kwargs["edgecolor"] = "none"
    ax.add_patch(Rectangle((xx, yy), ww, hh, **kwargs))
    return ax


# EOF
