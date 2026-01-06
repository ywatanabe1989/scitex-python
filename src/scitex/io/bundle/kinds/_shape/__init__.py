#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/_shape/__init__.py

"""Shape kind - Shape annotation elements.

A shape bundle contains geometric shapes (no payload data).
Used for arrows, rectangles, circles, lines in figures.

Structure:
- canonical/node.json: Shape geometry and styling
- No payload (shape definition is in node specification)
"""

from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch


def render_shape(
    ax: "Axes",
    shape_type: str,
    x: float,
    y: float,
    width: Optional[float] = None,
    height: Optional[float] = None,
    radius: Optional[float] = None,
    x2: Optional[float] = None,
    y2: Optional[float] = None,
    facecolor: Optional[str] = None,
    edgecolor: Optional[str] = None,
    linewidth: float = 1.0,
    linestyle: str = "-",
    alpha: float = 1.0,
    **kwargs: Any,
) -> Optional["Patch"]:
    """Render shape annotation on axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render on
    shape_type : str
        Type of shape: rectangle, circle, ellipse, line, arrow
    x, y : float
        Position/start point in axes coordinates
    width, height : float, optional
        Dimensions for rectangle/ellipse
    radius : float, optional
        Radius for circle
    x2, y2 : float, optional
        End point for line/arrow
    facecolor : str, optional
        Fill color
    edgecolor : str, optional
        Edge/line color
    linewidth : float
        Line width in points
    linestyle : str
        Line style (-, --, :, -.)
    alpha : float
        Opacity (0-1)
    **kwargs : Any
        Additional kwargs passed to patch constructor

    Returns
    -------
    Patch or None
        The created matplotlib patch, or None for lines
    """
    from matplotlib.patches import (
        Rectangle,
        Circle,
        Ellipse,
        FancyArrowPatch,
    )
    from matplotlib.lines import Line2D

    patch_kwargs = {
        "linewidth": linewidth,
        "linestyle": linestyle,
        "alpha": alpha,
    }
    if facecolor is not None:
        patch_kwargs["facecolor"] = facecolor
    if edgecolor is not None:
        patch_kwargs["edgecolor"] = edgecolor
    patch_kwargs.update(kwargs)

    patch = None

    if shape_type == "rectangle":
        w = width or 0.1
        h = height or 0.1
        patch = Rectangle(
            (x, y), w, h, transform=ax.transAxes, **patch_kwargs
        )
        ax.add_patch(patch)

    elif shape_type == "circle":
        r = radius or 0.05
        patch = Circle((x, y), r, transform=ax.transAxes, **patch_kwargs)
        ax.add_patch(patch)

    elif shape_type == "ellipse":
        w = width or 0.1
        h = height or 0.05
        patch = Ellipse((x, y), w, h, transform=ax.transAxes, **patch_kwargs)
        ax.add_patch(patch)

    elif shape_type == "line":
        line = Line2D(
            [x, x2 or x + 0.1],
            [y, y2 or y],
            transform=ax.transAxes,
            color=edgecolor or "black",
            linewidth=linewidth,
            linestyle=linestyle,
            alpha=alpha,
        )
        ax.add_line(line)

    elif shape_type == "arrow":
        patch = FancyArrowPatch(
            (x, y),
            (x2 or x + 0.1, y2 or y),
            transform=ax.transAxes,
            arrowstyle="->",
            mutation_scale=10,
            **patch_kwargs,
        )
        ax.add_patch(patch)

    return patch


__all__ = ["render_shape"]

# EOF
