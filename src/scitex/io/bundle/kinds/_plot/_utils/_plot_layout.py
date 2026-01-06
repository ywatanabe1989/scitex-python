#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fsb/_fig/_utils/_plot_layout.py

"""Blueprint-style visualization for FTS layout and coordinate system.

Provides architectural drawing style visualizations with:
- Canvas boundaries with dimension annotations
- Element bounding boxes with labels
- Rulers (horizontal and vertical)
- Grid lines
- Before/after comparison for auto-crop

Usage:
    from scitex.io.bundle._fig._utils import plot_layout, plot_auto_crop_comparison

    # Single layout visualization
    fig, ax = plot_layout(elements, canvas_size, title="My Figure")

    # Before/after auto-crop comparison
    fig = plot_auto_crop_comparison(elements_before, elements_after,
                                     size_before, size_after)
"""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._calc_bounds import element_bounds
from ._normalize import normalize_size

__all__ = [
    "plot_layout",
    "plot_auto_crop_comparison",
    "BLUEPRINT_STYLE",
]

# Blueprint color scheme
BLUEPRINT_STYLE = {
    "bg_color": "#1a2744",  # Dark blue background
    "grid_color": "#2a3f5f",  # Subtle grid
    "canvas_color": "#ffffff",  # White canvas
    "canvas_edge": "#4a90d9",  # Blue canvas border
    "element_fill": "#e8f4fc",  # Light blue element fill
    "element_edge": "#2171b5",  # Blue element border
    "ruler_color": "#ff6b35",  # Orange rulers
    "text_color": "#333333",  # Dark text
    "dimension_color": "#d62728",  # Red dimensions
    "origin_color": "#2ca02c",  # Green origin marker
}


def plot_layout(
    elements: List[Dict[str, Any]],
    canvas_size: Dict[str, float],
    title: str = "Layout",
    ax: Optional[Axes] = None,
    show_rulers: bool = True,
    show_grid: bool = True,
    show_dimensions: bool = True,
    show_origin: bool = True,
    style: Optional[Dict[str, str]] = None,
) -> Tuple[Figure, Axes]:
    """Plot layout with blueprint-style visualization.

    Args:
        elements: List of element specifications
        canvas_size: Canvas size {"width_mm", "height_mm"}
        title: Plot title
        ax: Existing axes to plot on (creates new if None)
        show_rulers: Show ruler markings
        show_grid: Show background grid
        show_dimensions: Show dimension annotations
        show_origin: Show origin marker
        style: Custom style dict (uses BLUEPRINT_STYLE if None)

    Returns:
        Tuple of (Figure, Axes)
    """
    s = style or BLUEPRINT_STYLE
    size = normalize_size(canvas_size)
    w, h = size["width_mm"], size["height_mm"]

    # Create figure if needed
    if ax is None:
        # Add space for rulers
        ruler_margin = 15 if show_rulers else 5
        fig_w = (w + ruler_margin * 2) / 25.4  # Convert mm to inches
        fig_h = (h + ruler_margin * 2) / 25.4
        fig, ax = plt.subplots(figsize=(fig_w * 1.5, fig_h * 1.5))
    else:
        fig = ax.figure

    # Set background
    ax.set_facecolor(s["bg_color"])

    # Draw grid
    if show_grid:
        _draw_grid(ax, w, h, s)

    # Draw canvas
    _draw_canvas(ax, w, h, s)

    # Draw origin marker
    if show_origin:
        _draw_origin(ax, s)

    # Draw elements
    for i, elem in enumerate(elements):
        _draw_element(ax, elem, i, s, show_dimensions)

    # Draw rulers
    if show_rulers:
        _draw_rulers(ax, w, h, s)

    # Draw canvas dimensions
    if show_dimensions:
        _draw_canvas_dimensions(ax, w, h, s)

    # Set axis properties
    margin = 20 if show_rulers else 5
    ax.set_xlim(-margin, w + margin)
    ax.set_ylim(h + margin, -margin)  # Inverted Y (origin at top-left)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12, fontweight="bold", color=s["text_color"])
    ax.axis("off")

    return fig, ax


def plot_auto_crop_comparison(
    elements_before: List[Dict[str, Any]],
    elements_after: List[Dict[str, Any]],
    size_before: Dict[str, float],
    size_after: Dict[str, float],
    title: str = "Auto-Crop Comparison",
    style: Optional[Dict[str, str]] = None,
) -> Figure:
    """Plot before/after comparison for auto-crop.

    Args:
        elements_before: Elements before auto-crop
        elements_after: Elements after auto-crop
        size_before: Canvas size before
        size_after: Canvas size after
        title: Overall title
        style: Custom style dict

    Returns:
        Figure with side-by-side comparison
    """
    s = style or BLUEPRINT_STYLE

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Before
    plot_layout(
        elements_before,
        size_before,
        title="Before Auto-Crop",
        ax=ax1,
        style=s,
    )

    # After
    plot_layout(
        elements_after,
        size_after,
        title="After Auto-Crop",
        ax=ax2,
        style=s,
    )

    # Add size annotations
    sb = normalize_size(size_before)
    sa = normalize_size(size_after)
    ax1.text(
        0.5,
        -0.05,
        f"Canvas: {sb['width_mm']:.1f} x {sb['height_mm']:.1f} mm",
        transform=ax1.transAxes,
        ha="center",
        fontsize=10,
    )
    ax2.text(
        0.5,
        -0.05,
        f"Canvas: {sa['width_mm']:.1f} x {sa['height_mm']:.1f} mm",
        transform=ax2.transAxes,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    return fig


def _draw_grid(ax: Axes, w: float, h: float, s: Dict[str, str]) -> None:
    """Draw background grid."""
    # Major grid every 10mm
    for x in range(0, int(w) + 1, 10):
        ax.axvline(x, color=s["grid_color"], linewidth=0.5, alpha=0.5)
    for y in range(0, int(h) + 1, 10):
        ax.axhline(y, color=s["grid_color"], linewidth=0.5, alpha=0.5)


def _draw_canvas(ax: Axes, w: float, h: float, s: Dict[str, str]) -> None:
    """Draw canvas rectangle."""
    rect = mpatches.Rectangle(
        (0, 0),
        w,
        h,
        linewidth=2,
        edgecolor=s["canvas_edge"],
        facecolor=s["canvas_color"],
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(rect)


def _draw_origin(ax: Axes, s: Dict[str, str]) -> None:
    """Draw origin marker at (0,0)."""
    # Origin cross
    ax.plot([-3, 3], [0, 0], color=s["origin_color"], linewidth=2, zorder=10)
    ax.plot([0, 0], [-3, 3], color=s["origin_color"], linewidth=2, zorder=10)
    # Origin label
    ax.text(
        -5,
        -5,
        "(0,0)",
        fontsize=8,
        color=s["origin_color"],
        ha="right",
        va="bottom",
        fontweight="bold",
    )


def _draw_element(
    ax: Axes,
    elem: Dict[str, Any],
    index: int,
    s: Dict[str, str],
    show_dimensions: bool,
) -> None:
    """Draw a single element with bounding box."""
    bounds = element_bounds(elem)
    x, y = bounds["x_mm"], bounds["y_mm"]
    w, h = bounds["width_mm"], bounds["height_mm"]

    # Element rectangle
    rect = mpatches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=s["element_edge"],
        facecolor=s["element_fill"],
        alpha=0.8,
        zorder=5,
    )
    ax.add_patch(rect)

    # Element label
    elem_id = elem.get("id", f"E{index}")
    elem_type = elem.get("type", "unknown")
    label = f"{elem_id}\n({elem_type})"
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=9,
        color=s["text_color"],
        fontweight="bold",
        zorder=6,
    )

    # Position annotation
    if show_dimensions:
        ax.text(
            x,
            y - 2,
            f"({x:.0f}, {y:.0f})",
            fontsize=7,
            color=s["dimension_color"],
            ha="left",
            va="bottom",
        )
        # Size annotation
        ax.text(
            x + w,
            y + h + 2,
            f"{w:.0f}x{h:.0f}",
            fontsize=7,
            color=s["dimension_color"],
            ha="right",
            va="top",
        )


def _draw_rulers(ax: Axes, w: float, h: float, s: Dict[str, str]) -> None:
    """Draw rulers along edges."""
    ruler_offset = -12

    # Horizontal ruler (top)
    ax.plot([0, w], [ruler_offset, ruler_offset], color=s["ruler_color"], linewidth=1)
    for x in range(0, int(w) + 1, 10):
        tick_len = 3 if x % 50 == 0 else 1.5
        ax.plot(
            [x, x],
            [ruler_offset, ruler_offset + tick_len],
            color=s["ruler_color"],
            linewidth=1,
        )
        if x % 50 == 0:
            ax.text(
                x,
                ruler_offset - 2,
                str(x),
                fontsize=7,
                ha="center",
                va="bottom",
                color=s["ruler_color"],
            )

    # Vertical ruler (left)
    ax.plot([ruler_offset, ruler_offset], [0, h], color=s["ruler_color"], linewidth=1)
    for y in range(0, int(h) + 1, 10):
        tick_len = 3 if y % 50 == 0 else 1.5
        ax.plot(
            [ruler_offset, ruler_offset + tick_len],
            [y, y],
            color=s["ruler_color"],
            linewidth=1,
        )
        if y % 50 == 0:
            ax.text(
                ruler_offset - 2,
                y,
                str(y),
                fontsize=7,
                ha="right",
                va="center",
                color=s["ruler_color"],
            )


def _draw_canvas_dimensions(ax: Axes, w: float, h: float, s: Dict[str, str]) -> None:
    """Draw canvas dimension annotations."""
    # Width dimension (bottom)
    y_pos = h + 8
    ax.annotate(
        "",
        xy=(w, y_pos),
        xytext=(0, y_pos),
        arrowprops=dict(arrowstyle="<->", color=s["dimension_color"], lw=1.5),
    )
    ax.text(
        w / 2,
        y_pos + 3,
        f"{w:.0f} mm",
        ha="center",
        va="bottom",
        fontsize=9,
        color=s["dimension_color"],
        fontweight="bold",
    )

    # Height dimension (right)
    x_pos = w + 8
    ax.annotate(
        "",
        xy=(x_pos, h),
        xytext=(x_pos, 0),
        arrowprops=dict(arrowstyle="<->", color=s["dimension_color"], lw=1.5),
    )
    ax.text(
        x_pos + 3,
        h / 2,
        f"{h:.0f} mm",
        ha="left",
        va="center",
        fontsize=9,
        color=s["dimension_color"],
        fontweight="bold",
        rotation=90,
    )


# EOF
