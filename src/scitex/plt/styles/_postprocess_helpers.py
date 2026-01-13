#!/usr/bin/env python3
# Timestamp: "2026-01-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/styles/_postprocess_helpers.py

"""Helper functions for plot post-processing.

Extracted from _plot_postprocess.py to keep modules within line limits.
Delegates to figrecipe styling functions when available.
"""

import numpy as np

# Try to import figrecipe styling (delegate when available)
try:
    from figrecipe.styles._plot_styles import (
        apply_barplot_style as _fr_apply_barplot_style,
    )
    from figrecipe.styles._plot_styles import (
        apply_histogram_style as _fr_apply_histogram_style,
    )

    FIGRECIPE_AVAILABLE = True
except ImportError:
    FIGRECIPE_AVAILABLE = False

# ============================================================================
# Constants
# ============================================================================
CAP_WIDTH_RATIO = 1 / 3  # 33% of bar/box width


# ============================================================================
# Helper functions
# ============================================================================
def calculate_cap_width_from_box(box, ax):
    """Calculate cap width as 33% of box width in points."""
    # Get box width from path
    if hasattr(box, "get_path"):
        path = box.get_path()
        vertices = path.vertices
        x_coords = vertices[:, 0]
        box_width_data = x_coords.max() - x_coords.min()
    elif hasattr(box, "get_xdata"):
        x_data = box.get_xdata()
        box_width_data = max(x_data) - min(x_data)
    else:
        box_width_data = 0.5  # Default

    return data_width_to_points(box_width_data, ax, "x") * CAP_WIDTH_RATIO


def calculate_cap_width_from_bar(patch, ax, dimension):
    """Calculate cap width as 33% of bar width/height in points."""
    if dimension == "width":
        bar_size = patch.get_width()
        return data_width_to_points(bar_size, ax, "x") * CAP_WIDTH_RATIO
    else:  # height
        bar_size = patch.get_height()
        return data_width_to_points(bar_size, ax, "y") * CAP_WIDTH_RATIO


def data_width_to_points(data_size, ax, axis="x"):
    """Convert a data-space size to points."""
    fig = ax.get_figure()
    bbox = ax.get_position()

    if axis == "x":
        ax_size_inches = bbox.width * fig.get_figwidth()
        lim = ax.get_xlim()
    else:
        ax_size_inches = bbox.height * fig.get_figheight()
        lim = ax.get_ylim()

    data_range = lim[1] - lim[0]
    size_inches = (data_size / data_range) * ax_size_inches
    return size_inches * 72  # 72 points per inch


def make_errorbar_one_sided(barlinecols, direction):
    """Make error bar line segments one-sided (outward only)."""
    if not barlinecols or len(barlinecols) == 0:
        return

    for lc in barlinecols:
        if not hasattr(lc, "get_segments"):
            continue

        segs = lc.get_segments()
        new_segs = []
        for seg in segs:
            if len(seg) < 2:
                continue

            if direction == "vertical":
                # Keep upper half
                bottom_y = min(seg[0][1], seg[1][1])
                top_y = max(seg[0][1], seg[1][1])
                mid_y = (bottom_y + top_y) / 2
                new_seg = np.array([[seg[0][0], mid_y], [seg[0][0], top_y]])
            else:  # horizontal
                # Keep right half
                left_x = min(seg[0][0], seg[1][0])
                right_x = max(seg[0][0], seg[1][0])
                mid_x = (left_x + right_x) / 2
                new_seg = np.array([[mid_x, seg[0][1]], [right_x, seg[0][1]]])

            new_segs.append(new_seg)

        if new_segs:
            lc.set_segments(new_segs)


def apply_bar_edge_style(ax, line_width_mm):
    """Apply bar edge styling, delegating to figrecipe if available.

    Parameters
    ----------
    ax : matplotlib Axes or AxisWrapper
        The axes containing bar patches.
    line_width_mm : float
        Line width in millimeters.
    """
    from scitex.plt.utils import mm_to_pt

    ax_mpl = getattr(ax, "_axis_mpl", ax)

    if FIGRECIPE_AVAILABLE:
        _fr_apply_barplot_style(ax_mpl, {"barplot_edge_mm": line_width_mm})
    else:
        # Fallback: apply edge styling directly
        from matplotlib.patches import Rectangle

        line_width_pt = mm_to_pt(line_width_mm)
        for patch in ax_mpl.patches:
            if isinstance(patch, Rectangle):
                patch.set_edgecolor("black")
                patch.set_linewidth(line_width_pt)


def apply_hist_edge_style(ax, line_width_mm):
    """Apply histogram edge styling, delegating to figrecipe if available."""
    from scitex.plt.utils import mm_to_pt

    ax_mpl = getattr(ax, "_axis_mpl", ax)

    if FIGRECIPE_AVAILABLE:
        _fr_apply_histogram_style(ax_mpl, {"histogram_edge_mm": line_width_mm})
    else:
        from matplotlib.patches import Rectangle

        line_width_pt = mm_to_pt(line_width_mm)
        for patch in ax_mpl.patches:
            if isinstance(patch, Rectangle):
                patch.set_edgecolor("black")
                patch.set_linewidth(line_width_pt)


# EOF
