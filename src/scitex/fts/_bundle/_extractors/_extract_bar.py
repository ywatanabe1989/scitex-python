#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_extractors/_extract_bar.py

"""Bar chart data and encoding extraction."""

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._fig._dataclasses import TraceEncoding


def _is_valid_bar(rect, xlim, ylim, axes_width, axes_height) -> bool:
    """Check if a Rectangle is a valid bar (not UI element).

    Filters out legend boxes, axes frames, and other UI elements.
    """
    w = rect.get_width()
    h = rect.get_height()
    x = rect.get_x()
    y = rect.get_y()

    # Filter out: zero/negative dimensions
    if w <= 0 or h == 0:
        return False
    # Filter out: full-width elements (likely axes frame)
    if abs(w - axes_width) < 0.01 * axes_width:
        return False
    # Filter out: full-height elements
    if abs(h - axes_height) < 0.01 * axes_height:
        return False
    # Filter out: very thin bars (likely spines)
    if w < 0.01 * axes_width:
        return False
    # Filter out: elements outside data area
    if x < xlim[0] - 0.1 * axes_width or x > xlim[1]:
        return False
    if y < ylim[0] - 0.1 * axes_height:
        return False

    return True


def extract_bar_data(ax: "Axes", ax_idx: int) -> Dict[str, np.ndarray]:
    """Extract bar chart data from axes (Rectangle patches).

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for column naming

    Returns:
        Dict mapping column names to data arrays
    """
    from matplotlib.patches import Rectangle

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axes_width = xlim[1] - xlim[0]
    axes_height = ylim[1] - ylim[0]

    bars_x = []
    bars_height = []

    for child in ax.get_children():
        if isinstance(child, Rectangle):
            if _is_valid_bar(child, xlim, ylim, axes_width, axes_height):
                w = child.get_width()
                h = child.get_height()
                x = child.get_x()
                bars_x.append(x + w / 2)
                bars_height.append(h)

    data = {}
    if len(bars_x) >= 2:  # At least 2 bars to be a bar chart
        data[f"ax{ax_idx}_bar_x"] = np.array(bars_x, dtype=float)
        data[f"ax{ax_idx}_bar_height"] = np.array(bars_height, dtype=float)

    return data


def count_valid_bars(ax: "Axes") -> int:
    """Count valid bar rectangles in axes."""
    from matplotlib.patches import Rectangle

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axes_width = xlim[1] - xlim[0]
    axes_height = ylim[1] - ylim[0]

    count = 0
    for child in ax.get_children():
        if isinstance(child, Rectangle):
            if _is_valid_bar(child, xlim, ylim, axes_width, axes_height):
                count += 1

    return count


def build_bar_traces(ax: "Axes", ax_idx: int) -> List["TraceEncoding"]:
    """Build encoding traces for bar charts.

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for trace ID

    Returns:
        List of TraceEncoding objects
    """
    from ..._fig._dataclasses import ChannelEncoding, TraceEncoding

    traces = []
    bar_count = count_valid_bars(ax)

    if bar_count >= 2:
        trace = TraceEncoding(
            trace_id=f"bar_{ax_idx}",
            x=ChannelEncoding(column=f"ax{ax_idx}_bar_x"),
            y=ChannelEncoding(column=f"ax{ax_idx}_bar_height"),
        )
        traces.append(trace)

    return traces


__all__ = ["extract_bar_data", "count_valid_bars", "build_bar_traces"]

# EOF
