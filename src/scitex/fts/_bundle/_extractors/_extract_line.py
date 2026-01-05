#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_extractors/_extract_line.py

"""Line plot data and encoding extraction."""

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._fig._dataclasses import TraceEncoding


def extract_line_data(ax: "Axes", ax_idx: int) -> Dict[str, np.ndarray]:
    """Extract line plot data from axes.

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for column naming

    Returns:
        Dict mapping column names to data arrays
    """
    data = {}
    for line_idx, line in enumerate(ax.get_lines()):
        label = line.get_label()
        if label is None or label.startswith("_"):
            label = f"series_{line_idx}"

        xdata, ydata = line.get_data()
        if len(xdata) > 0:
            x_col = f"ax{ax_idx}_line{line_idx}_x"
            y_col = f"ax{ax_idx}_line{line_idx}_y"
            data[x_col] = np.array(xdata, dtype=float)
            data[y_col] = np.array(ydata, dtype=float)

    return data


def build_line_traces(ax: "Axes", ax_idx: int) -> List["TraceEncoding"]:
    """Build encoding traces for line plots.

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for trace ID

    Returns:
        List of TraceEncoding objects
    """
    from ..._fig._dataclasses import ChannelEncoding, TraceEncoding

    traces = []
    for line_idx, line in enumerate(ax.get_lines()):
        label = line.get_label()
        if label and not label.startswith("_"):
            trace = TraceEncoding(
                trace_id=f"line_{ax_idx}_{line_idx}",
                x=ChannelEncoding(column=f"ax{ax_idx}_line{line_idx}_x"),
                y=ChannelEncoding(column=f"ax{ax_idx}_line{line_idx}_y"),
            )
            traces.append(trace)

    return traces


__all__ = ["extract_line_data", "build_line_traces"]

# EOF
