#!/usr/bin/env python3
# Timestamp: 2025-12-20
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_bundle/_extractors/_extract_scatter.py

"""Scatter plot data and encoding extraction."""

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from ..._fig._dataclasses import TraceEncoding


def extract_scatter_data(ax: "Axes", ax_idx: int) -> Dict[str, np.ndarray]:
    """Extract scatter plot data from axes (PathCollection).

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for column naming

    Returns:
        Dict mapping column names to data arrays
    """
    from matplotlib.collections import PathCollection

    data = {}
    scatter_idx = 0

    for child in ax.get_children():
        if isinstance(child, PathCollection):
            offsets = child.get_offsets()
            if len(offsets) > 0:
                x_col = f"ax{ax_idx}_scatter{scatter_idx}_x"
                y_col = f"ax{ax_idx}_scatter{scatter_idx}_y"
                data[x_col] = np.array(offsets[:, 0], dtype=float)
                data[y_col] = np.array(offsets[:, 1], dtype=float)
                scatter_idx += 1

    return data


def build_scatter_traces(ax: "Axes", ax_idx: int) -> List["TraceEncoding"]:
    """Build encoding traces for scatter plots.

    Args:
        ax: Matplotlib axes
        ax_idx: Axes index for trace ID

    Returns:
        List of TraceEncoding objects
    """
    from matplotlib.collections import PathCollection

    from ..._fig._dataclasses import ChannelEncoding, TraceEncoding

    traces = []
    scatter_idx = 0

    for child in ax.get_children():
        if isinstance(child, PathCollection):
            offsets = child.get_offsets()
            if len(offsets) > 0:
                trace = TraceEncoding(
                    trace_id=f"scatter_{ax_idx}_{scatter_idx}",
                    x=ChannelEncoding(column=f"ax{ax_idx}_scatter{scatter_idx}_x"),
                    y=ChannelEncoding(column=f"ax{ax_idx}_scatter{scatter_idx}_y"),
                )
                traces.append(trace)
                scatter_idx += 1

    return traces


__all__ = ["extract_scatter_data", "build_scatter_traces"]

# EOF
