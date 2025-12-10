#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 12:00:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_fillv.py

"""CSV formatter for stx_fillv() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_plot_fillv(id, tracked_dict, kwargs):
    """Format data from a stx_fillv call.

    Formats data similar to line plot format for better compatibility.
    Uses standard column naming convention:
    (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to stx_fillv

    Returns:
        pd.DataFrame: Formatted fillv data in a long-format dataframe
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Try to get starts/ends directly from tracked_dict first
    starts = tracked_dict.get("starts")
    ends = tracked_dict.get("ends")

    # If not found, get from args
    if starts is None or ends is None:
        args = tracked_dict.get("args", [])

        # Extract data if available from args
        if len(args) >= 2:
            starts, ends = args[0], args[1]

    # If we have valid starts and ends, create a DataFrame in a format similar to line plot
    if starts is not None and ends is not None:
        # Convert to numpy arrays if they're lists for better handling
        if isinstance(starts, list):
            starts = np.array(starts)
        if isinstance(ends, list):
            ends = np.array(ends)

        # Get standard column names
        x_col = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
        y_col = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
        type_col = get_csv_column_name("type", ax_row, ax_col, trace_id=trace_id)

        # Create a DataFrame with x, y pairs for each fill span
        rows = []
        for start, end in zip(starts, ends):
            rows.append({x_col: start, y_col: 0, type_col: "start"})
            rows.append({x_col: end, y_col: 0, type_col: "end"})

        if rows:
            return pd.DataFrame(rows)

    return pd.DataFrame()
