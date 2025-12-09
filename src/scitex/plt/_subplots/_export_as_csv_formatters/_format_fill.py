#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_fill.py

import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_fill(id, tracked_dict, kwargs):
    """Format data from a fill call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to fill

    Returns:
        pd.DataFrame: Formatted data from fill plot
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    args = tracked_dict.get("args", [])

    # Fill creates a polygon based on points
    if len(args) >= 2:
        x = args[0]
        col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
        data = {col_x: x}

        for i, y in enumerate(args[1:]):
            col_y = get_csv_column_name(f"y{i:02d}", ax_row, ax_col, trace_id=trace_id)
            data[col_y] = y

        return pd.DataFrame(data)

    return pd.DataFrame()


# EOF
