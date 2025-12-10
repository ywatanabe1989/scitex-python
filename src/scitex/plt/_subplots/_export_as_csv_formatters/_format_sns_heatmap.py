#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_heatmap.py

"""CSV formatter for sns.heatmap() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_sns_heatmap(id, tracked_dict, kwargs):
    """Format data from a sns_heatmap call.

    Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to sns_heatmap

    Returns:
        pd.DataFrame: Formatted data with standard column names
    """
    # Check if tracked_dict is empty
    if not tracked_dict:
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    def _format_dataframe(df):
        result = pd.DataFrame()
        for col in df.columns:
            col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
            result[col_name] = df[col]
        return result

    def _format_array(arr):
        rows, cols = arr.shape if len(arr.shape) >= 2 else (arr.shape[0], 1)
        result = pd.DataFrame()
        for i in range(cols):
            col_data = arr[:, i] if len(arr.shape) >= 2 else arr
            col_name = get_csv_column_name(f"data-col-{i}", ax_row, ax_col, trace_id=trace_id)
            result[col_name] = col_data
        return result

    # If tracked_dict is a dictionary
    if isinstance(tracked_dict, dict):
        if "data" in tracked_dict:
            data = tracked_dict["data"]

            if isinstance(data, pd.DataFrame):
                return _format_dataframe(data)
            elif isinstance(data, np.ndarray):
                return _format_array(data)

        # Legacy handling for args
        args = tracked_dict.get("args", [])
        if len(args) > 0:
            data = args[0]

            if isinstance(data, pd.DataFrame):
                return _format_dataframe(data)
            elif isinstance(data, np.ndarray):
                return _format_array(data)

    # If tracked_dict is a DataFrame directly
    elif isinstance(tracked_dict, pd.DataFrame):
        return _format_dataframe(tracked_dict)

    # If tracked_dict is a numpy array directly
    elif isinstance(tracked_dict, np.ndarray):
        return _format_array(tracked_dict)

    return pd.DataFrame()
