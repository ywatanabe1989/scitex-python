#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_histplot.py

"""CSV formatter for sns.histplot() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_sns_histplot(id, tracked_dict, kwargs):
    """Format data from a sns_histplot call as a bar plot representation.

    Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to sns_histplot

    Returns:
        pd.DataFrame: Formatted data with standard column names
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    columns = {}

    # Check if histogram result is available in tracked_dict
    hist_result = tracked_dict.get("hist_result", None)

    # If we have histogram result (counts and bin edges)
    if hist_result is not None:
        counts, bin_edges = hist_result

        # Calculate bin centers for bar plot representation
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Add bin information with standard naming
        columns[get_csv_column_name("bin-centers", ax_row, ax_col, trace_id=trace_id)] = bin_centers
        columns[get_csv_column_name("bin-counts", ax_row, ax_col, trace_id=trace_id)] = counts
        columns[get_csv_column_name("bin-widths", ax_row, ax_col, trace_id=trace_id)] = bin_widths
        columns[get_csv_column_name("bin-edges-left", ax_row, ax_col, trace_id=trace_id)] = bin_edges[:-1]
        columns[get_csv_column_name("bin-edges-right", ax_row, ax_col, trace_id=trace_id)] = bin_edges[1:]

    # Get raw data if available
    if "data" in tracked_dict:
        df = tracked_dict["data"]
        if isinstance(df, pd.DataFrame):
            x_col = kwargs.get("x")
            if x_col and x_col in df.columns:
                columns[get_csv_column_name("raw-data", ax_row, ax_col, trace_id=trace_id)] = df[x_col].values

    # Legacy handling for args
    elif "args" in tracked_dict:
        args = tracked_dict["args"]
        if len(args) >= 1:
            x = args[0]
            if hasattr(x, "values"):
                columns[get_csv_column_name("raw-data", ax_row, ax_col, trace_id=trace_id)] = x.values
            else:
                columns[get_csv_column_name("raw-data", ax_row, ax_col, trace_id=trace_id)] = x

    # If we have data to return
    if columns:
        # Ensure all arrays are the same length by padding with NaN
        max_length = max(
            len(value) for value in columns.values() if hasattr(value, "__len__")
        )
        for key, value in list(columns.items()):
            if hasattr(value, "__len__") and len(value) < max_length:
                if isinstance(value, np.ndarray):
                    columns[key] = np.pad(
                        value,
                        (0, max_length - len(value)),
                        mode="constant",
                        constant_values=np.nan,
                    )
                else:
                    padded = list(value) + [np.nan] * (max_length - len(value))
                    columns[key] = np.array(padded)

        return pd.DataFrame(columns)

    return pd.DataFrame()
