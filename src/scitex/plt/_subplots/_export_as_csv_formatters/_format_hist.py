#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_hist.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_hist(id, tracked_dict, kwargs):
    """
    Format data from a hist call as a bar plot representation.

    This formatter extracts both the raw data and the binned data from histogram plots,
    returning them in a format that can be visualized as a bar plot.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to hist

    Returns:
        pd.DataFrame: DataFrame containing both raw data and bin information
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get the args from tracked_dict
    args = tracked_dict.get("args", [])

    # Check if histogram result (bin counts and edges) is available in tracked_dict
    hist_result = tracked_dict.get("hist_result", None)

    columns = {}

    # Extract raw data if available
    if len(args) >= 1:
        x = args[0]
        col_raw = get_csv_column_name("raw-data", ax_row, ax_col, trace_id=trace_id)
        columns[col_raw] = x

    # If we have histogram result (counts and bin edges)
    if hist_result is not None:
        counts, bin_edges = hist_result

        # Calculate bin centers for bar plot representation
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_widths = bin_edges[1:] - bin_edges[:-1]

        # Use structured column naming
        col_centers = get_csv_column_name("bin-centers", ax_row, ax_col, trace_id=trace_id)
        col_counts = get_csv_column_name("bin-counts", ax_row, ax_col, trace_id=trace_id)
        col_widths = get_csv_column_name("bin-widths", ax_row, ax_col, trace_id=trace_id)
        col_left = get_csv_column_name("bin-edges-left", ax_row, ax_col, trace_id=trace_id)
        col_right = get_csv_column_name("bin-edges-right", ax_row, ax_col, trace_id=trace_id)

        # Add bin information to DataFrame
        columns[col_centers] = bin_centers
        columns[col_counts] = counts
        columns[col_widths] = bin_widths
        columns[col_left] = bin_edges[:-1]
        columns[col_right] = bin_edges[1:]

        # Create DataFrame with aligned length
        max_length = max(len(value) for value in columns.values())
        for key, value in list(columns.items()):
            if len(value) < max_length:
                # Pad with NaN if needed - convert to float first for NaN support
                arr = np.asarray(value, dtype=float)
                padded = np.full(max_length, np.nan)
                padded[:len(arr)] = arr
                columns[key] = padded

    # Return DataFrame or empty DataFrame if no data
    if columns:
        return pd.DataFrame(columns)

    return pd.DataFrame()
