#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_scatter_hist.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_plot_scatter_hist(id, tracked_dict, kwargs):
    """Format data from a stx_scatter_hist call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to stx_scatter_hist

    Returns:
        pd.DataFrame: Formatted scatter histogram data
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to extract axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Extract data from tracked_dict
    x = tracked_dict.get("x")
    y = tracked_dict.get("y")

    if x is not None and y is not None:
        # Create base DataFrame with x and y values
        df = pd.DataFrame({
            get_csv_column_name("scatter_hist_x", ax_row, ax_col, trace_id=trace_id): x,
            get_csv_column_name("scatter_hist_y", ax_row, ax_col, trace_id=trace_id): y,
        })

        # Add histogram data if available
        hist_x = tracked_dict.get("hist_x")
        hist_y = tracked_dict.get("hist_y")
        bin_edges_x = tracked_dict.get("bin_edges_x")
        bin_edges_y = tracked_dict.get("bin_edges_y")

        # If we have histogram data
        if hist_x is not None and bin_edges_x is not None:
            # Calculate bin centers for x-axis histogram
            bin_centers_x = 0.5 * (bin_edges_x[1:] + bin_edges_x[:-1])

            # Create a DataFrame for x histogram data
            hist_x_df = pd.DataFrame(
                {
                    get_csv_column_name("hist_x_bin_centers", ax_row, ax_col, trace_id=trace_id): bin_centers_x,
                    get_csv_column_name("hist_x_counts", ax_row, ax_col, trace_id=trace_id): hist_x,
                }
            )

            # Add it to the main DataFrame using a MultiIndex
            for i, (center, count) in enumerate(zip(bin_centers_x, hist_x)):
                df.loc[f"hist_x_{i}", get_csv_column_name("hist_x_bin", ax_row, ax_col, trace_id=trace_id)] = center
                df.loc[f"hist_x_{i}", get_csv_column_name("hist_x_count", ax_row, ax_col, trace_id=trace_id)] = count

        # If we have y histogram data
        if hist_y is not None and bin_edges_y is not None:
            # Calculate bin centers for y-axis histogram
            bin_centers_y = 0.5 * (bin_edges_y[1:] + bin_edges_y[:-1])

            # Create a DataFrame for y histogram data
            hist_y_df = pd.DataFrame(
                {
                    get_csv_column_name("hist_y_bin_centers", ax_row, ax_col, trace_id=trace_id): bin_centers_y,
                    get_csv_column_name("hist_y_counts", ax_row, ax_col, trace_id=trace_id): hist_y,
                }
            )

            # Add it to the main DataFrame using a MultiIndex
            for i, (center, count) in enumerate(zip(bin_centers_y, hist_y)):
                df.loc[f"hist_y_{i}", get_csv_column_name("hist_y_bin", ax_row, ax_col, trace_id=trace_id)] = center
                df.loc[f"hist_y_{i}", get_csv_column_name("hist_y_count", ax_row, ax_col, trace_id=trace_id)] = count

        return df

    return pd.DataFrame()
