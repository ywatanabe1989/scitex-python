#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 02:00:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_mean_ci.py

"""CSV formatter for stx_mean_ci() calls - uses standard column naming."""

import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_plot_mean_ci(id, tracked_dict, kwargs):
    """Format data from a stx_mean_ci call.

    Processes mean with confidence interval band plot data for CSV export using
    standard column naming (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Contains 'plot_df' (pandas DataFrame with mean and CI data)
        kwargs (dict): Keyword arguments passed to stx_mean_ci

    Returns:
        pd.DataFrame: Formatted mean and CI data with standard column names
    """
    # Mean-CI plot data is passed in the tracked_dict
    if not tracked_dict:
        return pd.DataFrame()

    # Get the plot_df from tracked_dict
    plot_df = tracked_dict.get("plot_df")

    if plot_df is None or not isinstance(plot_df, pd.DataFrame):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Create a copy to avoid modifying the original
    result = plot_df.copy()

    # Rename columns using standard naming convention
    renamed = {}
    for col in result.columns:
        renamed[col] = get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)

    return result.rename(columns=renamed)
