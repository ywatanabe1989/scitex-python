#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_joyplot.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
from scitex.pd import force_df
from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_plot_joyplot(id, tracked_dict, kwargs):
    """Format data from a stx_joyplot call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing 'joyplot_data' key with joyplot data
        kwargs (dict): Keyword arguments passed to stx_joyplot

    Returns:
        pd.DataFrame: Formatted joyplot data
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get joyplot_data from tracked_dict
    data = tracked_dict.get("joyplot_data")

    if data is None:
        return pd.DataFrame()

    # Handle different data types
    if isinstance(data, pd.DataFrame):
        # Make a copy to avoid modifying original
        result = data.copy()
        # Add prefix to column names using single source of truth
        if id is not None:
            result.columns = [
                get_csv_column_name(f"joyplot-{col}", ax_row, ax_col, trace_id=trace_id)
                for col in result.columns
            ]
        return result

    elif isinstance(data, dict):
        # Convert dictionary to DataFrame
        result = pd.DataFrame()
        for group, values in data.items():
            col_name = get_csv_column_name(
                f"joyplot-{group}", ax_row, ax_col, trace_id=trace_id
            )
            result[col_name] = pd.Series(values)
        return result

    elif isinstance(data, (list, tuple)) and all(
        isinstance(x, (np.ndarray, list)) for x in data
    ):
        # Convert list of arrays to DataFrame
        result = pd.DataFrame()
        for i, values in enumerate(data):
            col_name = get_csv_column_name(
                f"joyplot-group{i:02d}", ax_row, ax_col, trace_id=trace_id
            )
            result[col_name] = pd.Series(values)
        return result

    # Try to force to DataFrame as a last resort
    try:
        col_name = get_csv_column_name(
            "joyplot-data", ax_row, ax_col, trace_id=trace_id
        )
        return force_df({col_name: data})
    except:
        return pd.DataFrame()
