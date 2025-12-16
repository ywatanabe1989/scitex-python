#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_raster.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd
from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_plot_raster(id, tracked_dict, kwargs):
    """Format data from a stx_raster call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing 'raster_digit_df' key with raster plot data
        kwargs (dict): Keyword arguments passed to stx_raster

    Returns:
        pd.DataFrame: Formatted raster plot data
    """
    # Check if args is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get the raster_digit_df from args
    raster_df = tracked_dict.get("raster_digit_df")

    if raster_df is None or not isinstance(raster_df, pd.DataFrame):
        return pd.DataFrame()

    # Create a copy to avoid modifying the original
    result = raster_df.copy()

    # Add prefix to column names using single source of truth
    if id is not None:
        # Rename columns with ID prefix
        result.columns = [
            get_csv_column_name(f"raster-{col}", ax_row, ax_col, trace_id=trace_id)
            for col in result.columns
        ]

    return result
