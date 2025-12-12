#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/scitex_repo/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_lineplot.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_sns_lineplot(id, tracked_dict, kwargs):
    """Format data from a sns_lineplot call."""
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse the tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get the args from tracked_dict
    args = tracked_dict.get("args", [])

    # Line plot with potential error bands from seaborn
    if len(args) >= 1:
        data = args[0]
        x_var = kwargs.get("x")
        y_var = kwargs.get("y")

        # Handle DataFrame input with x, y variables
        if isinstance(data, pd.DataFrame) and x_var and y_var:
            result = pd.DataFrame(
                {
                    get_csv_column_name(x_var, ax_row, ax_col, trace_id=trace_id): data[x_var],
                    get_csv_column_name(y_var, ax_row, ax_col, trace_id=trace_id): data[y_var],
                }
            )

            # Add grouping variable if present
            hue_var = kwargs.get("hue")
            if hue_var and hue_var in data.columns:
                result[get_csv_column_name(hue_var, ax_row, ax_col, trace_id=trace_id)] = data[hue_var]

            return result

        # Handle direct x, y data arrays
        elif (
            len(args) > 1
            and isinstance(args[0], (np.ndarray, list))
            and isinstance(args[1], (np.ndarray, list))
        ):
            x_data, y_data = args[0], args[1]
            return pd.DataFrame({
                get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): x_data,
                get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): y_data
            })

        # Handle DataFrame input without x, y specified
        elif isinstance(data, pd.DataFrame):
            result = {}
            for col in data.columns:
                result[get_csv_column_name(col, ax_row, ax_col, trace_id=trace_id)] = data[col]
            return pd.DataFrame(result)

    return pd.DataFrame()
