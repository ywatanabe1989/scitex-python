#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_sns_kdeplot.py

"""CSV formatter for sns.kdeplot() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_sns_kdeplot(id, tracked_dict, kwargs):
    """Format data from a sns_kdeplot call.

    Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to sns_kdeplot

    Returns:
        pd.DataFrame: Formatted data with standard column names
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get args from tracked_dict
    args = tracked_dict.get("args", [])
    x_var = kwargs.get("x") if kwargs else None
    y_var = kwargs.get("y") if kwargs else None

    if len(args) >= 1:
        data = args[0]

        # Handle DataFrame input with x, y variables
        if isinstance(data, pd.DataFrame) and x_var:
            if y_var and y_var in data.columns:  # Bivariate KDE
                return pd.DataFrame({
                    get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): data[x_var],
                    get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): data[y_var],
                })
            elif x_var in data.columns:  # Univariate KDE
                return pd.DataFrame({
                    get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): data[x_var]
                })

        # Handle direct data array input
        elif isinstance(data, (np.ndarray, list)):
            y_data = (
                args[1]
                if len(args) > 1 and isinstance(args[1], (np.ndarray, list))
                else None
            )

            if y_data is not None:  # Bivariate KDE
                return pd.DataFrame({
                    get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): data,
                    get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): y_data,
                })
            else:  # Univariate KDE
                return pd.DataFrame({
                    get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): data
                })

        # Handle DataFrame input without x, y specified
        elif isinstance(data, pd.DataFrame):
            result = pd.DataFrame()
            for col in data.columns:
                col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
                result[col_name] = data[col]
            return result

    # Also check for 'data' key directly
    if "data" in tracked_dict:
        data = tracked_dict["data"]
        if isinstance(data, pd.DataFrame):
            result = pd.DataFrame()
            for col in data.columns:
                col_name = get_csv_column_name(f"data-{col}", ax_row, ax_col, trace_id=trace_id)
                result[col_name] = data[col]
            return result

    return pd.DataFrame()
