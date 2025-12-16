#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 02:30:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_box.py

"""CSV formatter for stx_box() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_plot_box(id, tracked_dict, kwargs):
    """Format data from a stx_box call.

    Uses standard column naming: ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to stx_box

    Returns:
        pd.DataFrame: Formatted box plot data with standard column names
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # First try to get data directly from tracked_dict
    data = tracked_dict.get("data")

    # If no data key, get from args
    if data is None:
        args = tracked_dict.get("args", [])
        if len(args) >= 1:
            data = args[0]
        else:
            return pd.DataFrame()

    # If data is a simple array or list of values
    if isinstance(data, (np.ndarray, list)) and len(data) > 0:
        try:
            # Check if it's a simple list of values or a list of lists
            if isinstance(data[0], (int, float, np.number)):
                col_name = get_csv_column_name(
                    "data", ax_row, ax_col, trace_id=trace_id
                )
                return pd.DataFrame({col_name: data})

            # If data is a list of arrays (multiple box plots)
            elif isinstance(data, (list, tuple)) and all(
                isinstance(x, (list, np.ndarray)) for x in data
            ):
                result = pd.DataFrame()
                for i, values in enumerate(data):
                    try:
                        col_name = get_csv_column_name(
                            f"data-{i}", ax_row, ax_col, trace_id=trace_id
                        )
                        result[col_name] = pd.Series(values)
                    except Exception:
                        pass
                return result
        except (IndexError, TypeError):
            pass

    # If data is a dictionary
    elif isinstance(data, dict):
        result = pd.DataFrame()
        for label, values in data.items():
            try:
                col_name = get_csv_column_name(
                    f"data-{label}", ax_row, ax_col, trace_id=trace_id
                )
                result[col_name] = pd.Series(values)
            except Exception:
                pass
        return result

    # If data is a DataFrame
    elif isinstance(data, pd.DataFrame):
        result = pd.DataFrame()
        for col in data.columns:
            col_name = get_csv_column_name(
                f"data-{col}", ax_row, ax_col, trace_id=trace_id
            )
            result[col_name] = data[col]
        return result

    # Default case: return empty DataFrame if nothing could be processed
    return pd.DataFrame()
