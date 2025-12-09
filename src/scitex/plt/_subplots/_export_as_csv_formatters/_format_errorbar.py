#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_errorbar.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_errorbar(id, tracked_dict, kwargs):
    """Format data from an errorbar call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to errorbar

    Returns:
        pd.DataFrame: Formatted data from errorbar plot
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    args = tracked_dict.get("args", [])

    if len(args) >= 2:
        x, y = args[:2]
        xerr = kwargs.get("xerr")
        yerr = kwargs.get("yerr")

        # Get column names from single source of truth
        col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
        col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)

        data = {col_x: x, col_y: y}

        if xerr is not None:
            if isinstance(xerr, (list, tuple)) and len(xerr) == 2:
                col_xerr_neg = get_csv_column_name("xerr-neg", ax_row, ax_col, trace_id=trace_id)
                col_xerr_pos = get_csv_column_name("xerr-pos", ax_row, ax_col, trace_id=trace_id)
                data[col_xerr_neg] = xerr[0]
                data[col_xerr_pos] = xerr[1]
            else:
                col_xerr = get_csv_column_name("xerr", ax_row, ax_col, trace_id=trace_id)
                data[col_xerr] = xerr

        if yerr is not None:
            if isinstance(yerr, (list, tuple)) and len(yerr) == 2:
                col_yerr_neg = get_csv_column_name("yerr-neg", ax_row, ax_col, trace_id=trace_id)
                col_yerr_pos = get_csv_column_name("yerr-pos", ax_row, ax_col, trace_id=trace_id)
                data[col_yerr_neg] = yerr[0]
                data[col_yerr_pos] = yerr[1]
            else:
                col_yerr = get_csv_column_name("yerr", ax_row, ax_col, trace_id=trace_id)
                data[col_yerr] = yerr

        # Handle different length arrays by padding
        max_len = max(
            len(arr) if hasattr(arr, "__len__") else 1
            for arr in data.values()
            if arr is not None
        )

        for key, value in list(data.items()):
            if value is None:
                continue
            if not hasattr(value, "__len__"):
                data[key] = [value] * max_len
            elif len(value) < max_len:
                data[key] = np.pad(
                    np.asarray(value),
                    (0, max_len - len(value)),
                    mode="constant",
                    constant_values=np.nan,
                )

        return pd.DataFrame(data)

    return pd.DataFrame()


# EOF
