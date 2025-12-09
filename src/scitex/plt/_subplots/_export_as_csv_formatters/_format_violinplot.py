#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_violinplot.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_violinplot(id, tracked_dict, kwargs):
    """Format data from a violinplot call.

    Formats data in a long-format for better compatibility.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to violinplot

    Returns:
        pd.DataFrame: Formatted violinplot data in long format
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    args = tracked_dict.get("args", [])

    if len(args) >= 1:
        data = args[0]

        # Handle case when data is a simple array or list
        if isinstance(data, (list, np.ndarray)) and not isinstance(
            data[0], (list, np.ndarray, dict)
        ):
            rows = [{"group": "0", "value": val} for val in data]
            df = pd.DataFrame(rows)
            # Use structured column naming
            col_group = get_csv_column_name("group", ax_row, ax_col, trace_id=trace_id)
            col_value = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)
            df.columns = [col_group, col_value]
            return df

        # Handle case when data is a dictionary
        elif isinstance(data, dict):
            rows = []
            for group, values in data.items():
                for val in values:
                    rows.append({"group": str(group), "value": val})

            if rows:
                df = pd.DataFrame(rows)
                col_group = get_csv_column_name("group", ax_row, ax_col, trace_id=trace_id)
                col_value = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)
                df.columns = [col_group, col_value]
                return df

        # Handle case when data is a list of arrays
        elif isinstance(data, (list, tuple)) and all(
            isinstance(x, (list, np.ndarray)) for x in data
        ):
            rows = []
            for i, values in enumerate(data):
                for val in values:
                    rows.append({"group": str(i), "value": val})

            if rows:
                df = pd.DataFrame(rows)
                col_group = get_csv_column_name("group", ax_row, ax_col, trace_id=trace_id)
                col_value = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)
                df.columns = [col_group, col_value]
                return df

    return pd.DataFrame()


# EOF
