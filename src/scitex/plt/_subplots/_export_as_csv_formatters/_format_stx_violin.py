#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-10 12:00:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_stx_violin.py

"""CSV formatter for stx_violin() calls - uses standard column naming."""

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name

from ._format_plot import _parse_tracking_id


def _format_plot_violin(id, tracked_dict, kwargs):
    """Format data from a stx_violin call.

    Formats data in a long-format for better compatibility.
    Uses standard column naming convention:
    (ax-row-{r}-col-{c}_trace-id-{id}_variable-{var}).

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to stx_violin

    Returns:
        pd.DataFrame: Formatted violin plot data in long format
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get standard column names
    group_col = get_csv_column_name("group", ax_row, ax_col, trace_id=trace_id)
    value_col = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)

    # Extract data from tracked_dict
    data = tracked_dict.get("data")

    if data is not None:
        # If data is a simple array or list
        if isinstance(data, (np.ndarray, list)) and not isinstance(
            data[0], (list, np.ndarray, dict)
        ):
            # Convert to long format with group and value columns
            rows = [{group_col: "0", value_col: val} for val in data]
            return pd.DataFrame(rows)

        # If data is a list of arrays (multiple violin plots)
        elif isinstance(data, (list, tuple)) and all(
            isinstance(x, (list, np.ndarray)) for x in data
        ):
            # Get labels if available
            labels = tracked_dict.get("labels")

            # Convert to long format
            rows = []
            for i, values in enumerate(data):
                # Use label if available, otherwise use index
                group = labels[i] if labels and i < len(labels) else f"group{i:02d}"
                for val in values:
                    rows.append({group_col: str(group), value_col: val})

            if rows:
                return pd.DataFrame(rows)

        # If data is a dictionary
        elif isinstance(data, dict):
            # Convert to long format
            rows = []
            for group, values in data.items():
                for val in values:
                    rows.append({group_col: str(group), value_col: val})

            if rows:
                return pd.DataFrame(rows)

        # If data is a DataFrame
        elif isinstance(data, pd.DataFrame):
            # For DataFrame data with x and y columns
            x = tracked_dict.get("x")
            y = tracked_dict.get("y")

            if (
                x is not None
                and y is not None
                and x in data.columns
                and y in data.columns
            ):
                # Convert to long format
                rows = []
                for group_name, group_data in data.groupby(x):
                    for val in group_data[y]:
                        rows.append({group_col: str(group_name), value_col: val})

                if rows:
                    return pd.DataFrame(rows)
            else:
                # For other dataframes, melt to long format
                try:
                    # Try to melt to long format
                    result = pd.melt(data)
                    # Rename columns using standard naming
                    result.columns = [group_col, value_col]
                    return result
                except Exception:
                    # If melt fails, just return empty
                    pass

    return pd.DataFrame()
