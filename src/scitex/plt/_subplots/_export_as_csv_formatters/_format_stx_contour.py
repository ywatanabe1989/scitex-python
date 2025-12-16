#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV formatter for stx_contour() calls."""

import pandas as pd
from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_stx_contour(id, tracked_dict, kwargs):
    """Format data from stx_contour call for CSV export.

    Parameters
    ----------
    id : str
        Identifier for the plot
    tracked_dict : dict
        Dictionary containing tracked data with 'contour_df'
    kwargs : dict
        Keyword arguments passed to stx_contour

    Returns
    -------
    pd.DataFrame
        Formatted contour data with X, Y, Z columns
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get contour_df from tracked_dict
    contour_df = tracked_dict.get("contour_df")
    if contour_df is not None and isinstance(contour_df, pd.DataFrame):
        result = contour_df.copy()

        # Rename columns using single source of truth
        renamed = {}
        for col in result.columns:
            if col == "X":
                renamed[col] = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
            elif col == "Y":
                renamed[col] = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
            elif col == "Z":
                renamed[col] = get_csv_column_name("z", ax_row, ax_col, trace_id=trace_id)
            else:
                renamed[col] = get_csv_column_name(col.lower(), ax_row, ax_col, trace_id=trace_id)

        return result.rename(columns=renamed)

    return pd.DataFrame()


# EOF
