#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV formatter for stx_imshow() calls."""

import numpy as np
import pandas as pd
from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_stx_imshow(id, tracked_dict, kwargs):
    """Format data from stx_imshow call for CSV export.

    Parameters
    ----------
    id : str
        Identifier for the plot
    tracked_dict : dict
        Dictionary containing tracked data with 'imshow_df'
    kwargs : dict
        Keyword arguments passed to stx_imshow

    Returns
    -------
    pd.DataFrame
        Formatted imshow data in row, col, value format (or row, col, R, G, B for RGB)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # Get imshow_df from tracked_dict
    imshow_df = tracked_dict.get("imshow_df")
    if imshow_df is not None and isinstance(imshow_df, pd.DataFrame):
        # Convert from 2D DataFrame format (with col_0, col_1, ... columns)
        # to row, col, value format for easier analysis
        n_rows, n_cols = imshow_df.shape

        # Create row and column indices
        row_indices = np.repeat(np.arange(n_rows), n_cols)
        col_indices = np.tile(np.arange(n_cols), n_rows)

        # Get column names from single source of truth
        col_row = get_csv_column_name("row", ax_row, ax_col, trace_id=trace_id)
        col_col = get_csv_column_name("col", ax_row, ax_col, trace_id=trace_id)
        col_value = get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id)

        # Flatten the DataFrame values
        values = imshow_df.values.flatten()

        result = pd.DataFrame({
            col_row: row_indices,
            col_col: col_indices,
            col_value: values
        })

        return result

    return pd.DataFrame()


# EOF
