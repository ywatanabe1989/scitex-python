#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-18 11:40:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot_imshow.py
# ----------------------------------------
import os

__FILE__ = __file__
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd
from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_plot_imshow(id, tracked_dict, kwargs):
    """Format data from a plot_imshow call.

    Args:
        id: Plot identifier
        tracked_dict: Dictionary containing tracked data with key "imshow_df"
        kwargs: Additional keyword arguments

    Returns:
        pd.DataFrame: Formatted image data for CSV export
    """
    # Check for imshow_df in tracked_dict
    if tracked_dict.get("imshow_df") is not None:
        df = tracked_dict["imshow_df"]

        # Add id prefix to column names if id is provided
        if id is not None:
            # Parse tracking ID to extract axes position and trace ID
            ax_row, ax_col, trace_id = _parse_tracking_id(id)

            # Use standardized column naming for each column
            df = df.copy()
            renamed_cols = {}
            for col in df.columns:
                # Create column name like "plot_imshow_row" or "plot_imshow_col"
                renamed_cols[col] = get_csv_column_name(
                    f"plot_imshow_{col}", ax_row, ax_col, trace_id=trace_id
                )
            df.rename(columns=renamed_cols, inplace=True)

        return df

    # Fallback: return empty DataFrame
    return pd.DataFrame()


# EOF
