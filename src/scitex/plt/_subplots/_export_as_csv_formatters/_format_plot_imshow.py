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
            df = df.copy()
            df.columns = [f"{id}_plot_imshow_{col}" for col in df.columns]

        return df

    # Fallback: return empty DataFrame
    return pd.DataFrame()


# EOF
