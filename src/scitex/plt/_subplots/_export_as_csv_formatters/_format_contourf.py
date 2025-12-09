#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-09 12:00:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_contourf.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_contourf(id, tracked_dict, kwargs):
    """Format data from a filled contour plot call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to contourf

    Returns:
        pd.DataFrame: Formatted data from contourf (flattened X, Y, Z grids)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple):
            # contourf can be called as:
            # contourf(Z) - Z is 2D
            # contourf(X, Y, Z) - X, Y are 1D or 2D, Z is 2D
            if len(args) == 1:
                Z = np.asarray(args[0])
                X, Y = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
            elif len(args) >= 3:
                X = np.asarray(args[0])
                Y = np.asarray(args[1])
                Z = np.asarray(args[2])
                # If X, Y are 1D, create meshgrid
                if X.ndim == 1 and Y.ndim == 1:
                    X, Y = np.meshgrid(X, Y)
            else:
                return pd.DataFrame()

            # Get column names from single source of truth
            col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
            col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
            col_z = get_csv_column_name("z", ax_row, ax_col, trace_id=trace_id)

            df = pd.DataFrame(
                {col_x: X.flatten(), col_y: Y.flatten(), col_z: Z.flatten()}
            )
            return df

    return pd.DataFrame()


# EOF
