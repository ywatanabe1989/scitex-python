#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_streamplot.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_streamplot(id, tracked_dict, kwargs):
    """Format data from a streamplot call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to streamplot

    Returns:
        pd.DataFrame: Formatted data from streamplot (X, Y positions and U, V vectors)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse the tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple) and len(args) >= 4:
            # streamplot(X, Y, U, V) - X, Y are 1D, U, V are 2D
            X = np.asarray(args[0])
            Y = np.asarray(args[1])
            U = np.asarray(args[2])
            V = np.asarray(args[3])

            # Create meshgrid if X, Y are 1D
            if X.ndim == 1 and Y.ndim == 1:
                X, Y = np.meshgrid(X, Y)

            df = pd.DataFrame(
                {
                    get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id): X.flatten(),
                    get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id): Y.flatten(),
                    get_csv_column_name("u", ax_row, ax_col, trace_id=trace_id): U.flatten(),
                    get_csv_column_name("v", ax_row, ax_col, trace_id=trace_id): V.flatten(),
                }
            )
            return df

    return pd.DataFrame()


# EOF
