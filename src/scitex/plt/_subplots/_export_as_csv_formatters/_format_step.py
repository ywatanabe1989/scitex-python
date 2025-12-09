#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_step.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_step(id, tracked_dict, kwargs):
    """Format data from a step plot call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to step

    Returns:
        pd.DataFrame: Formatted data from step plot
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple) and len(args) > 0:
            if len(args) == 1:
                y = np.asarray(args[0])
                x = np.arange(len(y))
            elif len(args) >= 2:
                x = np.asarray(args[0])
                y = np.asarray(args[1])
            else:
                return pd.DataFrame()

            # Use structured column naming: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
            col_x = get_csv_column_name("x", ax_row, ax_col, trace_id=trace_id)
            col_y = get_csv_column_name("y", ax_row, ax_col, trace_id=trace_id)
            df = pd.DataFrame({col_x: x, col_y: y})
            return df

    return pd.DataFrame()


# EOF
