#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-13 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_matshow.py

import numpy as np
import pandas as pd

from scitex.plt.utils._csv_column_naming import get_csv_column_name
from ._format_plot import _parse_tracking_id


def _format_matshow(id, tracked_dict, kwargs):
    """Format data from a matshow call.

    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to matshow

    Returns:
        pd.DataFrame: Formatted data from matshow (flattened matrix with row, col indices)
    """
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse the tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple) and len(args) > 0:
            Z = np.asarray(args[0])

            # Create row/col indices
            rows, cols = np.indices(Z.shape)

            df = pd.DataFrame(
                {
                    get_csv_column_name("row", ax_row, ax_col, trace_id=trace_id): rows.flatten(),
                    get_csv_column_name("col", ax_row, ax_col, trace_id=trace_id): cols.flatten(),
                    get_csv_column_name("value", ax_row, ax_col, trace_id=trace_id): Z.flatten(),
                }
            )
            return df

    return pd.DataFrame()


# EOF
