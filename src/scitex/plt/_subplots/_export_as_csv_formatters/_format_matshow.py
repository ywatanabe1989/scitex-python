#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_matshow.py

import numpy as np
import pandas as pd


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

    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple) and len(args) > 0:
            Z = np.asarray(args[0])

            # Create row/col indices
            rows, cols = np.indices(Z.shape)

            df = pd.DataFrame(
                {
                    f"{id}_matshow_row": rows.flatten(),
                    f"{id}_matshow_col": cols.flatten(),
                    f"{id}_matshow_value": Z.flatten(),
                }
            )
            return df

    return pd.DataFrame()


# EOF
