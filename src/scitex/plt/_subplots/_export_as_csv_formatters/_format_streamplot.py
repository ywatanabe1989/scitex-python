#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_streamplot.py

import numpy as np
import pandas as pd


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
                    f"{id}_streamplot_x": X.flatten(),
                    f"{id}_streamplot_y": Y.flatten(),
                    f"{id}_streamplot_u": U.flatten(),
                    f"{id}_streamplot_v": V.flatten(),
                }
            )
            return df

    return pd.DataFrame()


# EOF
