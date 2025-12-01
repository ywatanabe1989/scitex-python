#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 12:20:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/_subplots/_export_as_csv_formatters/_format_contourf.py

import numpy as np
import pandas as pd


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

    if 'args' in tracked_dict:
        args = tracked_dict['args']
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

            # Flatten all arrays
            df = pd.DataFrame({
                f"{id}_contourf_x": X.flatten(),
                f"{id}_contourf_y": Y.flatten(),
                f"{id}_contourf_z": Z.flatten()
            })
            return df

    return pd.DataFrame()


# EOF
