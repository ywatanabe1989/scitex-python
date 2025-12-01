#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-01 13:30:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot.py

"""CSV formatter for matplotlib plot() calls."""

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr


def _format_plot(
    id: str,
    tracked_dict: Optional[Dict[str, Any]],
    kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Format data from a plot() call for CSV export.

    Handles various input formats including:
    - Pre-formatted plot_df from scitex wrappers
    - Raw args from __getattr__ proxied matplotlib calls
    - Single array: plot(y) generates x from indices
    - Two arrays: plot(x, y)
    - 2D arrays: creates multiple x/y column pairs

    Parameters
    ----------
    id : str
        Identifier prefix for the output columns (e.g., "ax_00").
    tracked_dict : dict or None
        Dictionary containing tracked data. May include:
        - 'plot_df': Pre-formatted DataFrame from wrapper
        - 'args': Raw positional arguments (x, y) from plot()
    kwargs : dict
        Keyword arguments passed to plot (currently unused).

    Returns
    -------
    pd.DataFrame
        Formatted data with columns prefixed by id.
        For 1D data: {id}_plot_x, {id}_plot_y
        For 2D data: {id}_plot_x00, {id}_plot_y00, {id}_plot_x01, ...
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # For stx_line, we expect a 'plot_df' key
    if 'plot_df' in tracked_dict:
        plot_df = tracked_dict['plot_df']
        if isinstance(plot_df, pd.DataFrame):
            # Add the id prefix to all columns
            return plot_df.add_prefix(f"{id}_")

    # Handle raw args from __getattr__ proxied calls
    if 'args' in tracked_dict:
        args = tracked_dict['args']
        if isinstance(args, tuple) and len(args) > 0:
            # Handle single argument: plot(y) or plot(data_2d)
            if len(args) == 1:
                args_value = args[0]

                # Convert to numpy for consistent handling
                if hasattr(args_value, 'values'):  # pandas Series/DataFrame
                    args_value = args_value.values
                args_value = np.asarray(args_value)

                # 2D array: extract x and y columns
                if hasattr(args_value, 'ndim') and args_value.ndim == 2:
                    x, y = args_value[:, 0], args_value[:, 1]
                    df = pd.DataFrame({f"{id}_plot_x": x, f"{id}_plot_y": y})
                    return df

                # 1D array: generate x from indices (common case: plot(y))
                elif hasattr(args_value, 'ndim') and args_value.ndim == 1:
                    x = np.arange(len(args_value))
                    y = args_value
                    df = pd.DataFrame({f"{id}_plot_x": x, f"{id}_plot_y": y})
                    return df

            # Handle two arguments: plot(x, y)
            elif len(args) >= 2:
                x_arg, y_arg = args[0], args[1]

                # Convert to numpy
                x = np.asarray(x_arg.values if hasattr(x_arg, 'values') else x_arg)
                y = np.asarray(y_arg.values if hasattr(y_arg, 'values') else y_arg)

                # Handle 2D y array (multiple lines)
                if hasattr(y, 'ndim') and y.ndim == 2:
                    out = OrderedDict()
                    for ii in range(y.shape[1]):
                        out[f"{id}_plot_x{ii:02d}"] = x
                        out[f"{id}_plot_y{ii:02d}"] = y[:, ii]
                    df = pd.DataFrame(out)
                    return df

                # Handle DataFrame y
                if isinstance(y_arg, pd.DataFrame):
                    df = pd.DataFrame(
                        {
                            f"{id}_plot_x": x,
                            **{
                                f"{id}_plot_y{ii:02d}": np.array(y_arg[col])
                                for ii, col in enumerate(y_arg.columns)
                            },
                        }
                    )
                    return df

                # Handle 1D arrays (most common case: plot(x, y))
                if hasattr(y, 'ndim') and y.ndim == 1:
                    # Flatten x if needed
                    x_flat = np.ravel(x)
                    y_flat = np.ravel(y)
                    df = pd.DataFrame({f"{id}_plot_x": x_flat, f"{id}_plot_y": y_flat})
                    return df

                # Fallback for list-like y
                df = pd.DataFrame({f"{id}_plot_x": np.ravel(x), f"{id}_plot_y": np.ravel(y)})
                return df

    # Default empty DataFrame if we can't process the input
    return pd.DataFrame()