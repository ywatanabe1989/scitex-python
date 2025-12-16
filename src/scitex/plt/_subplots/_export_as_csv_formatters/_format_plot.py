#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-08 18:45:00 (ywatanabe)"
# File: ./src/scitex/plt/_subplots/_export_as_csv_formatters/_format_plot.py

"""CSV formatter for matplotlib plot() calls."""

from collections import OrderedDict
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import xarray as xr

from scitex.plt.utils._csv_column_naming import get_csv_column_name


def _parse_tracking_id(id: str, record_index: int = 0) -> tuple:
    """Parse tracking ID to extract axes position and trace ID.

    Parameters
    ----------
    id : str
        Tracking ID like "ax_00_plot_0", "ax_00_stim-box", "plot_0",
        or user-provided like "sine"
    record_index : int
        Index of this record in the history (fallback for trace_id)

    Returns
    -------
    tuple
        (ax_row, ax_col, trace_id)
        trace_id is a string - either the user-provided ID (e.g., "sine")
        or the record_index as string (e.g., "0")

    Note
    ----
    When user provides a custom ID like "sine", that ID is preserved in the
    column names for clarity and traceability.

    Examples
    --------
    >>> _parse_tracking_id("ax_00_plot_0")
    (0, 0, 'plot_0')
    >>> _parse_tracking_id("ax_00_stim-box")
    (0, 0, 'stim-box')
    >>> _parse_tracking_id("ax_12_text_0")
    (1, 2, 'text_0')
    >>> _parse_tracking_id("ax_10_violin")
    (1, 0, 'violin')
    """
    ax_row, ax_col = 0, 0
    trace_id = str(record_index)  # Default to record_index as string

    if id.startswith("ax_"):
        parts = id.split("_")
        if len(parts) >= 2:
            ax_pos = parts[1]
            if len(ax_pos) >= 2:
                try:
                    ax_row = int(ax_pos[0])
                    ax_col = int(ax_pos[1])
                except ValueError:
                    pass
        # Extract trace ID from parts[2:] (everything after "ax_XX_")
        # e.g., "ax_00_stim-box" -> parts = ["ax", "00", "stim-box"] -> trace_id = "stim-box"
        # e.g., "ax_00_plot_0" -> parts = ["ax", "00", "plot", "0"] -> trace_id = "plot_0"
        # e.g., "ax_12_text_0" -> parts = ["ax", "12", "text", "0"] -> trace_id = "text_0"
        if len(parts) >= 3:
            trace_id = "_".join(parts[2:])
    elif id.startswith("plot_"):
        # Extract everything after "plot_" as the trace_id
        trace_id = id[5:] if len(id) > 5 else str(record_index)
    else:
        # User-provided ID like "sine", "cosine" - use it directly
        trace_id = id

    return ax_row, ax_col, trace_id


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
        Identifier prefix for the output columns (e.g., "ax_00_plot_0").
    tracked_dict : dict or None
        Dictionary containing tracked data. May include:
        - 'plot_df': Pre-formatted DataFrame from wrapper
        - 'args': Raw positional arguments (x, y) from plot()
    kwargs : dict
        Keyword arguments passed to plot (currently unused).

    Returns
    -------
    pd.DataFrame
        Formatted data with columns using single source of truth naming.
        Format: ax-row_0_ax-col_0_trace-id_sine_variable_x
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()

    # Parse the tracking ID to get axes position and trace ID
    ax_row, ax_col, trace_id = _parse_tracking_id(id)

    # For stx_line, we expect a 'plot_df' key
    if "plot_df" in tracked_dict:
        plot_df = tracked_dict["plot_df"]
        if isinstance(plot_df, pd.DataFrame):
            # Rename columns using single source of truth
            renamed = {}
            for col in plot_df.columns:
                if col == "plot_x":
                    renamed[col] = get_csv_column_name(
                        "x", ax_row, ax_col, trace_id=trace_id
                    )
                elif col == "plot_y":
                    renamed[col] = get_csv_column_name(
                        "y", ax_row, ax_col, trace_id=trace_id
                    )
                else:
                    # For other columns, use simplified naming
                    renamed[col] = get_csv_column_name(
                        col, ax_row, ax_col, trace_id=trace_id
                    )
            return plot_df.rename(columns=renamed)

    # Handle raw args from __getattr__ proxied calls
    if "args" in tracked_dict:
        args = tracked_dict["args"]
        if isinstance(args, tuple) and len(args) > 0:
            # Get column names from single source of truth
            x_col = get_csv_column_name(
                "x", ax_row, ax_col, trace_id=trace_id
            )
            y_col = get_csv_column_name(
                "y", ax_row, ax_col, trace_id=trace_id
            )

            # Handle single argument: plot(y) or plot(data_2d)
            if len(args) == 1:
                args_value = args[0]

                # Convert to numpy for consistent handling
                if hasattr(args_value, "values"):  # pandas Series/DataFrame
                    args_value = args_value.values
                args_value = np.asarray(args_value)

                # 2D array: extract x and y columns
                if hasattr(args_value, "ndim") and args_value.ndim == 2:
                    x, y = args_value[:, 0], args_value[:, 1]
                    df = pd.DataFrame({x_col: x, y_col: y})
                    return df

                # 1D array: generate x from indices (common case: plot(y))
                elif hasattr(args_value, "ndim") and args_value.ndim == 1:
                    x = np.arange(len(args_value))
                    y = args_value
                    df = pd.DataFrame({x_col: x, y_col: y})
                    return df

            # Handle two arguments: plot(x, y)
            elif len(args) >= 2:
                x_arg, y_arg = args[0], args[1]

                # Convert to numpy
                x = np.asarray(x_arg.values if hasattr(x_arg, "values") else x_arg)
                y = np.asarray(y_arg.values if hasattr(y_arg, "values") else y_arg)

                # Handle 2D y array (multiple lines)
                if hasattr(y, "ndim") and y.ndim == 2:
                    out = OrderedDict()
                    for ii in range(y.shape[1]):
                        x_col_i = get_csv_column_name(
                            f"x{ii:02d}", ax_row, ax_col, trace_id=f"{trace_id}-{ii}"
                        )
                        y_col_i = get_csv_column_name(
                            f"y{ii:02d}", ax_row, ax_col, trace_id=f"{trace_id}-{ii}"
                        )
                        out[x_col_i] = x
                        out[y_col_i] = y[:, ii]
                    df = pd.DataFrame(out)
                    return df

                # Handle DataFrame y
                if isinstance(y_arg, pd.DataFrame):
                    result = {x_col: x}
                    for ii, col in enumerate(y_arg.columns):
                        y_col_i = get_csv_column_name(
                            f"y{ii:02d}", ax_row, ax_col, trace_id=f"{trace_id}-{ii}"
                        )
                        result[y_col_i] = np.array(y_arg[col])
                    df = pd.DataFrame(result)
                    return df

                # Handle 1D arrays (most common case: plot(x, y))
                if hasattr(y, "ndim") and y.ndim == 1:
                    # Flatten x if needed
                    x_flat = np.ravel(x)
                    y_flat = np.ravel(y)
                    df = pd.DataFrame({x_col: x_flat, y_col: y_flat})
                    return df

                # Fallback for list-like y
                df = pd.DataFrame({x_col: np.ravel(x), y_col: np.ravel(y)})
                return df

    # Default empty DataFrame if we can't process the input
    return pd.DataFrame()
