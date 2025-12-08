#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/plt/utils/_csv_column_naming.py

"""
Single source of truth for CSV column naming in scitex.

This module ensures consistent column naming between:
- CSV export (_export_as_csv)
- JSON metadata (_collect_figure_metadata)
- GUI editors (reading CSV data back)

Column naming convention:
    ax_{row}{col}_{trace_id}_{data_type}

Where:
    - row, col: axes position in grid (e.g., "00" for single axes)
    - trace_id: unique identifier for the trace (from label, id kwarg, or index)
    - data_type: type of data (e.g., "plot_x", "plot_y", "hist_bins", etc.)
"""

__all__ = [
    'get_csv_column_name',
    'get_csv_column_prefix',
    'parse_csv_column_name',
    'sanitize_trace_id',
]


def sanitize_trace_id(trace_id: str) -> str:
    """Sanitize trace ID for use in CSV column names.

    Removes or replaces characters that could cause issues in column names.

    Parameters
    ----------
    trace_id : str
        Raw trace identifier (label, id kwarg, or generated)

    Returns
    -------
    str
        Sanitized trace ID safe for CSV column names
    """
    if not trace_id:
        return "unnamed"

    # Replace problematic characters
    sanitized = str(trace_id)
    # Keep alphanumeric, underscore, hyphen; replace others with underscore
    result = []
    for char in sanitized:
        if char.isalnum() or char in ('_', '-'):
            result.append(char)
        elif char in (' ', '(', ')', '[', ']', '{', '}', '/', '\\', '.'):
            result.append('_')
        # Skip other characters

    sanitized = ''.join(result)

    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')

    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized if sanitized else "unnamed"


def get_csv_column_prefix(ax_row: int = 0, ax_col: int = 0, trace_id: str = None, trace_index: int = None) -> str:
    """Get CSV column prefix for a trace.

    Parameters
    ----------
    ax_row : int
        Row position of axes in grid (default: 0)
    ax_col : int
        Column position of axes in grid (default: 0)
    trace_id : str, optional
        Trace identifier (from label or id kwarg). If None, uses trace_index.
    trace_index : int, optional
        Index of trace when no trace_id is provided (default: 0)

    Returns
    -------
    str
        Column prefix like "ax_00_sin_x_" or "ax_01_plot_0_"
    """
    ax_pos = f"{ax_row}{ax_col}"

    if trace_id:
        safe_id = sanitize_trace_id(trace_id)
    elif trace_index is not None:
        safe_id = f"plot_{trace_index}"
    else:
        safe_id = "plot_0"

    return f"ax_{ax_pos}_{safe_id}_"


def get_csv_column_name(
    data_type: str,
    ax_row: int = 0,
    ax_col: int = 0,
    trace_id: str = None,
    trace_index: int = None,
) -> str:
    """Get full CSV column name for a data field.

    Parameters
    ----------
    data_type : str
        Type of data (e.g., "plot_x", "plot_y", "hist_bins", "bar_heights")
    ax_row : int
        Row position of axes in grid (default: 0)
    ax_col : int
        Column position of axes in grid (default: 0)
    trace_id : str, optional
        Trace identifier (from label or id kwarg)
    trace_index : int, optional
        Index of trace when no trace_id is provided

    Returns
    -------
    str
        Full column name like "ax_00_sin_x_plot_x" or "ax_01_plot_0_plot_y"

    Examples
    --------
    >>> get_csv_column_name("plot_x", trace_id="sin(x)")
    'ax_00_sin_x_plot_x'
    >>> get_csv_column_name("plot_y", ax_row=1, ax_col=2, trace_index=0)
    'ax_12_plot_0_plot_y'
    """
    prefix = get_csv_column_prefix(ax_row, ax_col, trace_id, trace_index)
    return f"{prefix}{data_type}"


def parse_csv_column_name(column_name: str) -> dict:
    """Parse CSV column name to extract components.

    Parameters
    ----------
    column_name : str
        Full column name (e.g., "ax_00_sin_x_plot_x")

    Returns
    -------
    dict
        Dictionary with keys:
        - ax_row: int
        - ax_col: int
        - trace_id: str
        - data_type: str
        - valid: bool (True if parsing succeeded)

    Examples
    --------
    >>> parse_csv_column_name("ax_00_sin_x_plot_x")
    {'ax_row': 0, 'ax_col': 0, 'trace_id': 'sin_x', 'data_type': 'plot_x', 'valid': True}
    """
    result = {
        'ax_row': 0,
        'ax_col': 0,
        'trace_id': '',
        'data_type': '',
        'valid': False,
    }

    if not column_name or not column_name.startswith('ax_'):
        return result

    parts = column_name.split('_')
    if len(parts) < 4:
        return result

    try:
        # Parse ax position (e.g., "00" from "ax_00_...")
        ax_pos = parts[1]
        if len(ax_pos) >= 2:
            result['ax_row'] = int(ax_pos[0])
            result['ax_col'] = int(ax_pos[1])

        # Last two parts are typically data_type (e.g., "plot_x", "hist_bins")
        # Everything in between is the trace_id
        data_type_parts = parts[-2:]  # e.g., ["plot", "x"]
        result['data_type'] = '_'.join(data_type_parts)

        # Trace ID is everything between ax_pos and data_type
        trace_parts = parts[2:-2]
        result['trace_id'] = '_'.join(trace_parts) if trace_parts else 'plot_0'

        result['valid'] = True

    except (ValueError, IndexError):
        pass

    return result


def get_trace_columns_from_df(df, trace_id: str = None, trace_index: int = None, ax_row: int = 0, ax_col: int = 0) -> dict:
    """Find CSV columns for a specific trace in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with CSV data
    trace_id : str, optional
        Trace identifier to search for
    trace_index : int, optional
        Trace index to search for (if trace_id not provided)
    ax_row : int
        Row position of axes
    ax_col : int
        Column position of axes

    Returns
    -------
    dict
        Dictionary mapping data types to column names, e.g.:
        {'plot_x': 'ax_00_sin_x_plot_x', 'plot_y': 'ax_00_sin_x_plot_y'}
    """
    result = {}
    prefix = get_csv_column_prefix(ax_row, ax_col, trace_id, trace_index)

    for col in df.columns:
        if col.startswith(prefix):
            # Extract data_type from column name
            data_type = col[len(prefix):]
            result[data_type] = col

    return result


# EOF
