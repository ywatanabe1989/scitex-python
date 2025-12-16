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
    {domain-name-value}_{domain-name-value}_...

    - Underscore (_) separates different domains
    - Hyphen (-) within domain name and between name-value

Format:
    ax-row-{row}-col-{col}_trace-id-{trace_id}_variable-{variable}

Where:
    - ax-row-{row}-col-{col}: axes position in grid
    - trace-id-{id}: unique identifier for the trace, which can be:
        * User-provided id kwarg (e.g., "sine", "my-data")
        * Generated from label (e.g., "sin-x" from "sin(x)")
        * Auto-generated index (e.g., "0", "1")
    - variable-{var}: type of data variable (e.g., "x", "y", "bins", "heights")

Examples:
    ax-row-0-col-0_trace-id-sine_variable-x       (row 0, col 0, id "sine", x)
    ax-row-0-col-0_trace-id-sine_variable-y       (row 0, col 0, id "sine", y)
    ax-row-0-col-1_trace-id-0_variable-x          (row 0, col 1, auto id 0, x)
    ax-row-1-col-0_trace-id-my-data_variable-bins (row 1, col 0, "my-data", bins)
"""

__all__ = [
    "get_csv_column_name",
    "get_csv_column_prefix",
    "parse_csv_column_name",
    "sanitize_trace_id",
    "get_unique_trace_id",
]


def sanitize_trace_id(trace_id: str) -> str:
    """Sanitize trace ID for use in CSV column names.

    Removes or replaces characters that could cause issues in column names.
    Uses hyphen (-) for word separation within values.

    Parameters
    ----------
    trace_id : str
        Raw trace identifier (label, id kwarg, or generated)

    Returns
    -------
    str
        Sanitized trace ID safe for CSV column names

    Examples
    --------
    >>> sanitize_trace_id("sin(x)")
    'sin-x'
    >>> sanitize_trace_id("My Data")
    'my-data'
    """
    if not trace_id:
        return "unnamed"

    # Replace problematic characters with hyphen (word separator within values)
    sanitized = str(trace_id).lower()
    result = []
    for char in sanitized:
        if char.isalnum():
            result.append(char)
        elif char in (" ", "_", "(", ")", "[", "]", "{", "}", "/", "\\", ".", "-"):
            result.append("-")
        # Skip other characters

    sanitized = "".join(result)

    # Remove consecutive hyphens
    while "--" in sanitized:
        sanitized = sanitized.replace("--", "-")

    # Remove leading/trailing hyphens
    sanitized = sanitized.strip("-")

    return sanitized if sanitized else "unnamed"


def get_unique_trace_id(trace_id: str, existing_ids: set) -> str:
    """Get unique trace ID, adding suffix if collision detected.

    This function ensures trace IDs remain unique even when multiple traces
    have IDs that sanitize to the same value (e.g., "A" and "a" both become "a").
    When a collision is detected, suffixes are added: a -> a-1 -> a-2, etc.

    Parameters
    ----------
    trace_id : str
        Raw trace identifier
    existing_ids : set
        Set of already-used trace IDs (will be modified in-place)

    Returns
    -------
    str
        Unique sanitized trace ID

    Examples
    --------
    >>> ids = set()
    >>> get_unique_trace_id("A", ids)
    'a'
    >>> ids
    {'a'}
    >>> get_unique_trace_id("a", ids)  # collision!
    'a-1'
    >>> ids
    {'a', 'a-1'}
    >>> get_unique_trace_id("A ", ids)  # another collision!
    'a-2'
    >>> ids
    {'a', 'a-1', 'a-2'}
    >>> get_unique_trace_id("B", ids)  # no collision
    'b'
    """
    base_id = sanitize_trace_id(trace_id)

    if base_id not in existing_ids:
        existing_ids.add(base_id)
        return base_id

    # Find unique suffix
    counter = 1
    while f"{base_id}-{counter}" in existing_ids:
        counter += 1

    unique_id = f"{base_id}-{counter}"
    existing_ids.add(unique_id)
    return unique_id


def get_csv_column_prefix(
    ax_row: int = 0, ax_col: int = 0, trace_id: str = None, trace_index: int = None
) -> str:
    """Get CSV column prefix for a trace.

    Format: ax-row-{row}-col-{col}_trace-id-{id}_variable-
    - Underscore (_) separates domains
    - Hyphen (-) within domain names and between name-value

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
        Column prefix like "ax-row-0-col-0_trace-id-sine_variable-"

    Examples
    --------
    >>> get_csv_column_prefix(trace_id="sine")
    'ax-row-0-col-0_trace-id-sine_variable-'
    >>> get_csv_column_prefix(ax_row=1, ax_col=2, trace_index=0)
    'ax-row-1-col-2_trace-id-0_variable-'
    """
    if trace_id:
        safe_id = sanitize_trace_id(trace_id)
    elif trace_index is not None:
        safe_id = str(trace_index)
    else:
        safe_id = "0"

    return f"ax-row-{ax_row}-col-{ax_col}_trace-id-{safe_id}_variable-"


def get_csv_column_name(
    variable: str,
    ax_row: int = 0,
    ax_col: int = 0,
    trace_id: str = None,
    trace_index: int = None,
) -> str:
    """Get full CSV column name for a data field.

    Format: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
    - Underscore (_) separates domains
    - Hyphen (-) within domain names and between name-value

    Parameters
    ----------
    variable : str
        Variable name (e.g., "x", "y", "bins", "heights")
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
        Full column name like "ax-row-0-col-0_trace-id-sine_variable-x"

    Examples
    --------
    >>> get_csv_column_name("x", trace_id="sin(x)")
    'ax-row-0-col-0_trace-id-sin-x_variable-x'
    >>> get_csv_column_name("y", ax_row=1, ax_col=2, trace_index=0)
    'ax-row-1-col-2_trace-id-0_variable-y'
    """
    prefix = get_csv_column_prefix(ax_row, ax_col, trace_id, trace_index)
    # Variable names are simple (x, y, bins, etc.)
    safe_variable = variable.lower()
    return f"{prefix}{safe_variable}"


def parse_csv_column_name(column_name: str) -> dict:
    """Parse CSV column name to extract components.

    Format: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
    - Underscore (_) separates domains
    - Hyphen (-) within domain names and between name-value

    Parameters
    ----------
    column_name : str
        Full column name (e.g., "ax-row-0-col-0_trace-id-sine_variable-x")

    Returns
    -------
    dict
        Dictionary with keys:
        - ax_row: int
        - ax_col: int
        - trace_id: str
        - variable: str
        - valid: bool (True if parsing succeeded)

    Examples
    --------
    >>> parse_csv_column_name("ax-row-0-col-0_trace-id-sine_variable-x")
    {'ax_row': 0, 'ax_col': 0, 'trace_id': 'sine', 'variable': 'x', 'valid': True}
    >>> parse_csv_column_name("ax-row-1-col-2_trace-id-my-data_variable-bins")
    {'ax_row': 1, 'ax_col': 2, 'trace_id': 'my-data', 'variable': 'bins', 'valid': True}
    """
    result = {
        "ax_row": 0,
        "ax_col": 0,
        "trace_id": "",
        "variable": "",
        "valid": False,
    }

    if not column_name or not column_name.startswith("ax-row-"):
        return result

    try:
        # Split by underscore to get domain groups
        parts = column_name.split("_")
        # Expected: ["ax-row-0-col-0", "trace-id-sine", "variable-x"]

        for part in parts:
            if part.startswith("ax-row-"):
                # Parse ax-row-{row}-col-{col}
                # Remove "ax-row-" prefix and split by "-col-"
                rest = part[7:]  # Remove "ax-row-"
                if "-col-" in rest:
                    row_str, col_str = rest.split("-col-")
                    result["ax_row"] = int(row_str)
                    result["ax_col"] = int(col_str)
            elif part.startswith("trace-id-"):
                # Extract trace id (everything after "trace-id-")
                result["trace_id"] = part[9:]
            elif part.startswith("variable-"):
                # Extract variable (everything after "variable-")
                result["variable"] = part[9:]

        # Validate we got all required fields
        if result["variable"]:
            result["valid"] = True

    except (ValueError, IndexError):
        pass

    return result


def get_trace_columns_from_df(
    df, trace_id: str = None, trace_index: int = None, ax_row: int = 0, ax_col: int = 0
) -> dict:
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
        Dictionary mapping variable names to column names, e.g.:
        {'x': 'ax-row-0-col-0_trace-id-sine_variable-x',
         'y': 'ax-row-0-col-0_trace-id-sine_variable-y'}
    """
    result = {}
    prefix = get_csv_column_prefix(ax_row, ax_col, trace_id, trace_index)

    for col in df.columns:
        if col.startswith(prefix):
            # Extract variable from column name
            variable = col[len(prefix):]
            result[variable] = col

    return result


# EOF
