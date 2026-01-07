#!/usr/bin/env python3
# Timestamp: 2025-12-08
# File: ./src/scitex/plt/utils/_csv_column_naming.py

"""
Single source of truth for CSV column naming in scitex.

This module ensures consistent column naming between:
- CSV export (_export_as_csv)
- JSON metadata (_collect_figure_metadata)
- GUI editors (reading CSV data back)
- figrecipe compatibility

Column naming convention (figrecipe-compatible):
    r{row}c{col}_{caller}-{id}_{var}

Where:
    - r{row}c{col}: axes position in grid (e.g., r0c0, r1c2)
    - {caller}: plotting method name (e.g., plot, scatter, bar)
    - {id}: user-provided id kwarg OR auto-generated per-method counter
    - {var}: variable name (e.g., x, y, bins, heights)

Examples:
    r0c0_plot-0_x        (row 0, col 0, plot method, auto-id 0, x variable)
    r0c0_plot-sine_y     (row 0, col 0, plot method, id "sine", y variable)
    r0c1_scatter-0_x     (row 0, col 1, scatter method, auto-id 0, x)
    r1c0_bar-sales_height (row 1, col 0, bar method, id "sales", height)

Legacy format (still supported for parsing):
    ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
"""

import re

__all__ = [
    "get_csv_column_name",
    "get_csv_column_prefix",
    "parse_csv_column_name",
    "sanitize_id",
    "get_unique_trace_id",
]


def sanitize_id(raw_id: str) -> str:
    """Sanitize ID for use in CSV column names.

    Removes or replaces characters that could cause issues in column names.
    Uses hyphen (-) for word separation within values.

    Parameters
    ----------
    raw_id : str
        Raw identifier (label, id kwarg, or generated)

    Returns
    -------
    str
        Sanitized ID safe for CSV column names

    Examples
    --------
    >>> sanitize_id("sin(x)")
    'sin-x'
    >>> sanitize_id("My Data")
    'my-data'
    >>> sanitize_id("hello_world")
    'hello-world'
    """
    if not raw_id:
        return "0"

    # Replace problematic characters with hyphen
    sanitized = str(raw_id).lower()
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

    return sanitized if sanitized else "0"


# Backward compatibility alias
sanitize_trace_id = sanitize_id


def get_unique_trace_id(trace_id: str, existing_ids: set) -> str:
    """Get unique trace ID, adding suffix if collision detected.

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
    >>> get_unique_trace_id("a", ids)  # collision!
    'a-1'
    """
    base_id = sanitize_id(trace_id)

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
    ax_row: int = 0,
    ax_col: int = 0,
    caller: str = "plot",
    trace_id: str = None,
    trace_index: int = None,
) -> str:
    """Get CSV column prefix for a trace.

    Format: r{row}c{col}_{caller}-{id}_

    Parameters
    ----------
    ax_row : int
        Row position of axes in grid (default: 0)
    ax_col : int
        Column position of axes in grid (default: 0)
    caller : str
        Plotting method name (default: "plot")
    trace_id : str, optional
        User-provided trace identifier. If None, uses trace_index.
    trace_index : int, optional
        Auto-generated index when no trace_id provided (default: 0)

    Returns
    -------
    str
        Column prefix like "r0c0_plot-sine_"

    Examples
    --------
    >>> get_csv_column_prefix(caller="plot", trace_id="sine")
    'r0c0_plot-sine_'
    >>> get_csv_column_prefix(ax_row=1, ax_col=2, caller="scatter", trace_index=0)
    'r1c2_scatter-0_'
    """
    if trace_id:
        safe_id = sanitize_id(trace_id)
    elif trace_index is not None:
        safe_id = str(trace_index)
    else:
        safe_id = "0"

    return f"r{ax_row}c{ax_col}_{caller}-{safe_id}_"


def _extract_caller_and_id(trace_id: str) -> tuple:
    """Extract caller (method) and id from a combined trace_id.

    Handles various formats:
    - "plot_0" -> ("plot", "0")
    - "scatter_1" -> ("scatter", "1")
    - "stx_line_0" -> ("stx_line", "0")
    - "sine" -> ("plot", "sine")  # user-provided, default to "plot"
    - "plot-sine" -> ("plot", "sine")

    Parameters
    ----------
    trace_id : str
        Combined trace identifier

    Returns
    -------
    tuple
        (caller, id)
    """
    if not trace_id:
        return ("plot", "0")

    # Known method prefixes (order matters - longer first)
    known_methods = [
        "stx_line",
        "stx_scatter",
        "stx_bar",
        "stx_barh",
        "stx_box",
        "stx_violin",
        "stx_heatmap",
        "stx_image",
        "stx_imshow",
        "stx_contour",
        "stx_raster",
        "stx_conf_mat",
        "stx_joyplot",
        "stx_rectangle",
        "stx_fillv",
        "stx_kde",
        "stx_ecdf",
        "stx_mean_std",
        "stx_mean_ci",
        "stx_median_iqr",
        "stx_shaded_line",
        "stx_errorbar",
        "stx_fill_between",
        "sns_boxplot",
        "sns_violinplot",
        "sns_barplot",
        "sns_histplot",
        "sns_kdeplot",
        "sns_scatterplot",
        "sns_lineplot",
        "sns_swarmplot",
        "sns_stripplot",
        "sns_heatmap",
        "sns_jointplot",
        "sns_pairplot",
        "plot_scatter",
        "plot_box",
        "plot_imshow",
        "plot_kde",
        "plot",
        "scatter",
        "bar",
        "barh",
        "hist",
        "boxplot",
        "violinplot",
        "errorbar",
        "fill_between",
        "contour",
        "contourf",
        "imshow",
        "pcolormesh",
        "quiver",
        "streamplot",
        "stem",
        "step",
        "pie",
        "hexbin",
        "matshow",
        "eventplot",
        "stackplot",
        "fill",
        "text",
        "annotate",
    ]

    # Check if trace_id starts with a known method
    for method in known_methods:
        # Check with underscore separator (e.g., "plot_0", "stx_line_0")
        if trace_id.startswith(f"{method}_"):
            remainder = trace_id[len(method) + 1 :]
            return (method, remainder if remainder else "0")
        # Check with hyphen separator (e.g., "plot-sine")
        if trace_id.startswith(f"{method}-"):
            remainder = trace_id[len(method) + 1 :]
            return (method, remainder if remainder else "0")

    # No known method prefix - assume user-provided ID, default caller to "plot"
    return ("plot", trace_id)


def get_csv_column_name(
    variable: str,
    ax_row: int = 0,
    ax_col: int = 0,
    caller: str = None,
    trace_id: str = None,
    trace_index: int = None,
) -> str:
    """Get full CSV column name for a data field.

    Format: r{row}c{col}_{caller}-{id}_{var}

    Parameters
    ----------
    variable : str
        Variable name (e.g., "x", "y", "bins", "heights")
    ax_row : int
        Row position of axes in grid (default: 0)
    ax_col : int
        Column position of axes in grid (default: 0)
    caller : str, optional
        Plotting method name. If None, extracted from trace_id or defaults to "plot"
    trace_id : str, optional
        User-provided trace identifier. May contain method prefix (e.g., "plot_0")
    trace_index : int, optional
        Auto-generated index when no trace_id provided

    Returns
    -------
    str
        Full column name like "r0c0_plot-sine_x"

    Examples
    --------
    >>> get_csv_column_name("x", caller="plot", trace_id="sine")
    'r0c0_plot-sine_x'
    >>> get_csv_column_name("y", ax_row=1, ax_col=2, caller="scatter", trace_index=0)
    'r1c2_scatter-0_y'
    >>> get_csv_column_name("x", trace_id="plot_0")  # backward compatible
    'r0c0_plot-0_x'
    """
    # If caller not provided, try to extract from trace_id
    if caller is None and trace_id:
        caller, trace_id = _extract_caller_and_id(trace_id)
    elif caller is None:
        caller = "plot"

    prefix = get_csv_column_prefix(ax_row, ax_col, caller, trace_id, trace_index)
    safe_variable = variable.lower()
    return f"{prefix}{safe_variable}"


def parse_csv_column_name(column_name: str) -> dict:
    """Parse CSV column name to extract components.

    Supports both new format and legacy format:
    - New: r{row}c{col}_{caller}-{id}_{var}
    - Legacy: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}

    Parameters
    ----------
    column_name : str
        Full column name

    Returns
    -------
    dict
        Dictionary with keys:
        - ax_row: int
        - ax_col: int
        - caller: str (method name, empty for legacy format)
        - trace_id: str
        - variable: str
        - valid: bool (True if parsing succeeded)

    Examples
    --------
    >>> parse_csv_column_name("r0c0_plot-sine_x")
    {'ax_row': 0, 'ax_col': 0, 'caller': 'plot', 'trace_id': 'sine', 'variable': 'x', 'valid': True}
    >>> parse_csv_column_name("r1c2_scatter-0_y")
    {'ax_row': 1, 'ax_col': 2, 'caller': 'scatter', 'trace_id': '0', 'variable': 'y', 'valid': True}
    """
    result = {
        "ax_row": 0,
        "ax_col": 0,
        "caller": "",
        "trace_id": "",
        "variable": "",
        "valid": False,
    }

    if not column_name:
        return result

    # Try new format: r{row}c{col}_{caller}-{id}_{var}
    new_pattern = re.compile(r"^r(\d+)c(\d+)_([a-z_]+)-([^_]+)_([a-z]+)$")
    match = new_pattern.match(column_name)
    if match:
        result["ax_row"] = int(match.group(1))
        result["ax_col"] = int(match.group(2))
        result["caller"] = match.group(3)
        result["trace_id"] = match.group(4)
        result["variable"] = match.group(5)
        result["valid"] = True
        return result

    # Try legacy format: ax-row-{row}-col-{col}_trace-id-{id}_variable-{var}
    if column_name.startswith("ax-row-"):
        try:
            parts = column_name.split("_")
            for part in parts:
                if part.startswith("ax-row-"):
                    rest = part[7:]
                    if "-col-" in rest:
                        row_str, col_str = rest.split("-col-")
                        result["ax_row"] = int(row_str)
                        result["ax_col"] = int(col_str)
                elif part.startswith("trace-id-"):
                    result["trace_id"] = part[9:]
                elif part.startswith("variable-"):
                    result["variable"] = part[9:]

            if result["variable"]:
                result["valid"] = True
        except (ValueError, IndexError):
            pass

    return result


def get_trace_columns_from_df(
    df,
    caller: str = None,
    trace_id: str = None,
    trace_index: int = None,
    ax_row: int = 0,
    ax_col: int = 0,
) -> dict:
    """Find CSV columns for a specific trace in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with CSV data
    caller : str, optional
        Plotting method name to filter by
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
        Dictionary mapping variable names to column names
    """
    result = {}

    if caller:
        prefix = get_csv_column_prefix(ax_row, ax_col, caller, trace_id, trace_index)
        for col in df.columns:
            if col.startswith(prefix):
                variable = col[len(prefix) :]
                result[variable] = col
    else:
        # Search all columns matching position and trace
        for col in df.columns:
            parsed = parse_csv_column_name(col)
            if (
                parsed["valid"]
                and parsed["ax_row"] == ax_row
                and parsed["ax_col"] == ax_col
            ):
                if trace_id and parsed["trace_id"] == sanitize_id(trace_id):
                    result[parsed["variable"]] = col
                elif trace_index is not None and parsed["trace_id"] == str(trace_index):
                    result[parsed["variable"]] = col

    return result


# EOF
