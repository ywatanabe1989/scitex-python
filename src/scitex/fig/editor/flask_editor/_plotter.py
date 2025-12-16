#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/plotter.py
"""CSV plotting functionality for Flask editor."""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np


def _apply_element_overrides(
    kwargs: Dict[str, Any],
    element_key: str,
    element_overrides: Dict[str, Any],
    element_type: str,
) -> Dict[str, Any]:
    """Apply element-specific overrides to kwargs.

    Args:
        kwargs: Original kwargs dict
        element_key: Key to look up in element_overrides (e.g., 'ax_00_scatter_0')
        element_overrides: Dict mapping element keys to their override values
        element_type: Type of element ('trace', 'scatter', 'fill', 'bar')

    Returns:
        Updated kwargs dict with overrides applied
    """
    if not element_key or element_key not in element_overrides:
        return kwargs

    overrides = element_overrides[element_key]
    result = kwargs.copy()

    if element_type == "trace":
        # Line/trace overrides: color, linewidth, linestyle, marker, markersize, alpha
        if "color" in overrides:
            result["color"] = overrides["color"]
        if "linewidth" in overrides:
            result["linewidth"] = overrides["linewidth"]
        if "linestyle" in overrides:
            result["linestyle"] = overrides["linestyle"]
        if "marker" in overrides and overrides["marker"]:
            result["marker"] = overrides["marker"]
        if "markersize" in overrides:
            result["markersize"] = overrides["markersize"]
        if "alpha" in overrides:
            result["alpha"] = overrides["alpha"]
        if "label" in overrides and overrides["label"]:
            result["label"] = overrides["label"]

    elif element_type == "scatter":
        # Scatter overrides: color (c), size (s), marker, alpha, edgecolor
        if "color" in overrides:
            result["c"] = overrides["color"]
            # Remove facecolors if present (conflicts with c)
            result.pop("facecolors", None)
        if "size" in overrides:
            result["s"] = overrides["size"]
        if "marker" in overrides:
            result["marker"] = overrides["marker"]
        if "alpha" in overrides:
            result["alpha"] = overrides["alpha"]
        if "edgecolor" in overrides:
            result["edgecolors"] = overrides["edgecolor"]

    elif element_type == "fill":
        # Fill overrides: color, alpha
        if "color" in overrides:
            result["color"] = overrides["color"]
        if "alpha" in overrides:
            result["alpha"] = overrides["alpha"]

    elif element_type == "bar":
        # Bar overrides: facecolor (color), edgecolor, alpha
        if "facecolor" in overrides:
            result["color"] = overrides["facecolor"]
        if "edgecolor" in overrides:
            result["edgecolor"] = overrides["edgecolor"]
        if "alpha" in overrides:
            result["alpha"] = overrides["alpha"]

    return result


def plot_from_csv(
    ax,
    csv_data: Optional[pd.DataFrame],
    overrides: Dict[str, Any],
    linewidth: float = 1.0,
):
    """Reconstruct plot from CSV data using trace info from overrides.

    Args:
        ax: Matplotlib axes object
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings including traces
        linewidth: Default line width in points
    """
    if csv_data is None or not isinstance(csv_data, pd.DataFrame):
        return

    df = csv_data
    o = overrides

    # Get legend settings from overrides
    legend_fontsize = o.get("legend_fontsize", 6)
    legend_visible = o.get("legend_visible", True)
    legend_frameon = o.get("legend_frameon", False)
    legend_loc = o.get("legend_loc", "best")
    legend_x = o.get("legend_x", 0.5)
    legend_y = o.get("legend_y", 0.5)

    # Get traces from overrides (which may have been edited by user)
    traces = o.get("traces", [])

    if traces:
        _plot_with_traces(
            ax,
            df,
            traces,
            linewidth,
            legend_fontsize,
            legend_visible,
            legend_frameon,
            legend_loc,
            legend_x,
            legend_y,
        )
    else:
        _plot_fallback(
            ax,
            df,
            linewidth,
            legend_fontsize,
            legend_visible,
            legend_frameon,
            legend_loc,
            legend_x,
            legend_y,
        )


def plot_from_recipe(
    ax,
    csv_data: Optional[pd.DataFrame],
    ax_spec: Dict[str, Any],
    overrides: Dict[str, Any],
    linewidth: float = 1.0,
    ax_id: str = "",
):
    """Plot from new recipe schema (scitex.plt.figure.recipe).

    Args:
        ax: Matplotlib axes object
        csv_data: DataFrame containing CSV data
        ax_spec: Axis specification from recipe JSON (includes 'calls' list)
        overrides: Dictionary with override settings
        linewidth: Default line width in points
        ax_id: Axis identifier (e.g., 'ax_00') for element override lookup
    """
    if csv_data is None or not isinstance(csv_data, pd.DataFrame):
        return

    df = csv_data
    calls = ax_spec.get("calls", [])

    # Get element overrides from overrides dict
    element_overrides = overrides.get("element_overrides", {})

    # Track element indices per type for this axis
    element_counts = {"trace": 0, "scatter": 0, "fill": 0, "bar": 0}

    for call in calls:
        method = call.get("method", "")
        data_ref = call.get("data_ref", {})
        kwargs = call.get("kwargs", {}).copy()  # Copy to avoid modifying original
        call_id = call.get("id", "")

        # Build element key for override lookup
        element_key = None

        try:
            if method == "plot":
                element_key = f"{ax_id}_trace_{element_counts['trace']}" if ax_id else f"trace_{element_counts['trace']}"
                kwargs = _apply_element_overrides(kwargs, element_key, element_overrides, "trace")
                _render_plot(ax, df, data_ref, kwargs, linewidth)
                element_counts["trace"] += 1
            elif method == "scatter":
                element_key = f"{ax_id}_scatter_{element_counts['scatter']}" if ax_id else f"scatter_{element_counts['scatter']}"
                kwargs = _apply_element_overrides(kwargs, element_key, element_overrides, "scatter")
                _render_scatter(ax, df, data_ref, kwargs)
                element_counts["scatter"] += 1
            elif method == "bar":
                element_key = f"{ax_id}_bar_{element_counts['bar']}" if ax_id else f"bar_{element_counts['bar']}"
                kwargs = _apply_element_overrides(kwargs, element_key, element_overrides, "bar")
                _render_bar(ax, df, data_ref, kwargs)
                element_counts["bar"] += 1
            elif method == "fill_between":
                element_key = f"{ax_id}_fill_{element_counts['fill']}" if ax_id else f"fill_{element_counts['fill']}"
                kwargs = _apply_element_overrides(kwargs, element_key, element_overrides, "fill")
                _render_fill_between(ax, df, data_ref, kwargs)
                element_counts["fill"] += 1
            elif method == "errorbar":
                _render_errorbar(ax, df, data_ref, kwargs, linewidth)
            elif method == "imshow":
                _render_imshow(ax, df, data_ref, kwargs)
            elif method == "contour":
                _render_contour(ax, df, data_ref, kwargs)
            elif method == "contourf":
                _render_contourf(ax, df, data_ref, kwargs)
            elif method in ("stx_shaded_line", "stx_fillv", "stx_violin",
                           "stx_box", "stx_rectangle", "stx_raster"):
                _render_stx_method(ax, df, method, data_ref, kwargs)
            elif method == "hist":
                _render_hist(ax, df, data_ref, kwargs)
            elif method == "text":
                _render_text(ax, df, data_ref, kwargs)
            else:
                # Try generic approach for unknown methods
                _render_generic(ax, df, method, data_ref, kwargs, linewidth)
        except Exception as e:
            print(f"Error rendering {method} for {call_id}: {e}")

    # Handle legend
    legend_fontsize = overrides.get("legend_fontsize", 6)
    legend_visible = overrides.get("legend_visible", True)
    legend_frameon = overrides.get("legend_frameon", False)

    if legend_visible:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=legend_fontsize, frameon=legend_frameon)


def _get_column_data(df, col_ref: str) -> Optional[np.ndarray]:
    """Get column data from DataFrame, handling missing columns gracefully.

    Handles the naming mismatch between JSON data_ref and CSV columns:
    - JSON: ax-row-0-col-0_trace-id-ax_00_ch1_variable-x
    - CSV:  ax-row-0-col-0_trace-id-ch1_variable-x
    """
    if not col_ref:
        return None

    # Direct match
    if col_ref in df.columns:
        return df[col_ref].dropna().values

    # Try removing ax_XX_ prefix from trace-id portion
    # Pattern: ax-row-X-col-Y_trace-id-ax_XX_NAME_variable-Z
    # Should become: ax-row-X-col-Y_trace-id-NAME_variable-Z
    import re
    simplified = re.sub(r'(trace-id-)ax_\d+_', r'\1', col_ref)
    if simplified in df.columns:
        return df[simplified].dropna().values

    # Try case variations (lowercase the trace-id part)
    simplified_lower = re.sub(r'(trace-id-)[^_]+', lambda m: m.group(0).lower(), simplified)
    if simplified_lower in df.columns:
        return df[simplified_lower].dropna().values

    # Try matching by suffix (variable-x, variable-y, etc.)
    suffix_match = re.search(r'_variable-(\w+)$', col_ref)
    if suffix_match:
        var_suffix = suffix_match.group(0)
        # Extract trace-id pattern
        trace_match = re.search(r'trace-id-(?:ax_\d+_)?([^_]+)', col_ref)
        if trace_match:
            trace_name = trace_match.group(1).lower()
            for col in df.columns:
                if f'trace-id-{trace_name}' in col.lower() and col.endswith(var_suffix):
                    return df[col].dropna().values

    # Last resort: fuzzy match by ending
    for col in df.columns:
        # Match if same variable suffix and similar trace pattern
        if col_ref.split('_variable-')[-1] == col.split('_variable-')[-1]:
            # Check if trace-id portion is similar
            ref_trace = re.search(r'trace-id-(?:ax_\d+_)?(.+?)_variable', col_ref)
            col_trace = re.search(r'trace-id-(.+?)_variable', col)
            if ref_trace and col_trace:
                if ref_trace.group(1).lower() == col_trace.group(1).lower():
                    return df[col].dropna().values

    return None


def _render_plot(ax, df, data_ref, kwargs, linewidth):
    """Render line plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)

    if x is not None and y is not None and len(x) == len(y):
        lw = kwargs.pop("linewidth", linewidth)
        ax.plot(x, y, linewidth=lw, **kwargs)


def _render_scatter(ax, df, data_ref, kwargs):
    """Render scatter plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)

    if x is not None and y is not None and len(x) == len(y):
        ax.scatter(x, y, **kwargs)


def _render_bar(ax, df, data_ref, kwargs):
    """Render bar plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)

    if x is not None and y is not None and len(x) == len(y):
        ax.bar(x, y, **kwargs)


def _render_fill_between(ax, df, data_ref, kwargs):
    """Render fill_between."""
    x_col = data_ref.get("x", "")
    y1_col = data_ref.get("y1", "")
    y2_col = data_ref.get("y2", "")

    x = _get_column_data(df, x_col)
    y1 = _get_column_data(df, y1_col)
    y2 = _get_column_data(df, y2_col)

    if x is not None and y1 is not None and y2 is not None:
        min_len = min(len(x), len(y1), len(y2))
        ax.fill_between(x[:min_len], y1[:min_len], y2[:min_len], **kwargs)


def _render_errorbar(ax, df, data_ref, kwargs, linewidth):
    """Render errorbar plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")
    yerr_col = data_ref.get("yerr", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)
    yerr = _get_column_data(df, yerr_col) if yerr_col else None

    if x is not None and y is not None and len(x) == len(y):
        ax.errorbar(x, y, yerr=yerr, linewidth=linewidth, **kwargs)


def _render_imshow(ax, df, data_ref, kwargs):
    """Render imshow (heatmap)."""
    data_col = data_ref.get("data", "")
    # For imshow, data is typically in a special format - try to reconstruct
    # Look for row/col/value columns
    row_col = data_ref.get("row", "")
    col_col = data_ref.get("col", "")
    value_col = data_ref.get("value", "")

    if row_col and col_col and value_col:
        rows = _get_column_data(df, row_col)
        cols = _get_column_data(df, col_col)
        values = _get_column_data(df, value_col)

        if rows is not None and cols is not None and values is not None:
            # Reconstruct 2D array
            n_rows = int(rows.max()) + 1
            n_cols = int(cols.max()) + 1
            data = np.zeros((n_rows, n_cols))
            for r, c, v in zip(rows.astype(int), cols.astype(int), values):
                data[r, c] = v
            ax.imshow(data, **kwargs)


def _render_contour(ax, df, data_ref, kwargs):
    """Render contour plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")
    z_col = data_ref.get("z", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)
    z = _get_column_data(df, z_col)

    if x is not None and y is not None and z is not None:
        # Assume data is on a grid - reconstruct
        n = int(np.sqrt(len(x)))
        if n * n == len(x):
            X = x.reshape(n, n)
            Y = y.reshape(n, n)
            Z = z.reshape(n, n)
            ax.contour(X, Y, Z, **kwargs)


def _render_contourf(ax, df, data_ref, kwargs):
    """Render filled contour plot."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")
    z_col = data_ref.get("z", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)
    z = _get_column_data(df, z_col)

    if x is not None and y is not None and z is not None:
        # Assume data is on a grid - reconstruct
        n = int(np.sqrt(len(x)))
        if n * n == len(x):
            X = x.reshape(n, n)
            Y = y.reshape(n, n)
            Z = z.reshape(n, n)
            ax.contourf(X, Y, Z, **kwargs)


def _render_stx_method(ax, df, method, data_ref, kwargs):
    """Render scitex-specific methods (shaded_line, fillv, etc.)."""
    # These are custom methods - for now, skip or implement basic versions
    if method == "stx_shaded_line":
        x_col = data_ref.get("x", "")
        y_lower_col = data_ref.get("y_lower", "")
        y_middle_col = data_ref.get("y_middle", "")
        y_upper_col = data_ref.get("y_upper", "")

        x = _get_column_data(df, x_col)
        y_lower = _get_column_data(df, y_lower_col)
        y_middle = _get_column_data(df, y_middle_col)
        y_upper = _get_column_data(df, y_upper_col)

        if all(v is not None for v in [x, y_lower, y_middle, y_upper]):
            min_len = min(len(x), len(y_lower), len(y_middle), len(y_upper))
            ax.fill_between(x[:min_len], y_lower[:min_len], y_upper[:min_len],
                           alpha=0.3, **{k: v for k, v in kwargs.items()
                                        if k not in ['linewidth']})
            ax.plot(x[:min_len], y_middle[:min_len], **kwargs)


def _render_hist(ax, df, data_ref, kwargs):
    """Render histogram."""
    data_col = data_ref.get("data", data_ref.get("x", ""))
    data = _get_column_data(df, data_col)
    if data is not None:
        ax.hist(data, **kwargs)


def _render_text(ax, df, data_ref, kwargs):
    """Render text annotation."""
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")
    content_col = data_ref.get("content", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)

    if x is not None and y is not None and len(x) > 0:
        # Get text content from CSV or kwargs
        text = kwargs.pop("text", "")
        if not text and content_col:
            content = _get_column_data(df, content_col)
            if content is not None and len(content) > 0:
                text = str(content[0])
        ax.text(x[0], y[0], text, **kwargs)


def _render_generic(ax, df, method, data_ref, kwargs, linewidth):
    """Try to render using generic approach."""
    # For unknown methods, try to get x/y data and plot as line
    x_col = data_ref.get("x", "")
    y_col = data_ref.get("y", "")

    x = _get_column_data(df, x_col)
    y = _get_column_data(df, y_col)

    if x is not None and y is not None and len(x) == len(y):
        # Filter out kwargs that are not valid for ax.plot()
        invalid_plot_kwargs = {
            'levels', 'extend', 'origin', 'extent', 'aspect',
            'norm', 'vmin', 'vmax', 'interpolation', 'filternorm',
            'filterrad', 'resample', 'bins', 'range', 'density',
            'weights', 'cumulative', 'bottom', 'histtype', 'align',
            'orientation', 'rwidth', 'log', 'stacked', 'data',
            'width', 'height', 'edgecolors', 's', 'c', 'facecolors',
        }
        filtered_kwargs = {k: v for k, v in kwargs.items()
                          if k not in invalid_plot_kwargs}
        ax.plot(x, y, linewidth=linewidth, **filtered_kwargs)


def _plot_with_traces(
    ax,
    df,
    traces,
    linewidth,
    legend_fontsize,
    legend_visible,
    legend_frameon,
    legend_loc,
    legend_x,
    legend_y,
):
    """Plot using trace information from overrides."""
    for trace in traces:
        # Support both old format (csv_columns.x/y) and new format (x_col/y_col)
        csv_cols = trace.get("csv_columns", {})
        x_col = csv_cols.get("x") or trace.get("x_col")
        y_col = csv_cols.get("y") or trace.get("y_col")

        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            ax.plot(
                df[x_col],
                df[y_col],
                label=trace.get("label", trace.get("id", "")),
                color=trace.get("color"),
                linestyle=trace.get("linestyle", "-"),
                linewidth=trace.get("linewidth", linewidth),
                marker=trace.get("marker", None),
                markersize=trace.get("markersize", 6),
            )

    # Add legend if there are labeled traces
    if legend_visible and any(t.get("label") for t in traces):
        _add_legend(ax, legend_fontsize, legend_frameon, legend_loc, legend_x, legend_y)


def _plot_fallback(
    ax,
    df,
    linewidth,
    legend_fontsize,
    legend_visible,
    legend_frameon,
    legend_loc,
    legend_x,
    legend_y,
):
    """Fallback plotting when no trace info available - smart parsing of column names."""
    cols = df.columns.tolist()

    # Group columns by trace ID
    trace_groups = {}
    for col in cols:
        if col.endswith("_x"):
            trace_id = col[:-2]  # Remove '_x'
            y_col = trace_id + "_y"
            if y_col in cols:
                # Extract label from column name (e.g., ax_00_sine_plot -> sine)
                parts = trace_id.split("_")
                label = parts[2] if len(parts) > 2 else trace_id
                trace_groups[trace_id] = {
                    "x_col": col,
                    "y_col": y_col,
                    "label": label,
                }

    if trace_groups:
        for trace_id, info in trace_groups.items():
            ax.plot(
                df[info["x_col"]],
                df[info["y_col"]],
                label=info["label"],
                linewidth=linewidth,
            )
        if legend_visible:
            _add_legend(
                ax, legend_fontsize, legend_frameon, legend_loc, legend_x, legend_y
            )

    elif len(cols) >= 2:
        # Last resort: assume first column is x, rest are y
        x_col = cols[0]
        for y_col in cols[1:]:
            try:
                ax.plot(df[x_col], df[y_col], label=str(y_col), linewidth=linewidth)
            except Exception:
                pass
        if len(cols) > 2 and legend_visible:
            _add_legend(
                ax, legend_fontsize, legend_frameon, legend_loc, legend_x, legend_y
            )


def _add_legend(ax, fontsize, frameon, loc, x, y):
    """Add legend to axes with specified settings."""
    if loc == "custom":
        ax.legend(
            fontsize=fontsize,
            frameon=frameon,
            loc="upper left",
            bbox_to_anchor=(x, y),
        )
    else:
        ax.legend(
            fontsize=fontsize,
            frameon=frameon,
            loc=loc,
        )


# EOF
