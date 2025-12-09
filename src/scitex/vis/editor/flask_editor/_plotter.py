#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/plotter.py
"""CSV plotting functionality for Flask editor."""

from typing import Dict, Any, Optional
import pandas as pd


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
        csv_cols = trace.get("csv_columns", {})
        x_col = csv_cols.get("x")
        y_col = csv_cols.get("y")

        if x_col in df.columns and y_col in df.columns:
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
