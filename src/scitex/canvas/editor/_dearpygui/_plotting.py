#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_plotting.py

"""
CSV data plotting for DearPyGui editor.

Handles reconstructing plots from CSV data with trace info.
"""

from typing import Any, Dict, Optional

import pandas as pd


def plot_from_csv(
    ax,
    overrides: Dict[str, Any],
    csv_data: pd.DataFrame,
    highlight_trace: Optional[int] = None,
    hover_trace: Optional[int] = None,
) -> None:
    """Reconstruct plot from CSV data using trace info.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on
    overrides : dict
        Current overrides containing trace info
    csv_data : pd.DataFrame
        CSV data to plot
    highlight_trace : int, optional
        Index of trace to highlight with selection effect (yellow glow)
    hover_trace : int, optional
        Index of trace to highlight with hover effect (cyan glow)
    """
    from .._defaults import _normalize_legend_loc

    if not isinstance(csv_data, pd.DataFrame):
        return

    df = csv_data
    linewidth = overrides.get("linewidth", 1.0)
    legend_visible = overrides.get("legend_visible", True)
    legend_fontsize = overrides.get("legend_fontsize", 6)
    legend_frameon = overrides.get("legend_frameon", False)
    legend_loc = _normalize_legend_loc(overrides.get("legend_loc", "best"))

    traces = overrides.get("traces", [])

    if traces:
        _plot_with_traces(
            ax,
            df,
            traces,
            linewidth,
            highlight_trace,
            hover_trace,
            legend_visible,
            legend_fontsize,
            legend_frameon,
            legend_loc,
        )
    else:
        _plot_fallback(
            ax,
            df,
            linewidth,
            legend_visible,
            legend_fontsize,
            legend_frameon,
            legend_loc,
        )


def _plot_with_traces(
    ax,
    df: pd.DataFrame,
    traces: list,
    linewidth: float,
    highlight_trace: Optional[int],
    hover_trace: Optional[int],
    legend_visible: bool,
    legend_fontsize: int,
    legend_frameon: bool,
    legend_loc: str,
) -> None:
    """Plot using trace definitions."""
    for i, trace in enumerate(traces):
        csv_cols = trace.get("csv_columns", {})
        x_col = csv_cols.get("x")
        y_col = csv_cols.get("y")

        if x_col in df.columns and y_col in df.columns:
            trace_linewidth = trace.get("linewidth", linewidth)
            is_selected = highlight_trace is not None and i == highlight_trace
            is_hovered = (
                hover_trace is not None and i == hover_trace and not is_selected
            )

            # Draw selection glow (yellow, stronger)
            if is_selected:
                ax.plot(
                    df[x_col],
                    df[y_col],
                    color="yellow",
                    linewidth=trace_linewidth * 4,
                    alpha=0.5,
                    zorder=0,
                )
            # Draw hover glow (cyan, subtler)
            elif is_hovered:
                ax.plot(
                    df[x_col],
                    df[y_col],
                    color="cyan",
                    linewidth=trace_linewidth * 3,
                    alpha=0.3,
                    zorder=0,
                )

            ax.plot(
                df[x_col],
                df[y_col],
                label=trace.get("label", trace.get("id", "")),
                color=trace.get("color"),
                linestyle=trace.get("linestyle", "-"),
                linewidth=trace_linewidth
                * (1.5 if is_selected else (1.2 if is_hovered else 1.0)),
                marker=trace.get("marker", None),
                markersize=trace.get("markersize", 6),
                zorder=10 if is_selected else (5 if is_hovered else 1),
            )

    if legend_visible and any(t.get("label") for t in traces):
        ax.legend(fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc)


def _plot_fallback(
    ax,
    df: pd.DataFrame,
    linewidth: float,
    legend_visible: bool,
    legend_fontsize: int,
    legend_frameon: bool,
    legend_loc: str,
) -> None:
    """Fallback plotting when no traces defined - parse column names."""
    cols = df.columns.tolist()
    trace_groups = {}

    for col in cols:
        if col.endswith("_x"):
            trace_id = col[:-2]
            y_col = trace_id + "_y"
            if y_col in cols:
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
            ax.legend(fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc)
    elif len(cols) >= 2:
        x_col = cols[0]
        for y_col in cols[1:]:
            try:
                ax.plot(df[x_col], df[y_col], label=str(y_col), linewidth=linewidth)
            except Exception:
                pass
        if len(cols) > 2 and legend_visible:
            ax.legend(fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc)


# EOF
