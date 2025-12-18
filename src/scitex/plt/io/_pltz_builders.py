#!/usr/bin/env python3
# Timestamp: 2025-12-19
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/io/_pltz_builders.py

"""Spec and style builders for pltz bundles."""

import hashlib
from typing import Any, List, Tuple

import numpy as np

from scitex import logging
from scitex.schema import (
    BboxRatio,
    PltzAxesItem,
    PltzAxesLabels,
    PltzAxesLimits,
    PltzDataSource,
    PltzFont,
    PltzLegendSpec,
    PltzSize,
    PltzSpec,
    PltzStyle,
    PltzTheme,
    PltzTraceSpec,
    PltzTraceStyle,
)

logger = logging.getLogger()

__all__ = [
    "build_pltz_spec",
    "build_pltz_style",
    "extract_axes_list",
]


def extract_axes_list(fig) -> List:
    """Extract axes list from figure, handling various wrapper types."""
    try:
        if hasattr(fig.axes, "__iter__") and not isinstance(fig.axes, str):
            return list(fig.axes)
        else:
            return [fig.axes]
    except TypeError:
        return [fig.axes]


def build_pltz_spec(
    fig,
    basename: str,
    csv_df=None,
) -> Tuple["PltzSpec", Any, str, List]:
    """Build PltzSpec from matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to extract spec from.
    basename : str
        Base filename for the bundle.
    csv_df : DataFrame, optional
        Provided data to embed.

    Returns
    -------
    tuple
        (spec, csv_df, csv_hash, axes_list)
    """
    axes_items = []
    traces = []
    extracted_data = {}

    axes_list = extract_axes_list(fig)
    logger.info(f"[build_pltz_spec] axes_list: {axes_list}")

    for ax_idx, ax in enumerate(axes_list):
        bbox = ax.get_position()
        ax_id = f"ax{ax_idx}"

        ax_item = PltzAxesItem(
            id=ax_id,
            bbox=BboxRatio(
                x0=round(bbox.x0, 4),
                y0=round(bbox.y0, 4),
                width=round(bbox.width, 4),
                height=round(bbox.height, 4),
                space="panel",
            ),
            limits=PltzAxesLimits(
                x=list(ax.get_xlim()),
                y=list(ax.get_ylim()),
            ),
            labels=PltzAxesLabels(
                xlabel=ax.get_xlabel() or None,
                ylabel=ax.get_ylabel() or None,
                title=ax.get_title() or None,
            ),
        )
        axes_items.append(ax_item)

        # Extract traces from lines
        for line_idx, line in enumerate(ax.get_lines()):
            label = line.get_label()
            if label is None or label.startswith("_"):
                label = f"series_{line_idx}"

            trace_id = f"{ax_id}-line-{line_idx}"
            xdata, ydata = line.get_data()

            if len(xdata) > 0:
                x_col = f"{ax_id}_trace-{trace_id}_x"
                y_col = f"{ax_id}_trace-{trace_id}_y"
                extracted_data[x_col] = np.array(xdata)
                extracted_data[y_col] = np.array(ydata)

                trace = PltzTraceSpec(
                    id=trace_id,
                    type="line",
                    axes_index=ax_idx,
                    x_col=x_col,
                    y_col=y_col,
                    label=label,
                )
                traces.append(trace)

    # Handle CSV data
    csv_hash = None
    if extracted_data:
        import pandas as pd

        max_len = max(len(v) for v in extracted_data.values())
        padded = {}
        for k, v in extracted_data.items():
            v_float = np.array(v, dtype=float)
            if len(v_float) < max_len:
                padded[k] = np.pad(
                    v_float, (0, max_len - len(v_float)), constant_values=np.nan
                )
            else:
                padded[k] = v_float
        csv_df = pd.DataFrame(padded)
        csv_str = csv_df.to_csv(index=False)
        csv_hash = f"sha256:{hashlib.sha256(csv_str.encode()).hexdigest()[:16]}"
    elif csv_df is not None:
        csv_str = csv_df.to_csv(index=False)
        csv_hash = f"sha256:{hashlib.sha256(csv_str.encode()).hexdigest()[:16]}"

    spec = PltzSpec(
        plot_id=basename,
        data=PltzDataSource(
            csv=f"{basename}.csv",
            format="wide",
            hash=csv_hash,
        ),
        axes=axes_items,
        traces=traces,
    )

    return spec, csv_df, csv_hash, axes_list


def build_pltz_style(
    fig,
    axes_list: List,
    fig_width_inch: float,
    fig_height_inch: float,
) -> "PltzStyle":
    """Build PltzStyle from matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to extract style from.
    axes_list : list
        List of axes from the figure.
    fig_width_inch : float
        Figure width in inches.
    fig_height_inch : float
        Figure height in inches.

    Returns
    -------
    PltzStyle
        The extracted style specification.
    """
    import matplotlib.colors as mcolors

    theme_mode = "light"
    if hasattr(fig, "_scitex_theme"):
        theme_mode = fig._scitex_theme

    trace_styles = []
    for ax_idx, ax in enumerate(axes_list):
        for line_idx, line in enumerate(ax.get_lines()):
            label = line.get_label()
            if label and not label.startswith("_"):
                color = line.get_color()
                if isinstance(color, (list, tuple)):
                    color = mcolors.to_hex(color)

                trace_id = f"ax{ax_idx}-line-{line_idx}"
                trace_styles.append(
                    PltzTraceStyle(
                        trace_id=trace_id,
                        color=color,
                        linewidth=line.get_linewidth(),
                        alpha=line.get_alpha(),
                    )
                )

    legend_spec = _extract_legend_spec(fig, axes_list)

    style = PltzStyle(
        theme=PltzTheme(
            mode=theme_mode,
            colors={
                "background": "transparent",
                "axes_bg": "white" if theme_mode == "light" else "transparent",
                "text": "black" if theme_mode == "light" else "#e8e8e8",
                "spine": "black" if theme_mode == "light" else "#e8e8e8",
                "tick": "black" if theme_mode == "light" else "#e8e8e8",
            },
        ),
        size=PltzSize(
            width_mm=round(fig_width_inch * 25.4, 1),
            height_mm=round(fig_height_inch * 25.4, 1),
        ),
        font=PltzFont(family="sans-serif", size_pt=8.0),
        traces=trace_styles,
        legend=legend_spec,
    )

    return style


def _extract_legend_spec(fig, axes_list: List) -> PltzLegendSpec:
    """Extract legend specification from axes."""
    legend_spec = PltzLegendSpec(visible=True, location="best")

    for ax in axes_list:
        legend = ax.get_legend()
        if legend is not None:
            loc = legend._loc
            loc_map = {
                0: "best",
                1: "upper right",
                2: "upper left",
                3: "lower left",
                4: "lower right",
                5: "right",
                6: "center left",
                7: "center right",
                8: "lower center",
                9: "upper center",
                10: "center",
            }
            if isinstance(loc, int):
                location = loc_map.get(loc, "best")
            else:
                location = str(loc) if loc else "best"

            if location == "best":
                location = _determine_legend_quadrant(fig, ax, legend)

            legend_spec = PltzLegendSpec(
                visible=legend.get_visible(),
                location=location,
                frameon=legend.get_frame_on(),
                fontsize=legend._fontsize if hasattr(legend, "_fontsize") else None,
                ncols=legend._ncols if hasattr(legend, "_ncols") else 1,
                title=legend.get_title().get_text() if legend.get_title() else None,
            )
            break

    return legend_spec


def _determine_legend_quadrant(fig, ax, legend) -> str:
    """Determine legend quadrant from rendered position."""
    try:
        bbox = legend.get_window_extent(fig.canvas.get_renderer())
        ax_bbox = ax.get_position()
        fig_width, fig_height = fig.get_size_inches() * fig.dpi

        legend_center_x = (bbox.x0 + bbox.x1) / 2
        legend_center_y = (bbox.y0 + bbox.y1) / 2
        ax_center_x = (ax_bbox.x0 + ax_bbox.x1) / 2 * fig_width
        ax_center_y = (ax_bbox.y0 + ax_bbox.y1) / 2 * fig_height

        is_right = legend_center_x > ax_center_x
        is_upper = legend_center_y > ax_center_y

        if is_upper and is_right:
            return "upper right"
        elif is_upper and not is_right:
            return "upper left"
        elif not is_upper and is_right:
            return "lower right"
        else:
            return "lower left"
    except Exception:
        return "best"


# EOF
