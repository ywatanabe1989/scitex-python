#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/renderer.py
"""Figure rendering for Flask editor - supports single and multi-axis figures."""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image

from scitex.plt.styles import get_default_dpi

from ._bbox import extract_bboxes, extract_bboxes_multi
from ._plotter import plot_from_csv, plot_from_recipe

# mm to pt conversion factor
MM_TO_PT = 2.83465


def render_preview_with_bboxes(
    csv_data,
    overrides: Dict[str, Any],
    axis_fontsize: int = 7,
    metadata: Optional[Dict[str, Any]] = None,
    dark_mode: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render figure and return base64 PNG along with element bounding boxes.

    Args:
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings
        axis_fontsize: Default font size for axis labels
        metadata: Optional JSON metadata (new schema with axes dict)
        dark_mode: Whether to render with dark mode colors (light text/spines)

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size)
    """
    # Check if this is a multi-axis figure (new schema)
    if metadata and "axes" in metadata and isinstance(metadata.get("axes"), dict):
        return render_multi_axis_preview(csv_data, overrides, metadata, dark_mode)

    # Fall back to single-axis rendering
    return render_single_axis_preview(csv_data, overrides, axis_fontsize, dark_mode)


def render_single_axis_preview(
    csv_data,
    overrides: Dict[str, Any],
    axis_fontsize: int = 7,
    dark_mode: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render single-axis figure (legacy mode)."""
    o = overrides

    # Dimensions
    dpi = o.get("dpi", get_default_dpi())
    fig_size = o.get("fig_size", [3.15, 2.68])

    # Font sizes
    axis_fontsize = o.get("axis_fontsize", 7)
    tick_fontsize = o.get("tick_fontsize", 7)
    title_fontsize = o.get("title_fontsize", 8)

    # Line/axis thickness
    linewidth_pt = o.get("linewidth", 0.57)
    axis_width_pt = o.get("axis_width", 0.2) * MM_TO_PT
    tick_length_pt = o.get("tick_length", 0.8) * MM_TO_PT
    tick_width_pt = o.get("tick_width", 0.2) * MM_TO_PT
    tick_direction = o.get("tick_direction", "out")
    x_n_ticks = o.get("x_n_ticks", o.get("n_ticks", 4))
    y_n_ticks = o.get("y_n_ticks", o.get("n_ticks", 4))
    hide_x_ticks = o.get("hide_x_ticks", False)
    hide_y_ticks = o.get("hide_y_ticks", False)

    transparent = o.get("transparent", True)

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    _apply_background(fig, ax, o, transparent)

    # Plot from CSV data
    if csv_data is not None:
        plot_from_csv(ax, csv_data, overrides, linewidth=linewidth_pt)
    else:
        ax.text(
            0.5,
            0.5,
            "No plot data available\n(CSV not found)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=axis_fontsize,
        )

    # Apply labels
    _apply_labels(ax, o, title_fontsize, axis_fontsize)

    # Tick styling
    _apply_tick_styling(
        ax,
        tick_fontsize,
        tick_length_pt,
        tick_width_pt,
        tick_direction,
        x_n_ticks,
        y_n_ticks,
        hide_x_ticks,
        hide_y_ticks,
    )

    # Apply grid, limits, spines
    _apply_style(ax, o, axis_width_pt)

    # Apply annotations
    _apply_annotations(ax, o, axis_fontsize)

    # Apply caption (below figure)
    caption_artist = _apply_caption(fig, o)

    # Apply dark mode styling if requested
    if dark_mode:
        _apply_dark_theme(ax)
        # Also style caption if present
        if caption_artist:
            caption_artist.set_color(DARK_THEME_TEXT_COLOR)

    fig.tight_layout()

    # Get element bounding boxes BEFORE saving (need renderer)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Save to buffer first to get actual image size
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", transparent=transparent
    )
    buf.seek(0)

    # Get actual saved image dimensions
    img = Image.open(buf)
    img_width, img_height = img.size
    buf.seek(0)

    # Get bboxes
    bboxes = extract_bboxes(fig, ax, renderer, img_width, img_height)

    img_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_data, bboxes, {"width": img_width, "height": img_height}


def render_multi_axis_preview(
    csv_data,
    overrides: Dict[str, Any],
    metadata: Dict[str, Any],
    dark_mode: bool = False,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render multi-axis figure from new schema (scitex.plt.figure.recipe).

    Args:
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings
        metadata: JSON metadata with axes dict
        dark_mode: Whether to render with dark mode colors

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size)
    """
    o = overrides
    axes_spec = metadata.get("axes", {})
    fig_spec = metadata.get("figure", {})

    # Get grid dimensions from axes positions
    nrows, ncols = _get_grid_dimensions(axes_spec)

    # Figure dimensions
    dpi = fig_spec.get("dpi", o.get("dpi", get_default_dpi()))
    size_mm = fig_spec.get("size_mm", [176, 106])
    # Convert mm to inches (1 inch = 25.4 mm)
    fig_size = (size_mm[0] / 25.4, size_mm[1] / 25.4)

    # Font sizes (from overrides)
    axis_fontsize = o.get("axis_fontsize", 7)
    tick_fontsize = o.get("tick_fontsize", 7)
    title_fontsize = o.get("title_fontsize", 8)

    # Line/axis thickness
    linewidth_pt = o.get("linewidth", 0.57)
    axis_width_pt = o.get("axis_width", 0.2) * MM_TO_PT
    tick_length_pt = o.get("tick_length", 0.8) * MM_TO_PT
    tick_width_pt = o.get("tick_width", 0.2) * MM_TO_PT
    tick_direction = o.get("tick_direction", "out")
    x_n_ticks = o.get("x_n_ticks", o.get("n_ticks", 4))
    y_n_ticks = o.get("y_n_ticks", o.get("n_ticks", 4))

    transparent = o.get("transparent", True)

    # Create multi-axis figure
    fig, axes_array = plt.subplots(nrows, ncols, figsize=fig_size, dpi=dpi)

    # Handle 1D or 2D array
    if nrows == 1 and ncols == 1:
        axes_array = np.array([[axes_array]])
    elif nrows == 1:
        axes_array = axes_array.reshape(1, -1)
    elif ncols == 1:
        axes_array = axes_array.reshape(-1, 1)

    # Apply background to figure
    if transparent:
        fig.patch.set_facecolor("none")
    elif o.get("facecolor"):
        fig.patch.set_facecolor(o["facecolor"])

    # Build mapping from axis ID to row/col
    ax_to_rowcol = _build_ax_to_rowcol_map(axes_spec)

    # Map axes by their ID
    axes_map = {}
    for ax_id, ax_spec in axes_spec.items():
        row, col = ax_to_rowcol.get(ax_id, (0, 0))
        ax = axes_array[row, col]
        axes_map[ax_id] = ax

        # Apply background
        if transparent:
            ax.patch.set_facecolor("none")
        elif o.get("facecolor"):
            ax.patch.set_facecolor(o["facecolor"])

        # Plot data - check which schema format
        if csv_data is not None:
            calls = ax_spec.get("calls", [])
            if calls:
                # Recipe schema with explicit calls
                plot_from_recipe(
                    ax, csv_data, ax_spec, overrides, linewidth_pt, ax_id=ax_id
                )
            else:
                # Editable schema - plot from CSV column names
                csv_row, csv_col = row, col  # Use computed row/col for CSV lookup
                _plot_from_editable_csv(
                    ax,
                    csv_data,
                    ax_id,
                    csv_row,
                    csv_col,
                    overrides,
                    linewidth_pt,
                    metadata,
                )

        # Get panel-specific overrides (e.g., ax_00_panel)
        panel_key = f"{ax_id}_panel"
        element_overrides = o.get("element_overrides", {})
        panel_overrides = element_overrides.get(panel_key, {})

        # Apply axis labels from spec, with panel overrides taking precedence
        xaxis = ax_spec.get("xaxis", {})
        yaxis = ax_spec.get("yaxis", {})

        # Panel title (from overrides or spec)
        panel_title = panel_overrides.get("title")
        if panel_title:
            ax.set_title(panel_title, fontsize=title_fontsize)

        # X/Y labels (panel overrides take precedence over spec)
        xlabel = panel_overrides.get("xlabel") or xaxis.get("label")
        ylabel = panel_overrides.get("ylabel") or yaxis.get("label")

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=axis_fontsize)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=axis_fontsize)

        # Apply axis limits
        if xaxis.get("lim"):
            ax.set_xlim(xaxis["lim"])
        if yaxis.get("lim"):
            ax.set_ylim(yaxis["lim"])

        # Tick styling
        _apply_tick_styling(
            ax,
            tick_fontsize,
            tick_length_pt,
            tick_width_pt,
            tick_direction,
            x_n_ticks,
            y_n_ticks,
            False,  # hide_x_ticks
            False,  # hide_y_ticks
        )

        # Apply spines
        if o.get("hide_top_spine", True):
            ax.spines["top"].set_visible(False)
        if o.get("hide_right_spine", True):
            ax.spines["right"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_linewidth(axis_width_pt)

        # Apply dark mode to this axis
        if dark_mode:
            _apply_dark_theme(ax)

    # Apply caption (below figure) - use global overrides
    caption_artist = _apply_caption(fig, o)
    if dark_mode and caption_artist:
        caption_artist.set_color(DARK_THEME_TEXT_COLOR)

    fig.tight_layout()

    # Get element bounding boxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Save to buffer
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight", transparent=transparent
    )
    buf.seek(0)

    # Get actual saved image dimensions
    img = Image.open(buf)
    img_width, img_height = img.size
    buf.seek(0)

    # Get bboxes for all axes
    bboxes = extract_bboxes_multi(fig, axes_map, renderer, img_width, img_height)

    img_data = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    return img_data, bboxes, {"width": img_width, "height": img_height}


def _get_grid_dimensions(axes_spec: Dict[str, Any]) -> Tuple[int, int]:
    """Get grid dimensions from axes specifications.

    Handles both:
    - New recipe schema with grid_position
    - Editable schema - infer grid from positions (y-position groups = rows)
    """
    # Check if any axis has grid_position
    has_grid_pos = any(ax_spec.get("grid_position") for ax_spec in axes_spec.values())

    if has_grid_pos:
        max_row = 0
        max_col = 0
        for ax_id, ax_spec in axes_spec.items():
            pos = ax_spec.get("grid_position", {})
            max_row = max(max_row, pos.get("row", 0))
            max_col = max(max_col, pos.get("col", 0))
        return max_row + 1, max_col + 1

    # Editable schema: infer grid from positions
    # Group by y-position to determine rows
    if not axes_spec:
        return 1, 1

    positions = []
    for ax_id, ax_spec in axes_spec.items():
        pos = ax_spec.get("position", [0, 0, 0, 0])
        if len(pos) >= 2:
            positions.append((ax_id, pos[0], pos[1]))  # (id, x, y)

    if not positions:
        return 1, 1

    # Cluster by y-position (tolerance for floating point)
    y_values = sorted(set(round(p[2], 2) for p in positions), reverse=True)
    n_rows = len(y_values)

    # Count columns in first row
    first_row_y = y_values[0]
    n_cols = sum(1 for p in positions if abs(p[2] - first_row_y) < 0.1)

    return n_rows, n_cols


def _build_ax_to_rowcol_map(axes_spec: Dict[str, Any]) -> Dict[str, Tuple[int, int]]:
    """Build mapping from axis ID to (row, col) based on positions."""
    if not axes_spec:
        return {}

    # Check if any axis has grid_position
    has_grid_pos = any(ax_spec.get("grid_position") for ax_spec in axes_spec.values())

    if has_grid_pos:
        result = {}
        for ax_id, ax_spec in axes_spec.items():
            pos = ax_spec.get("grid_position", {})
            result[ax_id] = (pos.get("row", 0), pos.get("col", 0))
        return result

    # Editable schema: compute from positions
    positions = []
    for ax_id, ax_spec in axes_spec.items():
        pos = ax_spec.get("position", [0, 0, 0, 0])
        if len(pos) >= 2:
            positions.append((ax_id, pos[0], pos[1]))  # (id, x, y)

    if not positions:
        return {ax_id: (0, 0) for ax_id in axes_spec}

    # Get unique y-values (rows) - higher y = lower row number (top of figure)
    y_values = sorted(set(round(p[2], 2) for p in positions), reverse=True)
    y_to_row = {y: i for i, y in enumerate(y_values)}

    # For each row, sort by x to get column
    result = {}
    for row_idx, row_y in enumerate(y_values):
        row_axes = [(ax_id, x) for ax_id, x, y in positions if abs(y - row_y) < 0.1]
        row_axes.sort(key=lambda t: t[1])  # Sort by x
        for col_idx, (ax_id, _) in enumerate(row_axes):
            result[ax_id] = (row_idx, col_idx)

    return result


def _get_row_col_from_ax_id(
    ax_id: str, ax_map: Optional[Dict[str, Tuple[int, int]]] = None
) -> Tuple[int, int]:
    """Extract row and col from axis ID using the mapping."""
    if ax_map and ax_id in ax_map:
        return ax_map[ax_id]
    # Fallback: try parsing ax_XY format
    import re

    match = re.match(r"ax_(\d)(\d)", ax_id)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def _plot_from_editable_csv(
    ax,
    csv_data,
    ax_id: str,
    row: int,
    col: int,
    overrides: Dict[str, Any],
    linewidth: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Plot data from editable schema CSV format.

    CSV columns follow pattern: ax-row-X-col-Y_trace-id-NAME_variable-VAR
    """
    import re

    df = csv_data
    elements = metadata.get("elements", {}) if metadata else {}

    # Find columns for this axis (by row/col)
    pattern = f"ax-row-{row}-col-{col}_"
    ax_cols = [c for c in df.columns if c.startswith(pattern)]

    if not ax_cols:
        return

    # Group columns by trace-id
    traces = {}
    for col_name in ax_cols:
        # Parse: ax-row-X-col-Y_trace-id-NAME_variable-VAR
        match = re.match(rf"{pattern}trace-id-(.+?)_variable-(.+)", col_name)
        if match:
            trace_id = match.group(1)
            var_name = match.group(2)
            if trace_id not in traces:
                traces[trace_id] = {}
            traces[trace_id][var_name] = col_name

    # Get element overrides from overrides dict
    element_overrides = overrides.get("element_overrides", {})

    # Plot each trace
    trace_idx = 0
    for trace_id, vars_dict in traces.items():
        # Find element info from metadata
        element_key = f"{ax_id}_line_{trace_idx:02d}"
        element_info = elements.get(element_key, {})
        label = element_info.get("label", trace_id)

        # Get user overrides for this trace
        override_key = f"{ax_id}_trace_{trace_idx}"
        trace_overrides = element_overrides.get(override_key, {})

        # Determine plot type based on variables present
        # Check more specific patterns first, then fall back to x/y

        if (
            "y_lower" in vars_dict
            and "y_middle" in vars_dict
            and "y_upper" in vars_dict
        ):
            # Shaded line plot (fill_between + line)
            x_col = vars_dict.get("x")
            if x_col:
                x = df[x_col].dropna().values
                y_lower = df[vars_dict["y_lower"]].dropna().values
                y_middle = df[vars_dict["y_middle"]].dropna().values
                y_upper = df[vars_dict["y_upper"]].dropna().values

                min_len = min(len(x), len(y_lower), len(y_middle), len(y_upper))
                if min_len > 0:
                    ax.fill_between(
                        x[:min_len], y_lower[:min_len], y_upper[:min_len], alpha=0.3
                    )
                    ax.plot(x[:min_len], y_middle[:min_len], linewidth=linewidth)
            trace_idx += 1

        elif "row" in vars_dict and "col" in vars_dict and "value" in vars_dict:
            # Heatmap / imshow
            rows_data = df[vars_dict["row"]].dropna().values
            cols_data = df[vars_dict["col"]].dropna().values
            values = df[vars_dict["value"]].dropna().values
            if len(rows_data) > 0:
                n_rows = int(rows_data.max()) + 1
                n_cols = int(cols_data.max()) + 1
                data = np.zeros((n_rows, n_cols))
                for r, c, v in zip(
                    rows_data.astype(int), cols_data.astype(int), values
                ):
                    data[r, c] = v
                ax.imshow(data, aspect="auto", origin="lower")

        elif "y1" in vars_dict and "y2" in vars_dict:
            # fill_between (CI band)
            x_col = vars_dict.get("x")
            if x_col:
                x = df[x_col].dropna().values
                y1 = df[vars_dict["y1"]].dropna().values
                y2 = df[vars_dict["y2"]].dropna().values
                min_len = min(len(x), len(y1), len(y2))
                if min_len > 0:
                    ax.fill_between(x[:min_len], y1[:min_len], y2[:min_len], alpha=0.3)

        elif "yerr" in vars_dict and "y" in vars_dict:
            # Error bars with bar chart
            x_col = vars_dict.get("x")
            y_col = vars_dict.get("y")
            if x_col and y_col:
                x = df[x_col].dropna().values
                y = df[y_col].dropna().values
                yerr = df[vars_dict["yerr"]].dropna().values
                min_len = min(len(x), len(y), len(yerr))
                if min_len > 0:
                    ax.bar(x[:min_len], y[:min_len], yerr=yerr[:min_len])

        elif "group" in vars_dict and "value" in vars_dict:
            # Violin/strip plot - plot as scatter for now
            groups = df[vars_dict["group"]].dropna().values
            values = df[vars_dict["value"]].dropna().values
            if len(groups) > 0 and len(values) > 0:
                min_len = min(len(groups), len(values))
                # Convert string groups to numeric positions
                unique_groups = list(dict.fromkeys(groups[:min_len]))  # Preserve order
                group_to_x = {g: i for i, g in enumerate(unique_groups)}
                x_positions = np.array([group_to_x.get(g, 0) for g in groups[:min_len]])
                # Add jitter for strip plot effect
                jitter = np.random.uniform(-0.1, 0.1, min_len)
                ax.scatter(x_positions + jitter, values[:min_len], alpha=0.6, s=20)
                # Set tick labels
                ax.set_xticks(range(len(unique_groups)))
                ax.set_xticklabels(unique_groups, fontsize=6)

        elif "width" in vars_dict and "height" in vars_dict:
            # Rectangle - skip for now
            pass

        elif "type" in vars_dict:
            # Skip type-only entries (like stim markers)
            pass

        elif "content" in vars_dict:
            # Text annotation - skip for preview
            pass

        elif "x" in vars_dict and "y" in vars_dict:
            # Default: line or scatter plot
            x_col = vars_dict["x"]
            y_col = vars_dict["y"]
            x = df[x_col].dropna().values
            y = df[y_col].dropna().values
            if len(x) > 0 and len(y) > 0:
                min_len = min(len(x), len(y))
                # Apply overrides
                color = trace_overrides.get("color")
                lw = trace_overrides.get("linewidth", linewidth)
                ls = trace_overrides.get("linestyle", "-")
                marker = trace_overrides.get("marker")
                alpha = trace_overrides.get("alpha", 1.0)

                kwargs = {"linewidth": lw, "linestyle": ls, "alpha": alpha}
                if color:
                    kwargs["color"] = color
                if marker:
                    kwargs["marker"] = marker
                if label and label != trace_id:
                    kwargs["label"] = label

                # Check if this looks like scatter data (trace-id contains 'scatter' or 'strip')
                if "scatter" in trace_id.lower() or "strip" in trace_id.lower():
                    scatter_kwargs = {"alpha": alpha, "s": 20}
                    if color:
                        scatter_kwargs["c"] = color
                    ax.scatter(x[:min_len], y[:min_len], **scatter_kwargs)
                else:
                    ax.plot(x[:min_len], y[:min_len], **kwargs)
            trace_idx += 1


def _apply_background(fig, ax, o, transparent):
    """Apply background settings to figure."""
    if transparent:
        fig.patch.set_facecolor("none")
        ax.patch.set_facecolor("none")
    elif o.get("facecolor"):
        fig.patch.set_facecolor(o["facecolor"])
        ax.patch.set_facecolor(o["facecolor"])


def _apply_labels(ax, o, title_fontsize, axis_fontsize):
    """Apply title and axis labels."""
    # Show title only if enabled (default True)
    if o.get("show_title", True) and o.get("title"):
        ax.set_title(o["title"], fontsize=title_fontsize)
    if o.get("xlabel"):
        ax.set_xlabel(o["xlabel"], fontsize=axis_fontsize)
    if o.get("ylabel"):
        ax.set_ylabel(o["ylabel"], fontsize=axis_fontsize)


def _apply_caption(fig, o, caption_fontsize=7):
    """Apply caption below the figure."""
    if not o.get("show_caption", False) or not o.get("caption"):
        return None

    caption_text = o.get("caption", "")
    fontsize = o.get("caption_fontsize", caption_fontsize)

    # Place caption below the figure
    # Using fig.text with y position slightly below 0
    caption_artist = fig.text(
        0.5,
        -0.02,  # Centered, below the figure
        caption_text,
        ha="center",
        va="top",
        fontsize=fontsize,
        wrap=True,
        transform=fig.transFigure,
    )
    return caption_artist


def _apply_tick_styling(
    ax,
    tick_fontsize,
    tick_length_pt,
    tick_width_pt,
    tick_direction,
    x_n_ticks,
    y_n_ticks,
    hide_x_ticks,
    hide_y_ticks,
):
    """Apply tick styling to axes."""
    ax.tick_params(
        axis="both",
        labelsize=tick_fontsize,
        length=tick_length_pt,
        width=tick_width_pt,
        direction=tick_direction,
    )

    if hide_x_ticks:
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=x_n_ticks))
    if hide_y_ticks:
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=y_n_ticks))


def _apply_style(ax, o, axis_width_pt):
    """Apply grid, axis limits, and spine settings."""
    if o.get("grid"):
        ax.grid(True, linewidth=axis_width_pt, alpha=0.3)

    if o.get("xlim"):
        ax.set_xlim(o["xlim"])
    if o.get("ylim"):
        ax.set_ylim(o["ylim"])

    if o.get("hide_top_spine", True):
        ax.spines["top"].set_visible(False)
    if o.get("hide_right_spine", True):
        ax.spines["right"].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(axis_width_pt)


def _apply_annotations(ax, o, axis_fontsize):
    """Apply text annotations to figure."""
    for annot in o.get("annotations", []):
        if annot.get("type") == "text":
            ax.text(
                annot.get("x", 0.5),
                annot.get("y", 0.5),
                annot.get("text", ""),
                transform=ax.transAxes,
                fontsize=annot.get("fontsize", axis_fontsize),
            )


def render_panel_preview(
    panel_dir,
    dark_mode: bool = False,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[Dict[str, int]]]:
    """Render a panel from its plot bundle directory with dark mode support.

    Args:
        panel_dir: Path to the .plot panel directory
        dark_mode: Whether to render with dark mode colors

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size) or (None, None, None) on error
    """
    import json
    from pathlib import Path

    import pandas as pd

    panel_dir = Path(panel_dir)

    try:
        # Load spec.json
        spec_path = panel_dir / "spec.json"
        if not spec_path.exists():
            # Try legacy format
            for f in panel_dir.glob("*.json"):
                if f.name != "style.json":
                    spec_path = f
                    break

        if not spec_path.exists():
            return None, None, None

        with open(spec_path, "r") as f:
            metadata = json.load(f)

        # Load CSV data
        csv_data = None
        csv_path = panel_dir / "data.csv"
        if not csv_path.exists():
            for f in panel_dir.glob("*.csv"):
                csv_path = f
                break

        if csv_path.exists():
            csv_data = pd.read_csv(csv_path)

        # Load style.json for overrides
        style_path = panel_dir / "style.json"
        overrides = {}
        if style_path.exists():
            with open(style_path, "r") as f:
                style = json.load(f)
            # Convert style to overrides format
            size = style.get("size", {})
            if size:
                width_mm = size.get("width_mm", 80)
                height_mm = size.get("height_mm", 68)
                overrides["fig_size"] = [width_mm / 25.4, height_mm / 25.4]
            overrides["transparent"] = True

        # Render with dark mode
        return render_preview_with_bboxes(
            csv_data,
            overrides,
            metadata=metadata,
            dark_mode=dark_mode,
        )

    except Exception as e:
        import traceback

        print(f"Error rendering panel {panel_dir}: {e}")
        traceback.print_exc()
        return None, None, None


# Dark mode theme colors
DARK_THEME_TEXT_COLOR = "#e8e8e8"  # Light gray for visibility on dark background
DARK_THEME_SPINE_COLOR = "#e8e8e8"
DARK_THEME_TICK_COLOR = "#e8e8e8"


def _apply_dark_theme(ax):
    """Apply dark mode colors to axes for visibility on dark backgrounds.

    Changes title, labels, tick labels, spines, and legend text to light colors.
    """
    # Title
    title = ax.get_title()
    if title:
        ax.title.set_color(DARK_THEME_TEXT_COLOR)

    # Axis labels
    ax.xaxis.label.set_color(DARK_THEME_TEXT_COLOR)
    ax.yaxis.label.set_color(DARK_THEME_TEXT_COLOR)

    # Tick labels
    ax.tick_params(
        axis="both", colors=DARK_THEME_TICK_COLOR, labelcolor=DARK_THEME_TEXT_COLOR
    )

    # Spines
    for spine in ax.spines.values():
        spine.set_color(DARK_THEME_SPINE_COLOR)

    # Legend (if exists)
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color(DARK_THEME_TEXT_COLOR)
        # Legend title
        legend_title = legend.get_title()
        if legend_title:
            legend_title.set_color(DARK_THEME_TEXT_COLOR)
        # Legend frame (make transparent or dark)
        legend.get_frame().set_facecolor("none")
        legend.get_frame().set_edgecolor(DARK_THEME_SPINE_COLOR)


# EOF
