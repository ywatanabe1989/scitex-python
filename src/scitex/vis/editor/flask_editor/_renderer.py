#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/renderer.py
"""Figure rendering for Flask editor - supports single and multi-axis figures."""

from typing import Dict, Any, Tuple, Optional, List
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image
import numpy as np

from ._plotter import plot_from_csv, plot_from_recipe
from ._bbox import extract_bboxes, extract_bboxes_multi

# mm to pt conversion factor
MM_TO_PT = 2.83465


def render_preview_with_bboxes(
    csv_data, overrides: Dict[str, Any], axis_fontsize: int = 7,
    metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render figure and return base64 PNG along with element bounding boxes.

    Args:
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings
        axis_fontsize: Default font size for axis labels
        metadata: Optional JSON metadata (new schema with axes dict)

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size)
    """
    # Check if this is a multi-axis figure (new schema)
    if metadata and "axes" in metadata and isinstance(metadata.get("axes"), dict):
        return render_multi_axis_preview(csv_data, overrides, metadata)

    # Fall back to single-axis rendering
    return render_single_axis_preview(csv_data, overrides, axis_fontsize)


def render_single_axis_preview(
    csv_data, overrides: Dict[str, Any], axis_fontsize: int = 7
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render single-axis figure (legacy mode)."""
    o = overrides

    # Dimensions
    dpi = o.get("dpi", 300)
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
    csv_data, overrides: Dict[str, Any], metadata: Dict[str, Any]
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """Render multi-axis figure from new schema (scitex.plt.figure.recipe).

    Args:
        csv_data: DataFrame containing CSV data
        overrides: Dictionary with override settings
        metadata: JSON metadata with axes dict

    Returns:
        tuple: (base64_image_data, bboxes_dict, image_size)
    """
    o = overrides
    axes_spec = metadata.get("axes", {})
    fig_spec = metadata.get("figure", {})

    # Get grid dimensions from axes positions
    nrows, ncols = _get_grid_dimensions(axes_spec)

    # Figure dimensions
    dpi = fig_spec.get("dpi", o.get("dpi", 300))
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

    # Map axes by their ID
    axes_map = {}
    for ax_id, ax_spec in axes_spec.items():
        pos = ax_spec.get("grid_position", {})
        row = pos.get("row", 0)
        col = pos.get("col", 0)
        ax = axes_array[row, col]
        axes_map[ax_id] = ax

        # Apply background
        if transparent:
            ax.patch.set_facecolor("none")
        elif o.get("facecolor"):
            ax.patch.set_facecolor(o["facecolor"])

        # Plot data from recipe
        if csv_data is not None:
            plot_from_recipe(ax, csv_data, ax_spec, overrides, linewidth_pt, ax_id=ax_id)

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
    """Get grid dimensions from axes specifications."""
    max_row = 0
    max_col = 0
    for ax_id, ax_spec in axes_spec.items():
        pos = ax_spec.get("grid_position", {})
        max_row = max(max_row, pos.get("row", 0))
        max_col = max(max_col, pos.get("col", 0))
    return max_row + 1, max_col + 1


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
    if o.get("title"):
        ax.set_title(o["title"], fontsize=title_fontsize)
    if o.get("xlabel"):
        ax.set_xlabel(o["xlabel"], fontsize=axis_fontsize)
    if o.get("ylabel"):
        ax.set_ylabel(o["ylabel"], fontsize=axis_fontsize)


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


# EOF
