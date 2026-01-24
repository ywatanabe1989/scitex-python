#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_rendering.py

"""
Figure rendering for DearPyGui editor.

Handles matplotlib figure rendering, element highlights, and hover overlays.
"""

import io
from typing import TYPE_CHECKING, List, Tuple

from ._utils import MM_TO_PT, create_checkerboard

if TYPE_CHECKING:
    from ._state import EditorState


def update_preview(state: "EditorState", dpg) -> None:
    """Update the figure preview (full re-render).

    Parameters
    ----------
    state : EditorState
        Editor state
    dpg : module
        DearPyGui module
    """
    try:
        # Mark cache dirty and do full render
        state.cache_dirty = True
        img_data, width, height = render_figure(state, dpg)

        # Update texture
        dpg.set_value("preview_texture", img_data)

        # Update status
        dpg.set_value("status_text", f"Preview updated ({width}x{height})")

    except Exception as e:
        dpg.set_value("status_text", f"Error: {str(e)}")


def update_hover_overlay(state: "EditorState", dpg) -> None:
    """Fast hover overlay update using cached base image (no matplotlib re-render).

    Parameters
    ----------
    state : EditorState
        Editor state
    dpg : module
        DearPyGui module
    """
    import numpy as np
    from PIL import ImageDraw

    # If no cached base, do full render
    if state.cached_base_image is None:
        update_preview(state, dpg)
        return

    try:
        # Start with a copy of cached base
        img = state.cached_base_image.copy()
        draw = ImageDraw.Draw(img, "RGBA")

        # Get hover element type
        hovered_type = (
            state.hovered_element.get("type") if state.hovered_element else None
        )
        selected_type = (
            state.selected_element.get("type") if state.selected_element else None
        )

        # Draw hover highlight (outline only, no fill) for non-trace elements
        if hovered_type and hovered_type != "trace" and hovered_type != selected_type:
            bbox = state.element_bboxes.get(hovered_type)
            if bbox:
                x0, y0, x1, y1 = bbox
                # Transparent outline only - no fill to avoid covering content
                draw.rectangle(
                    [x0 - 2, y0 - 2, x1 + 2, y1 + 2],
                    fill=None,
                    outline=(100, 180, 255, 100),
                    width=1,
                )

        # Draw selection highlight (outline only, no fill) for non-trace elements
        if selected_type and selected_type != "trace":
            bbox = state.element_bboxes.get(selected_type)
            if bbox:
                x0, y0, x1, y1 = bbox
                # Transparent outline only - no fill to avoid covering content
                draw.rectangle(
                    [x0 - 2, y0 - 2, x1 + 2, y1 + 2],
                    fill=None,
                    outline=(255, 200, 80, 150),
                    width=2,
                )

        # Convert to DearPyGui texture format
        img_array = np.array(img).astype(np.float32) / 255.0
        img_data = img_array.flatten().tolist()

        # Update texture
        dpg.set_value("preview_texture", img_data)

    except Exception:
        # Fallback to full render on error
        update_preview(state, dpg)


def render_figure(state: "EditorState", dpg) -> Tuple[List[float], int, int]:
    """Render figure and return as RGBA data for texture.

    Parameters
    ----------
    state : EditorState
        Editor state
    dpg : module
        DearPyGui module

    Returns
    -------
    tuple
        (img_data, width, height)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator
    from PIL import Image

    from ._plotting import plot_from_csv

    o = state.current_overrides

    # Dimensions - use fixed size for preview
    preview_dpi = 100
    fig_size = o.get("fig_size", [3.15, 2.68])

    # Create figure with white background for preview
    fig, ax = plt.subplots(figsize=fig_size, dpi=preview_dpi)

    # For preview, use white background (transparent doesn't show well in GUI)
    fig.patch.set_facecolor("white")
    ax.patch.set_facecolor("white")

    # Plot from CSV data (only pass selection, hover is via PIL overlay for speed)
    if state.csv_data is not None:
        plot_from_csv(ax, o, state.csv_data, highlight_trace=state.selected_trace_index)
    else:
        ax.text(
            0.5,
            0.5,
            "No plot data available\n(CSV not found)",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=o.get("axis_fontsize", 7),
        )

    # Apply labels
    if o.get("title"):
        ax.set_title(o["title"], fontsize=o.get("title_fontsize", 8))
    if o.get("xlabel"):
        ax.set_xlabel(o["xlabel"], fontsize=o.get("axis_fontsize", 7))
    if o.get("ylabel"):
        ax.set_ylabel(o["ylabel"], fontsize=o.get("axis_fontsize", 7))

    # Tick styling
    ax.tick_params(
        axis="both",
        labelsize=o.get("tick_fontsize", 7),
        length=o.get("tick_length", 0.8) * MM_TO_PT,
        width=o.get("tick_width", 0.2) * MM_TO_PT,
        direction=o.get("tick_direction", "out"),
    )

    # Number of ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))

    # Grid
    if o.get("grid"):
        ax.grid(True, linewidth=o.get("axis_width", 0.2) * MM_TO_PT, alpha=0.3)

    # Axis limits
    if o.get("xlim"):
        ax.set_xlim(o["xlim"])
    if o.get("ylim"):
        ax.set_ylim(o["ylim"])

    # Spines
    if o.get("hide_top_spine", True):
        ax.spines["top"].set_visible(False)
    if o.get("hide_right_spine", True):
        ax.spines["right"].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(o.get("axis_width", 0.2) * MM_TO_PT)

    # Annotations
    for annot in o.get("annotations", []):
        if annot.get("type") == "text":
            ax.text(
                annot.get("x", 0.5),
                annot.get("y", 0.5),
                annot.get("text", ""),
                transform=ax.transAxes,
                fontsize=annot.get("fontsize", o.get("axis_fontsize", 7)),
            )

    fig.tight_layout()

    # Draw before collecting bboxes so we have accurate positions
    fig.canvas.draw()

    # Draw hover/selection highlights for non-trace elements
    draw_element_highlights(state, fig, ax)

    # Store axes transform info for click-to-select
    fig.canvas.draw()
    ax_bbox = ax.get_position()
    fig_width_px = int(fig_size[0] * preview_dpi)
    fig_height_px = int(fig_size[1] * preview_dpi)

    # Collect element bboxes for click detection
    _collect_element_bboxes(state, fig, ax)

    # Convert to RGBA data for DearPyGui texture
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=preview_dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    buf.seek(0)

    # Load with PIL and convert to normalized RGBA
    img = Image.open(buf).convert("RGBA")
    width, height = img.size

    # Resize to fit within max preview size while preserving aspect ratio
    max_width, max_height = 800, 600
    ratio = min(max_width / width, max_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    img = img.resize((new_width, new_height), Image.LANCZOS)

    # Store preview bounds for coordinate conversion (after resize)
    x_offset = (max_width - new_width) // 2
    y_offset = (max_height - new_height) // 2
    state.preview_bounds = (x_offset, y_offset, new_width, new_height)

    # Scale element bboxes to preview coordinates
    _scale_element_bboxes(state, ratio, x_offset, y_offset, new_height)

    # Store axes transform info (scaled to resized image)
    ax_x0 = int(ax_bbox.x0 * new_width)
    ax_y0 = int((1 - ax_bbox.y1) * new_height)  # Flip y (0 at top)
    ax_width = int(ax_bbox.width * new_width)
    ax_height = int(ax_bbox.height * new_height)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    state.axes_transform = (ax_x0, ax_y0, ax_width, ax_height, xlim, ylim)

    # Create background - checkerboard for transparent, white otherwise
    transparent = o.get("transparent", True)
    if transparent:
        padded = create_checkerboard(max_width, max_height, square_size=10)
    else:
        padded = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 255))

    # Paste figure centered on background
    padded.paste(img, (x_offset, y_offset), img)
    img = padded
    width, height = max_width, max_height

    # Cache the base image (without highlights) for fast hover updates
    state.cached_base_image = img.copy()
    state.cache_dirty = False

    # Convert to normalized float array for DearPyGui
    img_array = np.array(img).astype(np.float32) / 255.0
    img_data = img_array.flatten().tolist()

    plt.close(fig)

    # Update texture data
    dpg.set_value("preview_texture", img_data)

    return img_data, width, height


def _collect_element_bboxes(state: "EditorState", fig, ax) -> None:
    """Collect element bboxes for click detection."""
    renderer = fig.canvas.get_renderer()
    state.element_bboxes_raw = {}

    # Title bbox
    if ax.title.get_text():
        try:
            title_bbox = ax.title.get_window_extent(renderer)
            state.element_bboxes_raw["title"] = (
                title_bbox.x0,
                title_bbox.y0,
                title_bbox.x1,
                title_bbox.y1,
            )
        except Exception:
            pass

    # X label bbox
    if ax.xaxis.label.get_text():
        try:
            xlabel_bbox = ax.xaxis.label.get_window_extent(renderer)
            state.element_bboxes_raw["xlabel"] = (
                xlabel_bbox.x0,
                xlabel_bbox.y0,
                xlabel_bbox.x1,
                xlabel_bbox.y1,
            )
        except Exception:
            pass

    # Y label bbox
    if ax.yaxis.label.get_text():
        try:
            ylabel_bbox = ax.yaxis.label.get_window_extent(renderer)
            state.element_bboxes_raw["ylabel"] = (
                ylabel_bbox.x0,
                ylabel_bbox.y0,
                ylabel_bbox.x1,
                ylabel_bbox.y1,
            )
        except Exception:
            pass

    # Legend bbox
    legend = ax.get_legend()
    if legend:
        try:
            legend_bbox = legend.get_window_extent(renderer)
            state.element_bboxes_raw["legend"] = (
                legend_bbox.x0,
                legend_bbox.y0,
                legend_bbox.x1,
                legend_bbox.y1,
            )
        except Exception:
            pass

    # X axis (bottom spine area)
    try:
        xaxis_bbox = ax.spines["bottom"].get_window_extent(renderer)
        state.element_bboxes_raw["xaxis"] = (
            xaxis_bbox.x0,
            xaxis_bbox.y0 - 20,
            xaxis_bbox.x1,
            xaxis_bbox.y1 + 10,
        )
    except Exception:
        pass

    # Y axis (left spine area)
    try:
        yaxis_bbox = ax.spines["left"].get_window_extent(renderer)
        state.element_bboxes_raw["yaxis"] = (
            yaxis_bbox.x0 - 20,
            yaxis_bbox.y0,
            yaxis_bbox.x1 + 10,
            yaxis_bbox.y1,
        )
    except Exception:
        pass


def _scale_element_bboxes(
    state: "EditorState",
    ratio: float,
    x_offset: int,
    y_offset: int,
    new_height: int,
) -> None:
    """Scale element bboxes to preview coordinates."""
    state.element_bboxes = {}
    for elem_type, raw_bbox in state.element_bboxes_raw.items():
        if raw_bbox is None:
            continue
        rx0, ry0, rx1, ry1 = raw_bbox
        # Scale to resized image
        sx0 = int(rx0 * ratio) + x_offset
        sx1 = int(rx1 * ratio) + x_offset
        # Flip Y coordinate (matplotlib origin is bottom, preview is top)
        sy0 = new_height - int(ry1 * ratio) + y_offset
        sy1 = new_height - int(ry0 * ratio) + y_offset
        state.element_bboxes[elem_type] = (sx0, sy0, sx1, sy1)


def draw_element_highlights(state: "EditorState", fig, ax) -> None:
    """Draw selection highlights for non-trace elements."""
    from matplotlib.patches import FancyBboxPatch

    renderer = fig.canvas.get_renderer()

    selected_type = (
        state.selected_element.get("type") if state.selected_element else None
    )

    # Skip if selecting traces (handled separately in plot_from_csv)
    if selected_type == "trace":
        selected_type = None

    def add_highlight_box(text_obj, color, alpha, linewidth=2):
        """Add highlight rectangle around a text object (outline only)."""
        try:
            bbox = text_obj.get_window_extent(renderer)
            fig_bbox = bbox.transformed(fig.transFigure.inverted())
            padding = 0.01
            rect = FancyBboxPatch(
                (fig_bbox.x0 - padding, fig_bbox.y0 - padding),
                fig_bbox.width + 2 * padding,
                fig_bbox.height + 2 * padding,
                boxstyle="round,pad=0.02,rounding_size=0.01",
                facecolor="none",
                edgecolor=color,
                alpha=0.7,
                linewidth=linewidth,
                transform=fig.transFigure,
                zorder=100,
            )
            fig.patches.append(rect)
        except Exception:
            pass

    def add_spine_highlight(spine, color, alpha, linewidth=2):
        """Add highlight to a spine/axis (outline only)."""
        try:
            bbox = spine.get_window_extent(renderer)
            fig_bbox = bbox.transformed(fig.transFigure.inverted())
            padding = 0.01
            rect = FancyBboxPatch(
                (fig_bbox.x0 - padding, fig_bbox.y0 - padding),
                fig_bbox.width + 2 * padding,
                fig_bbox.height + 2 * padding,
                boxstyle="round,pad=0.01",
                facecolor="none",
                edgecolor=color,
                alpha=0.7,
                linewidth=linewidth,
                transform=fig.transFigure,
                zorder=100,
            )
            fig.patches.append(rect)
        except Exception:
            pass

    # Map element types to matplotlib objects
    element_map = {
        "title": ax.title,
        "xlabel": ax.xaxis.label,
        "ylabel": ax.yaxis.label,
    }

    # Draw selection highlight (outline only, no fill)
    select_color = "#FFC850"
    if selected_type in element_map:
        add_highlight_box(element_map[selected_type], select_color, 0.0, linewidth=2)
    elif selected_type == "xaxis":
        add_spine_highlight(ax.spines["bottom"], select_color, 0.0, linewidth=2)
    elif selected_type == "yaxis":
        add_spine_highlight(ax.spines["left"], select_color, 0.0, linewidth=2)
    elif selected_type == "legend":
        legend = ax.get_legend()
        if legend:
            try:
                bbox = legend.get_window_extent(renderer)
                fig_bbox = bbox.transformed(fig.transFigure.inverted())
                padding = 0.01
                rect = FancyBboxPatch(
                    (fig_bbox.x0 - padding, fig_bbox.y0 - padding),
                    fig_bbox.width + 2 * padding,
                    fig_bbox.height + 2 * padding,
                    boxstyle="round,pad=0.02",
                    facecolor="none",
                    edgecolor=select_color,
                    alpha=0.7,
                    linewidth=2,
                    transform=fig.transFigure,
                    zorder=100,
                )
                fig.patches.append(rect)
            except Exception:
                pass


# EOF
