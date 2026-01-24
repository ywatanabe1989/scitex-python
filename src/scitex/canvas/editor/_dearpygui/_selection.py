#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_selection.py

"""
Element selection for DearPyGui editor.

Handles click-to-select, hover detection, and element finding.
"""

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ._state import EditorState


def get_trace_labels(state: "EditorState") -> List[str]:
    """Get list of trace labels for selection combo."""
    traces = state.current_overrides.get("traces", [])
    if not traces:
        return ["(no traces)"]
    return [t.get("label", t.get("id", f"Trace {i}")) for i, t in enumerate(traces)]


def get_all_element_labels(state: "EditorState") -> List[str]:
    """Get list of all selectable element labels."""
    labels = []

    # Fixed elements
    labels.append("Title")
    labels.append("X Label")
    labels.append("Y Label")
    labels.append("X Axis")
    labels.append("Y Axis")
    labels.append("Legend")

    # Traces
    traces = state.current_overrides.get("traces", [])
    for i, t in enumerate(traces):
        label = t.get("label", t.get("id", f"Trace {i}"))
        labels.append(f"Trace: {label}")

    return labels


def find_clicked_element(
    state: "EditorState", click_x: float, click_y: float
) -> Optional[Dict]:
    """Find which element was clicked based on stored bboxes."""
    if not state.element_bboxes:
        return None

    # Check each element bbox
    for element_type, bbox in state.element_bboxes.items():
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        if x0 <= click_x <= x1 and y0 <= click_y <= y1:
            return {"type": element_type, "index": None}

    return None


def find_nearest_trace(
    state: "EditorState",
    click_x: float,
    click_y: float,
    preview_width: int,
    preview_height: int,
) -> Optional[int]:
    """Find the nearest trace to the click position."""
    if state.csv_data is None or not isinstance(state.csv_data, pd.DataFrame):
        return None

    traces = state.current_overrides.get("traces", [])
    if not traces:
        return None

    # Get preview bounds from last render
    if state.preview_bounds is None:
        return None

    x_offset, y_offset, fig_width, fig_height = state.preview_bounds

    # Adjust click coordinates to figure space
    fig_x = click_x - x_offset
    fig_y = click_y - y_offset

    # Check if click is within figure bounds
    if not (0 <= fig_x <= fig_width and 0 <= fig_y <= fig_height):
        return None

    # Get axes transform info
    if state.axes_transform is None:
        return None

    ax_x0, ax_y0, ax_width, ax_height, xlim, ylim = state.axes_transform

    # Convert figure pixel to axes pixel
    ax_pixel_x = fig_x - ax_x0
    ax_pixel_y = fig_y - ax_y0

    # Check if click is within axes bounds
    if not (0 <= ax_pixel_x <= ax_width and 0 <= ax_pixel_y <= ax_height):
        return None

    # Convert axes pixel to data coordinates
    # Note: y is flipped (0 at top in pixel space)
    data_x = xlim[0] + (ax_pixel_x / ax_width) * (xlim[1] - xlim[0])
    data_y = ylim[1] - (ax_pixel_y / ax_height) * (ylim[1] - ylim[0])

    # Find nearest trace
    df = state.csv_data
    min_dist = float("inf")
    nearest_idx = None

    for i, trace in enumerate(traces):
        csv_cols = trace.get("csv_columns", {})
        x_col = csv_cols.get("x")
        y_col = csv_cols.get("y")

        if x_col not in df.columns or y_col not in df.columns:
            continue

        trace_x = df[x_col].dropna().values
        trace_y = df[y_col].dropna().values

        if len(trace_x) == 0:
            continue

        # Normalize coordinates for distance calculation
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        norm_click_x = (data_x - xlim[0]) / x_range if x_range > 0 else 0
        norm_click_y = (data_y - ylim[0]) / y_range if y_range > 0 else 0

        norm_trace_x = (trace_x - xlim[0]) / x_range if x_range > 0 else trace_x
        norm_trace_y = (trace_y - ylim[0]) / y_range if y_range > 0 else trace_y

        # Calculate distances to all points
        distances = np.sqrt(
            (norm_trace_x - norm_click_x) ** 2 + (norm_trace_y - norm_click_y) ** 2
        )
        min_trace_dist = np.min(distances)

        if min_trace_dist < min_dist:
            min_dist = min_trace_dist
            nearest_idx = i

    # Only select if close enough (threshold in normalized space)
    if min_dist < 0.1:  # 10% of plot area
        return nearest_idx

    return None


def select_element(state: "EditorState", element: Dict, dpg) -> None:
    """Select an element and show appropriate controls."""
    from ._rendering import update_preview

    state.selected_element = element
    elem_type = element.get("type")
    elem_idx = element.get("index")

    # Hide all control groups first
    dpg.configure_item("trace_controls_group", show=False)
    dpg.configure_item("text_controls_group", show=False)
    dpg.configure_item("axis_controls_group", show=False)
    dpg.configure_item("legend_controls_group", show=False)

    # Update combo selection
    if elem_type == "trace":
        _select_trace(state, elem_idx, dpg)
    elif elem_type in ("title", "xlabel", "ylabel"):
        _select_text_element(state, elem_type, dpg)
    elif elem_type in ("xaxis", "yaxis"):
        _select_axis_element(state, elem_type, dpg)
    elif elem_type == "legend":
        _select_legend(state, dpg)

    # Redraw with highlight
    update_preview(state, dpg)


def _select_trace(state: "EditorState", trace_idx: Optional[int], dpg) -> None:
    """Handle trace selection."""
    traces = state.current_overrides.get("traces", [])
    if trace_idx is not None and trace_idx < len(traces):
        trace = traces[trace_idx]
        label = f"Trace: {trace.get('label', trace.get('id', f'Trace {trace_idx}'))}"
        dpg.set_value("element_selector_combo", label)

        # Show trace controls and populate
        dpg.configure_item("trace_controls_group", show=True)
        state.selected_trace_index = trace_idx
        dpg.set_value("trace_label_input", trace.get("label", ""))

        color_hex = trace.get("color", "#0080bf")
        try:
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)
            dpg.set_value("trace_color_picker", [r, g, b])
        except (ValueError, IndexError):
            dpg.set_value("trace_color_picker", [128, 128, 191])

        dpg.set_value("trace_linewidth_slider", trace.get("linewidth", 1.0))
        dpg.set_value("trace_linestyle_combo", trace.get("linestyle", "-"))
        dpg.set_value("trace_marker_combo", trace.get("marker", "") or "")
        dpg.set_value("trace_markersize_slider", trace.get("markersize", 6.0))

        dpg.set_value(
            "selection_text",
            f"Selected: {trace.get('label', f'Trace {trace_idx}')}",
        )


def _select_text_element(state: "EditorState", elem_type: str, dpg) -> None:
    """Handle text element selection (title, xlabel, ylabel)."""
    dpg.set_value(
        "element_selector_combo",
        elem_type.replace("x", "X ").replace("y", "Y ").title(),
    )
    dpg.configure_item("text_controls_group", show=True)

    o = state.current_overrides
    if elem_type == "title":
        dpg.set_value("element_text_input", o.get("title", ""))
        dpg.set_value("element_fontsize_slider", o.get("title_fontsize", 8))
    elif elem_type == "xlabel":
        dpg.set_value("element_text_input", o.get("xlabel", ""))
        dpg.set_value("element_fontsize_slider", o.get("axis_fontsize", 7))
    elif elem_type == "ylabel":
        dpg.set_value("element_text_input", o.get("ylabel", ""))
        dpg.set_value("element_fontsize_slider", o.get("axis_fontsize", 7))

    dpg.set_value("selection_text", f"Selected: {elem_type.title()}")


def _select_axis_element(state: "EditorState", elem_type: str, dpg) -> None:
    """Handle axis element selection (xaxis, yaxis)."""
    label = "X Axis" if elem_type == "xaxis" else "Y Axis"
    dpg.set_value("element_selector_combo", label)
    dpg.configure_item("axis_controls_group", show=True)

    o = state.current_overrides
    dpg.set_value("axis_linewidth_slider", o.get("axis_width", 0.2))
    dpg.set_value("axis_tick_length_slider", o.get("tick_length", 0.8))
    dpg.set_value("axis_tick_fontsize_slider", o.get("tick_fontsize", 7))

    if elem_type == "xaxis":
        dpg.set_value("axis_show_spine_checkbox", not o.get("hide_bottom_spine", False))
    else:
        dpg.set_value("axis_show_spine_checkbox", not o.get("hide_left_spine", False))

    dpg.set_value("selection_text", f"Selected: {label}")


def _select_legend(state: "EditorState", dpg) -> None:
    """Handle legend selection."""
    dpg.set_value("element_selector_combo", "Legend")
    dpg.configure_item("legend_controls_group", show=True)

    o = state.current_overrides
    dpg.set_value("legend_visible_edit", o.get("legend_visible", True))
    dpg.set_value("legend_frameon_edit", o.get("legend_frameon", False))
    dpg.set_value("legend_loc_edit", o.get("legend_loc", "best"))
    dpg.set_value("legend_fontsize_edit", o.get("legend_fontsize", 6))

    dpg.set_value("selection_text", "Selected: Legend")


def deselect_element(state: "EditorState", dpg) -> None:
    """Deselect the current element."""
    from ._rendering import update_preview

    state.selected_element = None
    state.selected_trace_index = None

    # Hide all control groups
    dpg.configure_item("trace_controls_group", show=False)
    dpg.configure_item("text_controls_group", show=False)
    dpg.configure_item("axis_controls_group", show=False)
    dpg.configure_item("legend_controls_group", show=False)

    dpg.set_value("selection_text", "")
    dpg.set_value("element_selector_combo", "")
    update_preview(state, dpg)


# EOF
