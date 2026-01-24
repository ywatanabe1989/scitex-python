#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_handlers.py

"""
Event handlers for DearPyGui editor.

Handles all callbacks and user interactions.
"""

import copy
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._state import EditorState


def on_value_change(state: "EditorState", dpg) -> None:
    """Handle value changes from widgets."""
    from ._rendering import update_preview

    state.user_modified = True
    collect_overrides(state, dpg)
    update_preview(state, dpg)


def collect_overrides(state: "EditorState", dpg) -> None:
    """Collect current values from all widgets."""
    o = state.current_overrides

    # Labels
    o["title"] = dpg.get_value("title_input")
    o["xlabel"] = dpg.get_value("xlabel_input")
    o["ylabel"] = dpg.get_value("ylabel_input")

    # Line style
    o["linewidth"] = dpg.get_value("linewidth_slider")

    # Font settings
    o["title_fontsize"] = dpg.get_value("title_fontsize_slider")
    o["axis_fontsize"] = dpg.get_value("axis_fontsize_slider")
    o["tick_fontsize"] = dpg.get_value("tick_fontsize_slider")
    o["legend_fontsize"] = dpg.get_value("legend_fontsize_slider")

    # Tick settings
    o["n_ticks"] = dpg.get_value("n_ticks_slider")
    o["tick_length"] = dpg.get_value("tick_length_slider")
    o["tick_width"] = dpg.get_value("tick_width_slider")
    o["tick_direction"] = dpg.get_value("tick_direction_combo")

    # Style
    o["grid"] = dpg.get_value("grid_checkbox")
    o["hide_top_spine"] = dpg.get_value("hide_top_spine_checkbox")
    o["hide_right_spine"] = dpg.get_value("hide_right_spine_checkbox")
    o["transparent"] = dpg.get_value("transparent_checkbox")
    o["axis_width"] = dpg.get_value("axis_width_slider")

    # Legend
    o["legend_visible"] = dpg.get_value("legend_visible_checkbox")
    o["legend_frameon"] = dpg.get_value("legend_frameon_checkbox")
    o["legend_loc"] = dpg.get_value("legend_loc_combo")

    # Dimensions
    o["fig_size"] = [
        dpg.get_value("fig_width_input"),
        dpg.get_value("fig_height_input"),
    ]
    o["dpi"] = dpg.get_value("dpi_slider")


def on_apply_limits(state: "EditorState", dpg) -> None:
    """Apply axis limits."""
    from ._rendering import update_preview

    xmin = dpg.get_value("xmin_input")
    xmax = dpg.get_value("xmax_input")
    ymin = dpg.get_value("ymin_input")
    ymax = dpg.get_value("ymax_input")

    if xmin < xmax:
        state.current_overrides["xlim"] = [xmin, xmax]
    if ymin < ymax:
        state.current_overrides["ylim"] = [ymin, ymax]

    state.user_modified = True
    update_preview(state, dpg)


def on_add_annotation(state: "EditorState", dpg) -> None:
    """Add text annotation."""
    from ._rendering import update_preview

    text = dpg.get_value("annot_text_input")
    if not text:
        return

    x = dpg.get_value("annot_x_input")
    y = dpg.get_value("annot_y_input")

    if "annotations" not in state.current_overrides:
        state.current_overrides["annotations"] = []

    state.current_overrides["annotations"].append(
        {
            "type": "text",
            "text": text,
            "x": x,
            "y": y,
            "fontsize": state.current_overrides.get("axis_fontsize", 7),
        }
    )

    dpg.set_value("annot_text_input", "")
    update_annotations_list(state, dpg)
    state.user_modified = True
    update_preview(state, dpg)


def on_remove_annotation(state: "EditorState", dpg) -> None:
    """Remove selected annotation."""
    from ._rendering import update_preview

    selected = dpg.get_value("annotations_listbox")
    annotations = state.current_overrides.get("annotations", [])

    if selected and annotations:
        # Find index by text
        for i, ann in enumerate(annotations):
            label = (
                f"{ann.get('text', '')[:20]} "
                f"({ann.get('x', 0):.2f}, {ann.get('y', 0):.2f})"
            )
            if label == selected:
                del annotations[i]
                break

        update_annotations_list(state, dpg)
        state.user_modified = True
        update_preview(state, dpg)


def update_annotations_list(state: "EditorState", dpg) -> None:
    """Update the annotations listbox."""
    annotations = state.current_overrides.get("annotations", [])
    items = []
    for ann in annotations:
        if ann.get("type") == "text":
            label = (
                f"{ann.get('text', '')[:20]} "
                f"({ann.get('x', 0):.2f}, {ann.get('y', 0):.2f})"
            )
            items.append(label)

    dpg.configure_item("annotations_listbox", items=items)


def on_preview_click(state: "EditorState", dpg, app_data) -> None:
    """Handle click on preview image to select element."""
    from ._selection import find_clicked_element, find_nearest_trace, select_element

    # Only handle left clicks
    if app_data != 0:  # 0 = left button
        return

    # Get mouse position relative to viewport
    mouse_pos = dpg.get_mouse_pos(local=False)

    # Get preview image position and size
    if not dpg.does_item_exist("preview_image"):
        return

    # Get the image item's position in the window
    img_pos = dpg.get_item_pos("preview_image")
    panel_pos = dpg.get_item_pos("preview_panel")

    # Calculate click position relative to image
    click_x = mouse_pos[0] - panel_pos[0] - img_pos[0]
    click_y = mouse_pos[1] - panel_pos[1] - img_pos[1]

    # Check if click is within image bounds (800x600)
    max_width, max_height = 800, 600
    if not (0 <= click_x <= max_width and 0 <= click_y <= max_height):
        return

    # First check if click is on a fixed element (title, labels, axes, legend)
    element = find_clicked_element(state, click_x, click_y)

    if element:
        select_element(state, element, dpg)
    else:
        # Fall back to trace selection
        trace_idx = find_nearest_trace(state, click_x, click_y, max_width, max_height)
        if trace_idx is not None:
            select_element(state, {"type": "trace", "index": trace_idx}, dpg)


def on_preview_hover(state: "EditorState", dpg, app_data) -> None:
    """Handle mouse move for hover effects on preview (optimized with caching)."""
    from ._rendering import update_hover_overlay
    from ._selection import find_clicked_element, find_nearest_trace

    # Throttle hover updates - reduced to 16ms (~60fps) since we use fast overlay
    current_time = time.time()
    if current_time - state.last_hover_check < 0.016:
        return
    state.last_hover_check = current_time

    # Get mouse position relative to viewport
    mouse_pos = dpg.get_mouse_pos(local=False)

    # Get preview image position
    if not dpg.does_item_exist("preview_image"):
        return

    img_pos = dpg.get_item_pos("preview_image")
    panel_pos = dpg.get_item_pos("preview_panel")

    # Calculate hover position relative to image
    hover_x = mouse_pos[0] - panel_pos[0] - img_pos[0]
    hover_y = mouse_pos[1] - panel_pos[1] - img_pos[1]

    # Check if within image bounds
    max_width, max_height = 800, 600
    if not (0 <= hover_x <= max_width and 0 <= hover_y <= max_height):
        if state.hovered_element is not None:
            state.hovered_element = None
            dpg.set_value("hover_text", "")
            # Use fast overlay update instead of full redraw
            update_hover_overlay(state, dpg)
        return

    # Find element under cursor
    element = find_clicked_element(state, hover_x, hover_y)

    if element is None:
        # Check for trace hover
        trace_idx = find_nearest_trace(state, hover_x, hover_y, max_width, max_height)
        if trace_idx is not None:
            element = {"type": "trace", "index": trace_idx}

    # Check if hover changed
    old_hover = state.hovered_element
    if element != old_hover:
        state.hovered_element = element
        if element:
            elem_type = element.get("type", "")
            elem_idx = element.get("index")
            if elem_type == "trace" and elem_idx is not None:
                traces = state.current_overrides.get("traces", [])
                if elem_idx < len(traces):
                    label = traces[elem_idx].get("label", f"Trace {elem_idx}")
                    dpg.set_value("hover_text", f"Hover: {label} (click to select)")
            else:
                label = elem_type.replace("x", "X ").replace("y", "Y ").title()
                dpg.set_value("hover_text", f"Hover: {label} (click to select)")
        else:
            dpg.set_value("hover_text", "")

        # Use fast overlay update for hover (no matplotlib re-render)
        update_hover_overlay(state, dpg)


def on_element_selected(state: "EditorState", dpg, app_data) -> None:
    """Handle element selection from combo box."""
    from ._selection import select_element

    if app_data == "Title":
        select_element(state, {"type": "title", "index": None}, dpg)
    elif app_data == "X Label":
        select_element(state, {"type": "xlabel", "index": None}, dpg)
    elif app_data == "Y Label":
        select_element(state, {"type": "ylabel", "index": None}, dpg)
    elif app_data == "X Axis":
        select_element(state, {"type": "xaxis", "index": None}, dpg)
    elif app_data == "Y Axis":
        select_element(state, {"type": "yaxis", "index": None}, dpg)
    elif app_data == "Legend":
        select_element(state, {"type": "legend", "index": None}, dpg)
    elif app_data.startswith("Trace: "):
        # Find trace index
        trace_label = app_data[7:]  # Remove "Trace: " prefix
        traces = state.current_overrides.get("traces", [])
        for i, t in enumerate(traces):
            if t.get("label", t.get("id", f"Trace {i}")) == trace_label:
                select_element(state, {"type": "trace", "index": i}, dpg)
                break


def on_text_element_change(state: "EditorState", dpg) -> None:
    """Handle changes to text element properties."""
    from ._rendering import update_preview

    if state.selected_element is None:
        return

    elem_type = state.selected_element.get("type")
    if elem_type not in ("title", "xlabel", "ylabel"):
        return

    text = dpg.get_value("element_text_input")
    fontsize = dpg.get_value("element_fontsize_slider")

    if elem_type == "title":
        state.current_overrides["title"] = text
        state.current_overrides["title_fontsize"] = fontsize
    elif elem_type == "xlabel":
        state.current_overrides["xlabel"] = text
        state.current_overrides["axis_fontsize"] = fontsize
    elif elem_type == "ylabel":
        state.current_overrides["ylabel"] = text
        state.current_overrides["axis_fontsize"] = fontsize

    state.user_modified = True
    update_preview(state, dpg)


def on_axis_element_change(state: "EditorState", dpg) -> None:
    """Handle changes to axis element properties."""
    from ._rendering import update_preview

    if state.selected_element is None:
        return

    elem_type = state.selected_element.get("type")
    if elem_type not in ("xaxis", "yaxis"):
        return

    state.current_overrides["axis_width"] = dpg.get_value("axis_linewidth_slider")
    state.current_overrides["tick_length"] = dpg.get_value("axis_tick_length_slider")
    state.current_overrides["tick_fontsize"] = dpg.get_value(
        "axis_tick_fontsize_slider"
    )

    show_spine = dpg.get_value("axis_show_spine_checkbox")
    if elem_type == "xaxis":
        state.current_overrides["hide_bottom_spine"] = not show_spine
    else:
        state.current_overrides["hide_left_spine"] = not show_spine

    state.user_modified = True
    update_preview(state, dpg)


def on_legend_element_change(state: "EditorState", dpg) -> None:
    """Handle changes to legend element properties."""
    from ._rendering import update_preview

    if state.selected_element is None:
        return

    elem_type = state.selected_element.get("type")
    if elem_type != "legend":
        return

    state.current_overrides["legend_visible"] = dpg.get_value("legend_visible_edit")
    state.current_overrides["legend_frameon"] = dpg.get_value("legend_frameon_edit")
    state.current_overrides["legend_loc"] = dpg.get_value("legend_loc_edit")
    state.current_overrides["legend_fontsize"] = dpg.get_value("legend_fontsize_edit")

    state.user_modified = True
    update_preview(state, dpg)


def on_trace_property_change(state: "EditorState", dpg) -> None:
    """Handle changes to selected trace properties."""
    from ._rendering import update_preview

    if state.selected_trace_index is None:
        return

    traces = state.current_overrides.get("traces", [])
    if state.selected_trace_index >= len(traces):
        return

    trace = traces[state.selected_trace_index]

    # Update trace properties from widgets
    trace["label"] = dpg.get_value("trace_label_input")

    # Convert RGB to hex
    color_rgb = dpg.get_value("trace_color_picker")
    if color_rgb and len(color_rgb) >= 3:
        r, g, b = int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2])
        trace["color"] = f"#{r:02x}{g:02x}{b:02x}"

    trace["linewidth"] = dpg.get_value("trace_linewidth_slider")
    trace["linestyle"] = dpg.get_value("trace_linestyle_combo")

    marker = dpg.get_value("trace_marker_combo")
    trace["marker"] = marker if marker else None

    trace["markersize"] = dpg.get_value("trace_markersize_slider")

    state.user_modified = True
    update_preview(state, dpg)


def on_save_manual(state: "EditorState", dpg) -> None:
    """Save current overrides to .manual.json."""
    from ..edit import save_manual_overrides

    try:
        collect_overrides(state, dpg)
        manual_path = save_manual_overrides(state.json_path, state.current_overrides)
        dpg.set_value("status_text", f"Saved: {manual_path.name}")
    except Exception as e:
        dpg.set_value("status_text", f"Error: {str(e)}")


def on_reset_overrides(state: "EditorState", dpg) -> None:
    """Reset to initial overrides."""
    from ._rendering import update_preview

    state.current_overrides = copy.deepcopy(state.initial_overrides)
    state.user_modified = False

    # Update all widgets
    dpg.set_value("title_input", state.current_overrides.get("title", ""))
    dpg.set_value("xlabel_input", state.current_overrides.get("xlabel", ""))
    dpg.set_value("ylabel_input", state.current_overrides.get("ylabel", ""))
    dpg.set_value("linewidth_slider", state.current_overrides.get("linewidth", 1.0))
    dpg.set_value("grid_checkbox", state.current_overrides.get("grid", False))
    dpg.set_value(
        "transparent_checkbox", state.current_overrides.get("transparent", True)
    )

    update_preview(state, dpg)
    dpg.set_value("status_text", "Reset to original")


def on_export_png(state: "EditorState", dpg) -> None:
    """Export current view to PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ._plotting import plot_from_csv

    try:
        collect_overrides(state, dpg)
        output_path = state.json_path.with_suffix(".edited.png")

        # Full resolution render
        o = state.current_overrides
        fig_size = o.get("fig_size", [3.15, 2.68])
        dpi = o.get("dpi", 300)

        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

        if state.csv_data is not None:
            plot_from_csv(ax, o, state.csv_data)

        if o.get("title"):
            ax.set_title(o["title"], fontsize=o.get("title_fontsize", 8))
        if o.get("xlabel"):
            ax.set_xlabel(o["xlabel"], fontsize=o.get("axis_fontsize", 7))
        if o.get("ylabel"):
            ax.set_ylabel(o["ylabel"], fontsize=o.get("axis_fontsize", 7))

        fig.tight_layout()
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            transparent=o.get("transparent", True),
        )
        plt.close(fig)

        dpg.set_value("status_text", f"Exported: {output_path.name}")
    except Exception as e:
        dpg.set_value("status_text", f"Error: {str(e)}")


# EOF
