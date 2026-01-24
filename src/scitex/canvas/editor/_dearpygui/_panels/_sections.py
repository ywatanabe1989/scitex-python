#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_panels/_sections.py

"""
Control panel section creators for DearPyGui editor.

Each function creates a collapsible section with related controls.
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .._state import EditorState


def create_labels_section(state: "EditorState", dpg, o: dict, on_value_change) -> None:
    """Create labels section."""
    with dpg.collapsing_header(label="Labels", default_open=True):
        dpg.add_input_text(
            label="Title",
            default_value=o.get("title", ""),
            tag="title_input",
            callback=lambda s, a: on_value_change(state, dpg),
            on_enter=True,
            width=250,
        )
        dpg.add_input_text(
            label="X Label",
            default_value=o.get("xlabel", ""),
            tag="xlabel_input",
            callback=lambda s, a: on_value_change(state, dpg),
            on_enter=True,
            width=250,
        )
        dpg.add_input_text(
            label="Y Label",
            default_value=o.get("ylabel", ""),
            tag="ylabel_input",
            callback=lambda s, a: on_value_change(state, dpg),
            on_enter=True,
            width=250,
        )


def create_axis_limits_section(
    state: "EditorState", dpg, o: dict, on_apply_limits
) -> None:
    """Create axis limits section."""
    with dpg.collapsing_header(label="Axis Limits", default_open=False):
        with dpg.group(horizontal=True):
            xlim = o.get("xlim", [0, 1])
            dpg.add_input_float(
                label="X Min",
                default_value=xlim[0] if xlim else 0,
                tag="xmin_input",
                width=100,
            )
            dpg.add_input_float(
                label="X Max",
                default_value=xlim[1] if xlim else 1,
                tag="xmax_input",
                width=100,
            )
        with dpg.group(horizontal=True):
            ylim = o.get("ylim", [0, 1])
            dpg.add_input_float(
                label="Y Min",
                default_value=ylim[0] if ylim else 0,
                tag="ymin_input",
                width=100,
            )
            dpg.add_input_float(
                label="Y Max",
                default_value=ylim[1] if ylim else 1,
                tag="ymax_input",
                width=100,
            )
        dpg.add_button(
            label="Apply Limits",
            callback=lambda: on_apply_limits(state, dpg),
            width=150,
        )


def create_line_style_section(
    state: "EditorState", dpg, o: dict, on_value_change
) -> None:
    """Create line style section."""
    with dpg.collapsing_header(label="Line Style", default_open=True):
        dpg.add_slider_float(
            label="Line Width (pt)",
            default_value=o.get("linewidth", 1.0),
            min_value=0.1,
            max_value=5.0,
            tag="linewidth_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )


def create_font_settings_section(
    state: "EditorState", dpg, o: dict, on_value_change
) -> None:
    """Create font settings section."""
    with dpg.collapsing_header(label="Font Settings", default_open=False):
        dpg.add_slider_int(
            label="Title Font Size",
            default_value=o.get("title_fontsize", 8),
            min_value=6,
            max_value=20,
            tag="title_fontsize_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_slider_int(
            label="Axis Font Size",
            default_value=o.get("axis_fontsize", 7),
            min_value=6,
            max_value=16,
            tag="axis_fontsize_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_slider_int(
            label="Tick Font Size",
            default_value=o.get("tick_fontsize", 7),
            min_value=6,
            max_value=16,
            tag="tick_fontsize_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_slider_int(
            label="Legend Font Size",
            default_value=o.get("legend_fontsize", 6),
            min_value=4,
            max_value=14,
            tag="legend_fontsize_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )


def create_tick_settings_section(
    state: "EditorState", dpg, o: dict, on_value_change
) -> None:
    """Create tick settings section."""
    with dpg.collapsing_header(label="Tick Settings", default_open=False):
        dpg.add_slider_int(
            label="N Ticks",
            default_value=o.get("n_ticks", 4),
            min_value=2,
            max_value=10,
            tag="n_ticks_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_slider_float(
            label="Tick Length (mm)",
            default_value=o.get("tick_length", 0.8),
            min_value=0.2,
            max_value=3.0,
            tag="tick_length_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_slider_float(
            label="Tick Width (mm)",
            default_value=o.get("tick_width", 0.2),
            min_value=0.05,
            max_value=1.0,
            tag="tick_width_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )
        dpg.add_combo(
            label="Tick Direction",
            items=["out", "in", "inout"],
            default_value=o.get("tick_direction", "out"),
            tag="tick_direction_combo",
            callback=lambda s, a: on_value_change(state, dpg),
            width=150,
        )


def create_style_section(state: "EditorState", dpg, o: dict, on_value_change) -> None:
    """Create style section."""
    with dpg.collapsing_header(label="Style", default_open=True):
        dpg.add_checkbox(
            label="Show Grid",
            default_value=o.get("grid", False),
            tag="grid_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_checkbox(
            label="Hide Top Spine",
            default_value=o.get("hide_top_spine", True),
            tag="hide_top_spine_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_checkbox(
            label="Hide Right Spine",
            default_value=o.get("hide_right_spine", True),
            tag="hide_right_spine_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_checkbox(
            label="Transparent Background",
            default_value=o.get("transparent", True),
            tag="transparent_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_slider_float(
            label="Axis Width (mm)",
            default_value=o.get("axis_width", 0.2),
            min_value=0.05,
            max_value=1.0,
            tag="axis_width_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )


def create_legend_section(state: "EditorState", dpg, o: dict, on_value_change) -> None:
    """Create legend section."""
    with dpg.collapsing_header(label="Legend", default_open=False):
        dpg.add_checkbox(
            label="Show Legend",
            default_value=o.get("legend_visible", True),
            tag="legend_visible_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_checkbox(
            label="Show Frame",
            default_value=o.get("legend_frameon", False),
            tag="legend_frameon_checkbox",
            callback=lambda s, a: on_value_change(state, dpg),
        )
        dpg.add_combo(
            label="Position",
            items=[
                "best",
                "upper right",
                "upper left",
                "lower right",
                "lower left",
                "center right",
                "center left",
                "upper center",
                "lower center",
                "center",
            ],
            default_value=o.get("legend_loc", "best"),
            tag="legend_loc_combo",
            callback=lambda s, a: on_value_change(state, dpg),
            width=150,
        )


def create_dimensions_section(
    state: "EditorState", dpg, o: dict, on_value_change
) -> None:
    """Create dimensions section."""
    with dpg.collapsing_header(label="Dimensions", default_open=False):
        fig_size = o.get("fig_size", [3.15, 2.68])
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="Width (in)",
                default_value=fig_size[0],
                tag="fig_width_input",
                width=100,
            )
            dpg.add_input_float(
                label="Height (in)",
                default_value=fig_size[1],
                tag="fig_height_input",
                width=100,
            )
        dpg.add_slider_int(
            label="DPI",
            default_value=o.get("dpi", 300),
            min_value=72,
            max_value=600,
            tag="dpi_slider",
            callback=lambda s, a: on_value_change(state, dpg),
            width=200,
        )


def create_annotations_section(
    state: "EditorState", dpg, on_add_annotation, on_remove_annotation
) -> None:
    """Create annotations section."""
    with dpg.collapsing_header(label="Annotations", default_open=False):
        dpg.add_input_text(
            label="Text",
            tag="annot_text_input",
            width=200,
        )
        with dpg.group(horizontal=True):
            dpg.add_input_float(
                label="X",
                default_value=0.5,
                tag="annot_x_input",
                width=80,
            )
            dpg.add_input_float(
                label="Y",
                default_value=0.5,
                tag="annot_y_input",
                width=80,
            )
        dpg.add_button(
            label="Add Annotation",
            callback=lambda: on_add_annotation(state, dpg),
            width=150,
        )
        dpg.add_listbox(
            label="",
            items=[],
            tag="annotations_listbox",
            num_items=4,
            width=250,
        )
        dpg.add_button(
            label="Remove Selected",
            callback=lambda: on_remove_annotation(state, dpg),
            width=150,
        )


def create_selected_element_section(
    state: "EditorState",
    dpg,
    get_all_element_labels: Callable,
    on_element_selected: Callable,
    on_trace_property_change: Callable,
    on_text_element_change: Callable,
    on_axis_element_change: Callable,
    on_legend_element_change: Callable,
    deselect_element: Callable,
) -> None:
    """Create selected element section."""
    from ._element_controls import (
        create_axis_controls,
        create_legend_controls,
        create_text_controls,
        create_trace_controls,
    )

    with dpg.collapsing_header(
        label="Selected Element",
        default_open=True,
        tag="selected_element_header",
    ):
        dpg.add_text(
            "Click on preview to select elements",
            tag="element_hint_text",
            color=(150, 150, 150),
        )
        dpg.add_combo(
            label="Element",
            items=get_all_element_labels(state),
            tag="element_selector_combo",
            callback=lambda s, a: on_element_selected(state, dpg, a),
            width=200,
        )
        dpg.add_separator()

        # Trace-specific controls (shown when trace selected)
        create_trace_controls(state, dpg, on_trace_property_change)

        # Text element controls (title, xlabel, ylabel)
        create_text_controls(state, dpg, on_text_element_change)

        # Axis element controls (xaxis, yaxis)
        create_axis_controls(state, dpg, on_axis_element_change)

        # Legend controls
        create_legend_controls(state, dpg, on_legend_element_change)

        dpg.add_button(
            label="Deselect",
            callback=lambda: deselect_element(state, dpg),
            width=100,
        )


# EOF
