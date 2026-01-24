#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_panels/_element_controls.py

"""
Element-specific control groups for DearPyGui editor.

Creates control groups for trace, text, axis, and legend elements.
"""

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .._state import EditorState


def create_trace_controls(
    state: "EditorState", dpg, on_trace_property_change: Callable
) -> None:
    """Create trace-specific controls."""
    with dpg.group(tag="trace_controls_group", show=False):
        dpg.add_input_text(
            label="Label",
            tag="trace_label_input",
            callback=lambda s, a: on_trace_property_change(state, dpg),
            on_enter=True,
            width=200,
        )
        dpg.add_color_edit(
            label="Color",
            tag="trace_color_picker",
            callback=lambda s, a: on_trace_property_change(state, dpg),
            no_alpha=True,
            width=200,
        )
        dpg.add_slider_float(
            label="Line Width",
            tag="trace_linewidth_slider",
            default_value=1.0,
            min_value=0.1,
            max_value=5.0,
            callback=lambda s, a: on_trace_property_change(state, dpg),
            width=200,
        )
        dpg.add_combo(
            label="Line Style",
            items=["-", "--", "-.", ":", ""],
            default_value="-",
            tag="trace_linestyle_combo",
            callback=lambda s, a: on_trace_property_change(state, dpg),
            width=100,
        )
        dpg.add_combo(
            label="Marker",
            items=["", "o", "s", "^", "v", "D", "x", "+", "*"],
            default_value="",
            tag="trace_marker_combo",
            callback=lambda s, a: on_trace_property_change(state, dpg),
            width=100,
        )
        dpg.add_slider_float(
            label="Marker Size",
            tag="trace_markersize_slider",
            default_value=6.0,
            min_value=1.0,
            max_value=20.0,
            callback=lambda s, a: on_trace_property_change(state, dpg),
            width=200,
        )


def create_text_controls(
    state: "EditorState", dpg, on_text_element_change: Callable
) -> None:
    """Create text element controls."""
    with dpg.group(tag="text_controls_group", show=False):
        dpg.add_input_text(
            label="Text",
            tag="element_text_input",
            callback=lambda s, a: on_text_element_change(state, dpg),
            on_enter=True,
            width=200,
        )
        dpg.add_slider_int(
            label="Font Size",
            tag="element_fontsize_slider",
            default_value=8,
            min_value=4,
            max_value=24,
            callback=lambda s, a: on_text_element_change(state, dpg),
            width=200,
        )
        dpg.add_color_edit(
            label="Color",
            tag="element_text_color",
            callback=lambda s, a: on_text_element_change(state, dpg),
            no_alpha=True,
            default_value=[0, 0, 0],
            width=200,
        )


def create_axis_controls(
    state: "EditorState", dpg, on_axis_element_change: Callable
) -> None:
    """Create axis element controls."""
    with dpg.group(tag="axis_controls_group", show=False):
        dpg.add_slider_float(
            label="Line Width (mm)",
            tag="axis_linewidth_slider",
            default_value=0.2,
            min_value=0.05,
            max_value=1.0,
            callback=lambda s, a: on_axis_element_change(state, dpg),
            width=200,
        )
        dpg.add_slider_float(
            label="Tick Length (mm)",
            tag="axis_tick_length_slider",
            default_value=0.8,
            min_value=0.2,
            max_value=3.0,
            callback=lambda s, a: on_axis_element_change(state, dpg),
            width=200,
        )
        dpg.add_slider_int(
            label="Tick Font Size",
            tag="axis_tick_fontsize_slider",
            default_value=7,
            min_value=4,
            max_value=16,
            callback=lambda s, a: on_axis_element_change(state, dpg),
            width=200,
        )
        dpg.add_checkbox(
            label="Show Spine",
            tag="axis_show_spine_checkbox",
            default_value=True,
            callback=lambda s, a: on_axis_element_change(state, dpg),
        )


def create_legend_controls(
    state: "EditorState", dpg, on_legend_element_change: Callable
) -> None:
    """Create legend controls."""
    with dpg.group(tag="legend_controls_group", show=False):
        dpg.add_checkbox(
            label="Visible",
            tag="legend_visible_edit",
            default_value=True,
            callback=lambda s, a: on_legend_element_change(state, dpg),
        )
        dpg.add_checkbox(
            label="Show Frame",
            tag="legend_frameon_edit",
            default_value=False,
            callback=lambda s, a: on_legend_element_change(state, dpg),
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
            default_value="best",
            tag="legend_loc_edit",
            callback=lambda s, a: on_legend_element_change(state, dpg),
            width=150,
        )
        dpg.add_slider_int(
            label="Font Size",
            tag="legend_fontsize_edit",
            default_value=6,
            min_value=4,
            max_value=14,
            callback=lambda s, a: on_legend_element_change(state, dpg),
            width=200,
        )


# EOF
