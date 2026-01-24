#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/_dearpygui/_panels/_control.py

"""
Control panel creation for DearPyGui editor.

Main orchestrator that delegates to section modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._state import EditorState


def create_control_panel(state: "EditorState", dpg) -> None:
    """Create the control panel with all editing options."""
    from .._handlers import (
        on_add_annotation,
        on_apply_limits,
        on_axis_element_change,
        on_element_selected,
        on_export_png,
        on_legend_element_change,
        on_remove_annotation,
        on_reset_overrides,
        on_save_manual,
        on_text_element_change,
        on_trace_property_change,
        on_value_change,
    )
    from .._rendering import update_preview
    from .._selection import deselect_element, get_all_element_labels
    from ._sections import (
        create_annotations_section,
        create_axis_limits_section,
        create_dimensions_section,
        create_font_settings_section,
        create_labels_section,
        create_legend_section,
        create_line_style_section,
        create_selected_element_section,
        create_style_section,
        create_tick_settings_section,
    )

    o = state.current_overrides

    with dpg.child_window(width=-1, height=-1, tag="control_panel"):
        dpg.add_text("Properties", color=(100, 200, 100))
        dpg.add_separator()

        # Labels Section
        create_labels_section(state, dpg, o, on_value_change)

        # Axis Limits Section
        create_axis_limits_section(state, dpg, o, on_apply_limits)

        # Line Style Section
        create_line_style_section(state, dpg, o, on_value_change)

        # Font Settings Section
        create_font_settings_section(state, dpg, o, on_value_change)

        # Tick Settings Section
        create_tick_settings_section(state, dpg, o, on_value_change)

        # Style Section
        create_style_section(state, dpg, o, on_value_change)

        # Legend Section
        create_legend_section(state, dpg, o, on_value_change)

        # Selected Element Section
        create_selected_element_section(
            state,
            dpg,
            get_all_element_labels,
            on_element_selected,
            on_trace_property_change,
            on_text_element_change,
            on_axis_element_change,
            on_legend_element_change,
            deselect_element,
        )

        # Dimensions Section
        create_dimensions_section(state, dpg, o, on_value_change)

        # Annotations Section
        create_annotations_section(state, dpg, on_add_annotation, on_remove_annotation)

        dpg.add_separator()

        # Action buttons
        dpg.add_button(
            label="Update Preview",
            callback=lambda: update_preview(state, dpg),
            width=-1,
        )
        dpg.add_button(
            label="Save to .manual.json",
            callback=lambda: on_save_manual(state, dpg),
            width=-1,
        )
        dpg.add_button(
            label="Reset to Original",
            callback=lambda: on_reset_overrides(state, dpg),
            width=-1,
        )
        dpg.add_button(
            label="Export PNG",
            callback=lambda: on_export_png(state, dpg),
            width=-1,
        )


# EOF
