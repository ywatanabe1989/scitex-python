#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: 2025-12-08
# File: ./src/scitex/vis/editor/_dearpygui_editor.py
"""DearPyGui-based figure editor with GPU-accelerated rendering."""

from pathlib import Path
from typing import Dict, Any, Optional
import copy
import io
import base64


def _create_checkerboard(width: int, height: int, square_size: int = 10) -> "Image":
    """Create a checkerboard pattern image for transparency preview.

    Parameters
    ----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    square_size : int
        Size of each checkerboard square (default: 10)

    Returns
    -------
    PIL.Image
        RGBA image with checkerboard pattern (light/dark gray)
    """
    from PIL import Image
    import numpy as np

    # Create checkerboard pattern
    light_gray = (220, 220, 220, 255)
    dark_gray = (180, 180, 180, 255)

    # Create array
    img_array = np.zeros((height, width, 4), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Determine which square we're in
            square_x = x // square_size
            square_y = y // square_size
            if (square_x + square_y) % 2 == 0:
                img_array[y, x] = light_gray
            else:
                img_array[y, x] = dark_gray

    return Image.fromarray(img_array, "RGBA")


class DearPyGuiEditor:
    """
    GPU-accelerated figure editor using DearPyGui.

    Features:
    - Modern immediate-mode GUI with GPU acceleration
    - Real-time figure preview
    - Property editors with sliders, color pickers, and input fields
    - Click-to-select traces on preview
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    - Dark/light theme support
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.png_path = Path(png_path) if png_path else None
        self.manual_overrides = manual_overrides or {}

        # Get SciTeX defaults and merge with metadata
        from ._defaults import get_scitex_defaults, extract_defaults_from_metadata

        self.scitex_defaults = get_scitex_defaults()
        self.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Start with defaults, then overlay manual overrides
        self.current_overrides = copy.deepcopy(self.scitex_defaults)
        self.current_overrides.update(self.metadata_defaults)
        self.current_overrides.update(self.manual_overrides)

        # Track modifications
        self._initial_overrides = copy.deepcopy(self.current_overrides)
        self._user_modified = False
        self._texture_id = None

        # Click-to-select state
        self._selected_element = None  # {'type': 'trace'|'title'|'xlabel'|'ylabel'|'legend'|'xaxis'|'yaxis', 'index': int|None}
        self._selected_trace_index = None  # Legacy compat
        self._preview_bounds = (
            None  # (x_offset, y_offset, width, height) of figure in preview
        )
        self._axes_transform = None  # Transform info for data coordinates
        self._element_bboxes = {}  # Store bboxes for all selectable elements

        # Hover state
        self._hovered_element = None  # Element currently being hovered
        self._last_hover_check = 0  # For throttling hover updates
        self._backend_name = "dearpygui"  # Backend name for title

        # Cached rendering for fast hover response
        self._cached_base_image = None  # PIL Image of base figure (no highlights)
        self._cached_base_data = None  # Flattened RGBA data for DearPyGui
        self._cache_dirty = True  # Flag to indicate cache needs rebuild

    def run(self):
        """Launch the DearPyGui editor."""
        try:
            import dearpygui.dearpygui as dpg
        except ImportError:
            raise ImportError(
                "DearPyGui is required for this editor. "
                "Install with: pip install dearpygui"
            )

        dpg.create_context()

        # Configure viewport
        dpg.create_viewport(
            title=f"SciTeX Editor ({self._backend_name}) - {self.json_path.name}",
            width=1400,
            height=900,
        )

        # Create texture registry for image preview
        with dpg.texture_registry(show=False):
            # Create initial texture with placeholder
            width, height = 800, 600
            texture_data = [0.2, 0.2, 0.2, 1.0] * (width * height)
            self._texture_id = dpg.add_dynamic_texture(
                width=width,
                height=height,
                default_value=texture_data,
                tag="preview_texture",
            )

        # Create main window
        with dpg.window(label="SciTeX Figure Editor", tag="main_window"):
            with dpg.group(horizontal=True):
                # Left panel: Preview
                self._create_preview_panel(dpg)

                # Right panel: Controls
                self._create_control_panel(dpg)

        # Set main window as primary
        dpg.set_primary_window("main_window", True)

        # Initial render
        self._update_preview(dpg)

        # Setup and show
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

    def _create_preview_panel(self, dpg):
        """Create the preview panel with figure image, click handler, and hover detection."""
        with dpg.child_window(width=900, height=-1, tag="preview_panel"):
            dpg.add_text(
                "Figure Preview (click to select, hover to highlight)",
                color=(100, 200, 100),
            )
            dpg.add_separator()

            # Image display with click and move handlers
            with dpg.handler_registry(tag="preview_handler"):
                dpg.add_mouse_click_handler(callback=self._on_preview_click)
                dpg.add_mouse_move_handler(callback=self._on_preview_hover)

            dpg.add_image("preview_texture", tag="preview_image")

            dpg.add_separator()
            dpg.add_text("", tag="hover_text", color=(150, 200, 150))
            dpg.add_text("", tag="status_text", color=(150, 150, 150))
            dpg.add_text("", tag="selection_text", color=(200, 200, 100))

    def _create_control_panel(self, dpg):
        """Create the control panel with all editing options."""
        with dpg.child_window(width=-1, height=-1, tag="control_panel"):
            dpg.add_text("Properties", color=(100, 200, 100))
            dpg.add_separator()

            # Labels Section
            with dpg.collapsing_header(label="Labels", default_open=True):
                dpg.add_input_text(
                    label="Title",
                    default_value=self.current_overrides.get("title", ""),
                    tag="title_input",
                    callback=self._on_value_change,
                    on_enter=True,
                    width=250,
                )
                dpg.add_input_text(
                    label="X Label",
                    default_value=self.current_overrides.get("xlabel", ""),
                    tag="xlabel_input",
                    callback=self._on_value_change,
                    on_enter=True,
                    width=250,
                )
                dpg.add_input_text(
                    label="Y Label",
                    default_value=self.current_overrides.get("ylabel", ""),
                    tag="ylabel_input",
                    callback=self._on_value_change,
                    on_enter=True,
                    width=250,
                )

            # Axis Limits Section
            with dpg.collapsing_header(label="Axis Limits", default_open=False):
                with dpg.group(horizontal=True):
                    xlim = self.current_overrides.get("xlim", [0, 1])
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
                    ylim = self.current_overrides.get("ylim", [0, 1])
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
                    callback=self._apply_limits,
                    width=150,
                )

            # Line Style Section
            with dpg.collapsing_header(label="Line Style", default_open=True):
                dpg.add_slider_float(
                    label="Line Width (pt)",
                    default_value=self.current_overrides.get("linewidth", 1.0),
                    min_value=0.1,
                    max_value=5.0,
                    tag="linewidth_slider",
                    callback=self._on_value_change,
                    width=200,
                )

            # Font Settings Section
            with dpg.collapsing_header(label="Font Settings", default_open=False):
                dpg.add_slider_int(
                    label="Title Font Size",
                    default_value=self.current_overrides.get("title_fontsize", 8),
                    min_value=6,
                    max_value=20,
                    tag="title_fontsize_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_slider_int(
                    label="Axis Font Size",
                    default_value=self.current_overrides.get("axis_fontsize", 7),
                    min_value=6,
                    max_value=16,
                    tag="axis_fontsize_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_slider_int(
                    label="Tick Font Size",
                    default_value=self.current_overrides.get("tick_fontsize", 7),
                    min_value=6,
                    max_value=16,
                    tag="tick_fontsize_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_slider_int(
                    label="Legend Font Size",
                    default_value=self.current_overrides.get("legend_fontsize", 6),
                    min_value=4,
                    max_value=14,
                    tag="legend_fontsize_slider",
                    callback=self._on_value_change,
                    width=200,
                )

            # Tick Settings Section
            with dpg.collapsing_header(label="Tick Settings", default_open=False):
                dpg.add_slider_int(
                    label="N Ticks",
                    default_value=self.current_overrides.get("n_ticks", 4),
                    min_value=2,
                    max_value=10,
                    tag="n_ticks_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_slider_float(
                    label="Tick Length (mm)",
                    default_value=self.current_overrides.get("tick_length", 0.8),
                    min_value=0.2,
                    max_value=3.0,
                    tag="tick_length_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_slider_float(
                    label="Tick Width (mm)",
                    default_value=self.current_overrides.get("tick_width", 0.2),
                    min_value=0.05,
                    max_value=1.0,
                    tag="tick_width_slider",
                    callback=self._on_value_change,
                    width=200,
                )
                dpg.add_combo(
                    label="Tick Direction",
                    items=["out", "in", "inout"],
                    default_value=self.current_overrides.get("tick_direction", "out"),
                    tag="tick_direction_combo",
                    callback=self._on_value_change,
                    width=150,
                )

            # Style Section
            with dpg.collapsing_header(label="Style", default_open=True):
                dpg.add_checkbox(
                    label="Show Grid",
                    default_value=self.current_overrides.get("grid", False),
                    tag="grid_checkbox",
                    callback=self._on_value_change,
                )
                dpg.add_checkbox(
                    label="Hide Top Spine",
                    default_value=self.current_overrides.get("hide_top_spine", True),
                    tag="hide_top_spine_checkbox",
                    callback=self._on_value_change,
                )
                dpg.add_checkbox(
                    label="Hide Right Spine",
                    default_value=self.current_overrides.get("hide_right_spine", True),
                    tag="hide_right_spine_checkbox",
                    callback=self._on_value_change,
                )
                dpg.add_checkbox(
                    label="Transparent Background",
                    default_value=self.current_overrides.get("transparent", True),
                    tag="transparent_checkbox",
                    callback=self._on_value_change,
                )
                dpg.add_slider_float(
                    label="Axis Width (mm)",
                    default_value=self.current_overrides.get("axis_width", 0.2),
                    min_value=0.05,
                    max_value=1.0,
                    tag="axis_width_slider",
                    callback=self._on_value_change,
                    width=200,
                )

            # Legend Section
            with dpg.collapsing_header(label="Legend", default_open=False):
                dpg.add_checkbox(
                    label="Show Legend",
                    default_value=self.current_overrides.get("legend_visible", True),
                    tag="legend_visible_checkbox",
                    callback=self._on_value_change,
                )
                dpg.add_checkbox(
                    label="Show Frame",
                    default_value=self.current_overrides.get("legend_frameon", False),
                    tag="legend_frameon_checkbox",
                    callback=self._on_value_change,
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
                    default_value=self.current_overrides.get("legend_loc", "best"),
                    tag="legend_loc_combo",
                    callback=self._on_value_change,
                    width=150,
                )

            # Selected Element Section (click on preview to select)
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
                    items=self._get_all_element_labels(),
                    tag="element_selector_combo",
                    callback=self._on_element_selected,
                    width=200,
                )
                dpg.add_separator()

                # Trace-specific controls (shown when trace selected)
                with dpg.group(tag="trace_controls_group", show=False):
                    dpg.add_input_text(
                        label="Label",
                        tag="trace_label_input",
                        callback=self._on_trace_property_change,
                        on_enter=True,
                        width=200,
                    )
                    dpg.add_color_edit(
                        label="Color",
                        tag="trace_color_picker",
                        callback=self._on_trace_property_change,
                        no_alpha=True,
                        width=200,
                    )
                    dpg.add_slider_float(
                        label="Line Width",
                        tag="trace_linewidth_slider",
                        default_value=1.0,
                        min_value=0.1,
                        max_value=5.0,
                        callback=self._on_trace_property_change,
                        width=200,
                    )
                    dpg.add_combo(
                        label="Line Style",
                        items=["-", "--", "-.", ":", ""],
                        default_value="-",
                        tag="trace_linestyle_combo",
                        callback=self._on_trace_property_change,
                        width=100,
                    )
                    dpg.add_combo(
                        label="Marker",
                        items=["", "o", "s", "^", "v", "D", "x", "+", "*"],
                        default_value="",
                        tag="trace_marker_combo",
                        callback=self._on_trace_property_change,
                        width=100,
                    )
                    dpg.add_slider_float(
                        label="Marker Size",
                        tag="trace_markersize_slider",
                        default_value=6.0,
                        min_value=1.0,
                        max_value=20.0,
                        callback=self._on_trace_property_change,
                        width=200,
                    )

                # Text element controls (title, xlabel, ylabel)
                with dpg.group(tag="text_controls_group", show=False):
                    dpg.add_input_text(
                        label="Text",
                        tag="element_text_input",
                        callback=self._on_text_element_change,
                        on_enter=True,
                        width=200,
                    )
                    dpg.add_slider_int(
                        label="Font Size",
                        tag="element_fontsize_slider",
                        default_value=8,
                        min_value=4,
                        max_value=24,
                        callback=self._on_text_element_change,
                        width=200,
                    )
                    dpg.add_color_edit(
                        label="Color",
                        tag="element_text_color",
                        callback=self._on_text_element_change,
                        no_alpha=True,
                        default_value=[0, 0, 0],
                        width=200,
                    )

                # Axis element controls (xaxis, yaxis)
                with dpg.group(tag="axis_controls_group", show=False):
                    dpg.add_slider_float(
                        label="Line Width (mm)",
                        tag="axis_linewidth_slider",
                        default_value=0.2,
                        min_value=0.05,
                        max_value=1.0,
                        callback=self._on_axis_element_change,
                        width=200,
                    )
                    dpg.add_slider_float(
                        label="Tick Length (mm)",
                        tag="axis_tick_length_slider",
                        default_value=0.8,
                        min_value=0.2,
                        max_value=3.0,
                        callback=self._on_axis_element_change,
                        width=200,
                    )
                    dpg.add_slider_int(
                        label="Tick Font Size",
                        tag="axis_tick_fontsize_slider",
                        default_value=7,
                        min_value=4,
                        max_value=16,
                        callback=self._on_axis_element_change,
                        width=200,
                    )
                    dpg.add_checkbox(
                        label="Show Spine",
                        tag="axis_show_spine_checkbox",
                        default_value=True,
                        callback=self._on_axis_element_change,
                    )

                # Legend controls
                with dpg.group(tag="legend_controls_group", show=False):
                    dpg.add_checkbox(
                        label="Visible",
                        tag="legend_visible_edit",
                        default_value=True,
                        callback=self._on_legend_element_change,
                    )
                    dpg.add_checkbox(
                        label="Show Frame",
                        tag="legend_frameon_edit",
                        default_value=False,
                        callback=self._on_legend_element_change,
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
                        callback=self._on_legend_element_change,
                        width=150,
                    )
                    dpg.add_slider_int(
                        label="Font Size",
                        tag="legend_fontsize_edit",
                        default_value=6,
                        min_value=4,
                        max_value=14,
                        callback=self._on_legend_element_change,
                        width=200,
                    )

                dpg.add_button(
                    label="Deselect",
                    callback=self._deselect_element,
                    width=100,
                )

            # Dimensions Section
            with dpg.collapsing_header(label="Dimensions", default_open=False):
                fig_size = self.current_overrides.get("fig_size", [3.15, 2.68])
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
                    default_value=self.current_overrides.get("dpi", 300),
                    min_value=72,
                    max_value=600,
                    tag="dpi_slider",
                    callback=self._on_value_change,
                    width=200,
                )

            # Annotations Section
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
                    callback=self._add_annotation,
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
                    callback=self._remove_annotation,
                    width=150,
                )

            dpg.add_separator()

            # Action buttons
            dpg.add_button(
                label="Update Preview",
                callback=lambda: self._update_preview(dpg),
                width=-1,
            )
            dpg.add_button(
                label="Save to .manual.json",
                callback=self._save_manual,
                width=-1,
            )
            dpg.add_button(
                label="Reset to Original",
                callback=self._reset_overrides,
                width=-1,
            )
            dpg.add_button(
                label="Export PNG",
                callback=self._export_png,
                width=-1,
            )

    def _on_value_change(self, sender, app_data, user_data=None):
        """Handle value changes from widgets."""
        import dearpygui.dearpygui as dpg

        self._user_modified = True
        self._collect_overrides(dpg)
        self._update_preview(dpg)

    def _collect_overrides(self, dpg):
        """Collect current values from all widgets."""
        # Labels
        self.current_overrides["title"] = dpg.get_value("title_input")
        self.current_overrides["xlabel"] = dpg.get_value("xlabel_input")
        self.current_overrides["ylabel"] = dpg.get_value("ylabel_input")

        # Line style
        self.current_overrides["linewidth"] = dpg.get_value("linewidth_slider")

        # Font settings
        self.current_overrides["title_fontsize"] = dpg.get_value(
            "title_fontsize_slider"
        )
        self.current_overrides["axis_fontsize"] = dpg.get_value("axis_fontsize_slider")
        self.current_overrides["tick_fontsize"] = dpg.get_value("tick_fontsize_slider")
        self.current_overrides["legend_fontsize"] = dpg.get_value(
            "legend_fontsize_slider"
        )

        # Tick settings
        self.current_overrides["n_ticks"] = dpg.get_value("n_ticks_slider")
        self.current_overrides["tick_length"] = dpg.get_value("tick_length_slider")
        self.current_overrides["tick_width"] = dpg.get_value("tick_width_slider")
        self.current_overrides["tick_direction"] = dpg.get_value("tick_direction_combo")

        # Style
        self.current_overrides["grid"] = dpg.get_value("grid_checkbox")
        self.current_overrides["hide_top_spine"] = dpg.get_value(
            "hide_top_spine_checkbox"
        )
        self.current_overrides["hide_right_spine"] = dpg.get_value(
            "hide_right_spine_checkbox"
        )
        self.current_overrides["transparent"] = dpg.get_value("transparent_checkbox")
        self.current_overrides["axis_width"] = dpg.get_value("axis_width_slider")

        # Legend
        self.current_overrides["legend_visible"] = dpg.get_value(
            "legend_visible_checkbox"
        )
        self.current_overrides["legend_frameon"] = dpg.get_value(
            "legend_frameon_checkbox"
        )
        self.current_overrides["legend_loc"] = dpg.get_value("legend_loc_combo")

        # Dimensions
        self.current_overrides["fig_size"] = [
            dpg.get_value("fig_width_input"),
            dpg.get_value("fig_height_input"),
        ]
        self.current_overrides["dpi"] = dpg.get_value("dpi_slider")

    def _apply_limits(self, sender=None, app_data=None, user_data=None):
        """Apply axis limits."""
        import dearpygui.dearpygui as dpg

        xmin = dpg.get_value("xmin_input")
        xmax = dpg.get_value("xmax_input")
        ymin = dpg.get_value("ymin_input")
        ymax = dpg.get_value("ymax_input")

        if xmin < xmax:
            self.current_overrides["xlim"] = [xmin, xmax]
        if ymin < ymax:
            self.current_overrides["ylim"] = [ymin, ymax]

        self._user_modified = True
        self._update_preview(dpg)

    def _add_annotation(self, sender=None, app_data=None, user_data=None):
        """Add text annotation."""
        import dearpygui.dearpygui as dpg

        text = dpg.get_value("annot_text_input")
        if not text:
            return

        x = dpg.get_value("annot_x_input")
        y = dpg.get_value("annot_y_input")

        if "annotations" not in self.current_overrides:
            self.current_overrides["annotations"] = []

        self.current_overrides["annotations"].append(
            {
                "type": "text",
                "text": text,
                "x": x,
                "y": y,
                "fontsize": self.current_overrides.get("axis_fontsize", 7),
            }
        )

        dpg.set_value("annot_text_input", "")
        self._update_annotations_list(dpg)
        self._user_modified = True
        self._update_preview(dpg)

    def _remove_annotation(self, sender=None, app_data=None, user_data=None):
        """Remove selected annotation."""
        import dearpygui.dearpygui as dpg

        selected = dpg.get_value("annotations_listbox")
        annotations = self.current_overrides.get("annotations", [])

        if selected and annotations:
            # Find index by text
            for i, ann in enumerate(annotations):
                label = f"{ann.get('text', '')[:20]} ({ann.get('x', 0):.2f}, {ann.get('y', 0):.2f})"
                if label == selected:
                    del annotations[i]
                    break

            self._update_annotations_list(dpg)
            self._user_modified = True
            self._update_preview(dpg)

    def _update_annotations_list(self, dpg):
        """Update the annotations listbox."""
        annotations = self.current_overrides.get("annotations", [])
        items = []
        for ann in annotations:
            if ann.get("type") == "text":
                label = f"{ann.get('text', '')[:20]} ({ann.get('x', 0):.2f}, {ann.get('y', 0):.2f})"
                items.append(label)

        dpg.configure_item("annotations_listbox", items=items)

    def _update_preview(self, dpg):
        """Update the figure preview (full re-render)."""
        try:
            # Mark cache dirty and do full render
            self._cache_dirty = True
            img_data, width, height = self._render_figure()

            # Update texture
            dpg.set_value("preview_texture", img_data)

            # Update status
            dpg.set_value("status_text", f"Preview updated ({width}x{height})")

        except Exception as e:
            dpg.set_value("status_text", f"Error: {str(e)}")

    def _update_hover_overlay(self, dpg):
        """Fast hover overlay update using cached base image (no matplotlib re-render)."""
        import numpy as np
        from PIL import Image, ImageDraw

        # If no cached base, do full render
        if self._cached_base_image is None:
            self._update_preview(dpg)
            return

        try:
            # Start with a copy of cached base
            img = self._cached_base_image.copy()
            draw = ImageDraw.Draw(img, "RGBA")

            # Get hover element type
            hovered_type = (
                self._hovered_element.get("type") if self._hovered_element else None
            )
            selected_type = (
                self._selected_element.get("type") if self._selected_element else None
            )

            # Draw hover highlight (outline only, no fill) for non-trace elements
            if (
                hovered_type
                and hovered_type != "trace"
                and hovered_type != selected_type
            ):
                bbox = self._element_bboxes.get(hovered_type)
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
                bbox = self._element_bboxes.get(selected_type)
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

        except Exception as e:
            # Fallback to full render on error
            self._update_preview(dpg)

    def _render_figure(self):
        """Render figure and return as RGBA data for texture."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        import numpy as np
        from PIL import Image
        import dearpygui.dearpygui as dpg

        # mm to pt conversion
        mm_to_pt = 2.83465

        o = self.current_overrides

        # Dimensions - use fixed size for preview
        preview_dpi = 100
        fig_size = o.get("fig_size", [3.15, 2.68])

        # Create figure with white background for preview
        fig, ax = plt.subplots(figsize=fig_size, dpi=preview_dpi)

        # For preview, use white background (transparent doesn't show well in GUI)
        fig.patch.set_facecolor("white")
        ax.patch.set_facecolor("white")

        # Plot from CSV data (only pass selection, hover is via PIL overlay for speed)
        if self.csv_data is not None:
            self._plot_from_csv(ax, o, highlight_trace=self._selected_trace_index)
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
            length=o.get("tick_length", 0.8) * mm_to_pt,
            width=o.get("tick_width", 0.2) * mm_to_pt,
            direction=o.get("tick_direction", "out"),
        )

        # Number of ticks
        ax.xaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=o.get("n_ticks", 4)))

        # Grid
        if o.get("grid"):
            ax.grid(True, linewidth=o.get("axis_width", 0.2) * mm_to_pt, alpha=0.3)

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
            spine.set_linewidth(o.get("axis_width", 0.2) * mm_to_pt)

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
        self._draw_element_highlights(fig, ax)

        # Store axes transform info for click-to-select
        fig.canvas.draw()
        ax_bbox = ax.get_position()
        fig_width_px = int(fig_size[0] * preview_dpi)
        fig_height_px = int(fig_size[1] * preview_dpi)

        # Collect element bboxes for click detection (in figure pixel coordinates)
        # We'll scale these later after resize
        self._element_bboxes_raw = {}

        # Title bbox
        if ax.title.get_text():
            try:
                title_bbox = ax.title.get_window_extent(fig.canvas.get_renderer())
                self._element_bboxes_raw["title"] = (
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
                xlabel_bbox = ax.xaxis.label.get_window_extent(
                    fig.canvas.get_renderer()
                )
                self._element_bboxes_raw["xlabel"] = (
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
                ylabel_bbox = ax.yaxis.label.get_window_extent(
                    fig.canvas.get_renderer()
                )
                self._element_bboxes_raw["ylabel"] = (
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
                legend_bbox = legend.get_window_extent(fig.canvas.get_renderer())
                self._element_bboxes_raw["legend"] = (
                    legend_bbox.x0,
                    legend_bbox.y0,
                    legend_bbox.x1,
                    legend_bbox.y1,
                )
            except Exception:
                pass

        # X axis (bottom spine area)
        try:
            xaxis_bbox = ax.spines["bottom"].get_window_extent(
                fig.canvas.get_renderer()
            )
            # Expand bbox slightly for easier clicking
            self._element_bboxes_raw["xaxis"] = (
                xaxis_bbox.x0,
                xaxis_bbox.y0 - 20,
                xaxis_bbox.x1,
                xaxis_bbox.y1 + 10,
            )
        except Exception:
            pass

        # Y axis (left spine area)
        try:
            yaxis_bbox = ax.spines["left"].get_window_extent(fig.canvas.get_renderer())
            # Expand bbox slightly for easier clicking
            self._element_bboxes_raw["yaxis"] = (
                yaxis_bbox.x0 - 20,
                yaxis_bbox.y0,
                yaxis_bbox.x1 + 10,
                yaxis_bbox.y1,
            )
        except Exception:
            pass

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
        self._preview_bounds = (x_offset, y_offset, new_width, new_height)

        # Scale element bboxes to preview coordinates
        # Note: matplotlib uses bottom-left origin, we need top-left for preview
        self._element_bboxes = {}
        for elem_type, raw_bbox in getattr(self, "_element_bboxes_raw", {}).items():
            if raw_bbox is None:
                continue
            rx0, ry0, rx1, ry1 = raw_bbox
            # Scale to resized image
            sx0 = int(rx0 * ratio) + x_offset
            sx1 = int(rx1 * ratio) + x_offset
            # Flip Y coordinate (matplotlib origin is bottom, preview is top)
            sy0 = new_height - int(ry1 * ratio) + y_offset
            sy1 = new_height - int(ry0 * ratio) + y_offset
            self._element_bboxes[elem_type] = (sx0, sy0, sx1, sy1)

        # Store axes transform info (scaled to resized image)
        # ax_bbox is in figure fraction coordinates
        ax_x0 = int(ax_bbox.x0 * new_width)
        ax_y0 = int((1 - ax_bbox.y1) * new_height)  # Flip y (0 at top)
        ax_width = int(ax_bbox.width * new_width)
        ax_height = int(ax_bbox.height * new_height)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        self._axes_transform = (ax_x0, ax_y0, ax_width, ax_height, xlim, ylim)

        # Create background - checkerboard for transparent, white otherwise
        transparent = o.get("transparent", True)
        if transparent:
            # Create checkerboard pattern for transparency preview
            padded = _create_checkerboard(max_width, max_height, square_size=10)
        else:
            padded = Image.new("RGBA", (max_width, max_height), (255, 255, 255, 255))

        # Paste figure centered on background
        padded.paste(img, (x_offset, y_offset), img)  # Use img as mask for alpha
        img = padded
        width, height = max_width, max_height

        # Cache the base image (without highlights) for fast hover updates
        self._cached_base_image = img.copy()
        self._cache_dirty = False

        # Convert to normalized float array for DearPyGui
        img_array = np.array(img).astype(np.float32) / 255.0
        img_data = img_array.flatten().tolist()

        plt.close(fig)

        # Update texture data (don't recreate texture, just update values)
        dpg.set_value("preview_texture", img_data)

        return img_data, width, height

    def _draw_element_highlights(self, fig, ax):
        """Draw selection highlights for non-trace elements (hover handled via PIL overlay)."""
        from matplotlib.patches import FancyBboxPatch
        import matplotlib.transforms as transforms

        renderer = fig.canvas.get_renderer()

        # Only draw selection highlights here (hover is done via fast PIL overlay)
        selected_type = (
            self._selected_element.get("type") if self._selected_element else None
        )

        # Skip if selecting traces (handled separately in _plot_from_csv)
        if selected_type == "trace":
            selected_type = None

        def add_highlight_box(text_obj, color, alpha, linewidth=2):
            """Add highlight rectangle around a text object (outline only)."""
            try:
                bbox = text_obj.get_window_extent(renderer)
                # Convert to figure coordinates
                fig_bbox = bbox.transformed(fig.transFigure.inverted())
                # Add padding
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
        select_color = "#FFC850"  # Soft warm yellow for outline
        if selected_type in element_map:
            add_highlight_box(
                element_map[selected_type], select_color, 0.0, linewidth=2
            )
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

        # Note: Hover highlights are now drawn via fast PIL overlay in _update_hover_overlay()

    def _plot_from_csv(self, ax, o, highlight_trace=None, hover_trace=None):
        """Reconstruct plot from CSV data using trace info.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to plot on
        o : dict
            Current overrides containing trace info
        highlight_trace : int, optional
            Index of trace to highlight with selection effect (yellow glow)
        hover_trace : int, optional
            Index of trace to highlight with hover effect (cyan glow)
        """
        import pandas as pd
        from ._defaults import _normalize_legend_loc

        if not isinstance(self.csv_data, pd.DataFrame):
            return

        df = self.csv_data
        linewidth = o.get("linewidth", 1.0)
        legend_visible = o.get("legend_visible", True)
        legend_fontsize = o.get("legend_fontsize", 6)
        legend_frameon = o.get("legend_frameon", False)
        legend_loc = _normalize_legend_loc(o.get("legend_loc", "best"))

        traces = o.get("traces", [])

        if traces:
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
                ax.legend(
                    fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc
                )
        else:
            # Fallback: parse column names
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
                    ax.legend(
                        fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc
                    )
            elif len(cols) >= 2:
                x_col = cols[0]
                for y_col in cols[1:]:
                    try:
                        ax.plot(
                            df[x_col], df[y_col], label=str(y_col), linewidth=linewidth
                        )
                    except Exception:
                        pass
                if len(cols) > 2 and legend_visible:
                    ax.legend(
                        fontsize=legend_fontsize, frameon=legend_frameon, loc=legend_loc
                    )

    def _get_trace_labels(self):
        """Get list of trace labels for selection combo."""
        traces = self.current_overrides.get("traces", [])
        if not traces:
            return ["(no traces)"]
        return [t.get("label", t.get("id", f"Trace {i}")) for i, t in enumerate(traces)]

    def _get_all_element_labels(self):
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
        traces = self.current_overrides.get("traces", [])
        for i, t in enumerate(traces):
            label = t.get("label", t.get("id", f"Trace {i}"))
            labels.append(f"Trace: {label}")

        return labels

    def _on_preview_click(self, sender, app_data):
        """Handle click on preview image to select element."""
        import dearpygui.dearpygui as dpg

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
        element = self._find_clicked_element(click_x, click_y)

        if element:
            self._select_element(element, dpg)
        else:
            # Fall back to trace selection
            trace_idx = self._find_nearest_trace(
                click_x, click_y, max_width, max_height
            )
            if trace_idx is not None:
                self._select_element({"type": "trace", "index": trace_idx}, dpg)

    def _on_preview_hover(self, sender, app_data):
        """Handle mouse move for hover effects on preview (optimized with caching)."""
        import dearpygui.dearpygui as dpg
        import time

        # Throttle hover updates - reduced to 16ms (~60fps) since we use fast overlay
        current_time = time.time()
        if current_time - self._last_hover_check < 0.016:
            return
        self._last_hover_check = current_time

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
            if self._hovered_element is not None:
                self._hovered_element = None
                dpg.set_value("hover_text", "")
                # Use fast overlay update instead of full redraw
                self._update_hover_overlay(dpg)
            return

        # Find element under cursor
        element = self._find_clicked_element(hover_x, hover_y)

        if element is None:
            # Check for trace hover
            trace_idx = self._find_nearest_trace(
                hover_x, hover_y, max_width, max_height
            )
            if trace_idx is not None:
                element = {"type": "trace", "index": trace_idx}

        # Check if hover changed
        old_hover = self._hovered_element
        if element != old_hover:
            self._hovered_element = element
            if element:
                elem_type = element.get("type", "")
                elem_idx = element.get("index")
                if elem_type == "trace" and elem_idx is not None:
                    traces = self.current_overrides.get("traces", [])
                    if elem_idx < len(traces):
                        label = traces[elem_idx].get("label", f"Trace {elem_idx}")
                        dpg.set_value("hover_text", f"Hover: {label} (click to select)")
                else:
                    label = elem_type.replace("x", "X ").replace("y", "Y ").title()
                    dpg.set_value("hover_text", f"Hover: {label} (click to select)")
            else:
                dpg.set_value("hover_text", "")

            # Use fast overlay update for hover (no matplotlib re-render)
            self._update_hover_overlay(dpg)

    def _find_clicked_element(self, click_x, click_y):
        """Find which element was clicked based on stored bboxes."""
        if not self._element_bboxes:
            return None

        # Check each element bbox
        for element_type, bbox in self._element_bboxes.items():
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            if x0 <= click_x <= x1 and y0 <= click_y <= y1:
                return {"type": element_type, "index": None}

        return None

    def _select_element(self, element, dpg):
        """Select an element and show appropriate controls."""
        self._selected_element = element
        elem_type = element.get("type")
        elem_idx = element.get("index")

        # Hide all control groups first
        dpg.configure_item("trace_controls_group", show=False)
        dpg.configure_item("text_controls_group", show=False)
        dpg.configure_item("axis_controls_group", show=False)
        dpg.configure_item("legend_controls_group", show=False)

        # Update combo selection
        if elem_type == "trace":
            traces = self.current_overrides.get("traces", [])
            if elem_idx is not None and elem_idx < len(traces):
                trace = traces[elem_idx]
                label = (
                    f"Trace: {trace.get('label', trace.get('id', f'Trace {elem_idx}'))}"
                )
                dpg.set_value("element_selector_combo", label)

                # Show trace controls and populate
                dpg.configure_item("trace_controls_group", show=True)
                self._selected_trace_index = elem_idx
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
                    f"Selected: {trace.get('label', f'Trace {elem_idx}')}",
                )

        elif elem_type in ("title", "xlabel", "ylabel"):
            dpg.set_value(
                "element_selector_combo",
                elem_type.replace("x", "X ").replace("y", "Y ").title(),
            )
            dpg.configure_item("text_controls_group", show=True)

            o = self.current_overrides
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

        elif elem_type in ("xaxis", "yaxis"):
            label = "X Axis" if elem_type == "xaxis" else "Y Axis"
            dpg.set_value("element_selector_combo", label)
            dpg.configure_item("axis_controls_group", show=True)

            o = self.current_overrides
            dpg.set_value("axis_linewidth_slider", o.get("axis_width", 0.2))
            dpg.set_value("axis_tick_length_slider", o.get("tick_length", 0.8))
            dpg.set_value("axis_tick_fontsize_slider", o.get("tick_fontsize", 7))

            if elem_type == "xaxis":
                dpg.set_value(
                    "axis_show_spine_checkbox", not o.get("hide_bottom_spine", False)
                )
            else:
                dpg.set_value(
                    "axis_show_spine_checkbox", not o.get("hide_left_spine", False)
                )

            dpg.set_value("selection_text", f"Selected: {label}")

        elif elem_type == "legend":
            dpg.set_value("element_selector_combo", "Legend")
            dpg.configure_item("legend_controls_group", show=True)

            o = self.current_overrides
            dpg.set_value("legend_visible_edit", o.get("legend_visible", True))
            dpg.set_value("legend_frameon_edit", o.get("legend_frameon", False))
            dpg.set_value("legend_loc_edit", o.get("legend_loc", "best"))
            dpg.set_value("legend_fontsize_edit", o.get("legend_fontsize", 6))

            dpg.set_value("selection_text", "Selected: Legend")

        # Redraw with highlight
        self._update_preview(dpg)

    def _on_element_selected(self, sender, app_data):
        """Handle element selection from combo box."""
        import dearpygui.dearpygui as dpg

        if app_data == "Title":
            self._select_element({"type": "title", "index": None}, dpg)
        elif app_data == "X Label":
            self._select_element({"type": "xlabel", "index": None}, dpg)
        elif app_data == "Y Label":
            self._select_element({"type": "ylabel", "index": None}, dpg)
        elif app_data == "X Axis":
            self._select_element({"type": "xaxis", "index": None}, dpg)
        elif app_data == "Y Axis":
            self._select_element({"type": "yaxis", "index": None}, dpg)
        elif app_data == "Legend":
            self._select_element({"type": "legend", "index": None}, dpg)
        elif app_data.startswith("Trace: "):
            # Find trace index
            trace_label = app_data[7:]  # Remove "Trace: " prefix
            traces = self.current_overrides.get("traces", [])
            for i, t in enumerate(traces):
                if t.get("label", t.get("id", f"Trace {i}")) == trace_label:
                    self._select_element({"type": "trace", "index": i}, dpg)
                    break

    def _on_text_element_change(self, sender, app_data, user_data=None):
        """Handle changes to text element properties."""
        import dearpygui.dearpygui as dpg

        if self._selected_element is None:
            return

        elem_type = self._selected_element.get("type")
        if elem_type not in ("title", "xlabel", "ylabel"):
            return

        text = dpg.get_value("element_text_input")
        fontsize = dpg.get_value("element_fontsize_slider")

        if elem_type == "title":
            self.current_overrides["title"] = text
            self.current_overrides["title_fontsize"] = fontsize
        elif elem_type == "xlabel":
            self.current_overrides["xlabel"] = text
            self.current_overrides["axis_fontsize"] = fontsize
        elif elem_type == "ylabel":
            self.current_overrides["ylabel"] = text
            self.current_overrides["axis_fontsize"] = fontsize

        self._user_modified = True
        self._update_preview(dpg)

    def _on_axis_element_change(self, sender, app_data, user_data=None):
        """Handle changes to axis element properties."""
        import dearpygui.dearpygui as dpg

        if self._selected_element is None:
            return

        elem_type = self._selected_element.get("type")
        if elem_type not in ("xaxis", "yaxis"):
            return

        self.current_overrides["axis_width"] = dpg.get_value("axis_linewidth_slider")
        self.current_overrides["tick_length"] = dpg.get_value("axis_tick_length_slider")
        self.current_overrides["tick_fontsize"] = dpg.get_value(
            "axis_tick_fontsize_slider"
        )

        show_spine = dpg.get_value("axis_show_spine_checkbox")
        if elem_type == "xaxis":
            self.current_overrides["hide_bottom_spine"] = not show_spine
        else:
            self.current_overrides["hide_left_spine"] = not show_spine

        self._user_modified = True
        self._update_preview(dpg)

    def _on_legend_element_change(self, sender, app_data, user_data=None):
        """Handle changes to legend element properties."""
        import dearpygui.dearpygui as dpg

        if self._selected_element is None:
            return

        elem_type = self._selected_element.get("type")
        if elem_type != "legend":
            return

        self.current_overrides["legend_visible"] = dpg.get_value("legend_visible_edit")
        self.current_overrides["legend_frameon"] = dpg.get_value("legend_frameon_edit")
        self.current_overrides["legend_loc"] = dpg.get_value("legend_loc_edit")
        self.current_overrides["legend_fontsize"] = dpg.get_value(
            "legend_fontsize_edit"
        )

        self._user_modified = True
        self._update_preview(dpg)

    def _deselect_element(self, sender=None, app_data=None, user_data=None):
        """Deselect the current element."""
        import dearpygui.dearpygui as dpg

        self._selected_element = None
        self._selected_trace_index = None

        # Hide all control groups
        dpg.configure_item("trace_controls_group", show=False)
        dpg.configure_item("text_controls_group", show=False)
        dpg.configure_item("axis_controls_group", show=False)
        dpg.configure_item("legend_controls_group", show=False)

        dpg.set_value("selection_text", "")
        dpg.set_value("element_selector_combo", "")
        self._update_preview(dpg)

    def _find_nearest_trace(self, click_x, click_y, preview_width, preview_height):
        """Find the nearest trace to the click position."""
        import pandas as pd
        import numpy as np

        if self.csv_data is None or not isinstance(self.csv_data, pd.DataFrame):
            return None

        traces = self.current_overrides.get("traces", [])
        if not traces:
            return None

        # Get preview bounds from last render
        if self._preview_bounds is None:
            return None

        x_offset, y_offset, fig_width, fig_height = self._preview_bounds

        # Adjust click coordinates to figure space
        fig_x = click_x - x_offset
        fig_y = click_y - y_offset

        # Check if click is within figure bounds
        if not (0 <= fig_x <= fig_width and 0 <= fig_y <= fig_height):
            return None

        # Get axes transform info
        if self._axes_transform is None:
            return None

        ax_x0, ax_y0, ax_width, ax_height, xlim, ylim = self._axes_transform

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
        df = self.csv_data
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

    def _on_trace_property_change(self, sender, app_data, user_data=None):
        """Handle changes to selected trace properties."""
        import dearpygui.dearpygui as dpg

        if self._selected_trace_index is None:
            return

        traces = self.current_overrides.get("traces", [])
        if self._selected_trace_index >= len(traces):
            return

        trace = traces[self._selected_trace_index]

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

        self._user_modified = True
        self._update_preview(dpg)

    def _save_manual(self, sender=None, app_data=None, user_data=None):
        """Save current overrides to .manual.json."""
        import dearpygui.dearpygui as dpg
        from .edit import save_manual_overrides

        try:
            self._collect_overrides(dpg)
            manual_path = save_manual_overrides(self.json_path, self.current_overrides)
            dpg.set_value("status_text", f"Saved: {manual_path.name}")
        except Exception as e:
            dpg.set_value("status_text", f"Error: {str(e)}")

    def _reset_overrides(self, sender=None, app_data=None, user_data=None):
        """Reset to initial overrides."""
        import dearpygui.dearpygui as dpg

        self.current_overrides = copy.deepcopy(self._initial_overrides)
        self._user_modified = False

        # Update all widgets
        dpg.set_value("title_input", self.current_overrides.get("title", ""))
        dpg.set_value("xlabel_input", self.current_overrides.get("xlabel", ""))
        dpg.set_value("ylabel_input", self.current_overrides.get("ylabel", ""))
        dpg.set_value("linewidth_slider", self.current_overrides.get("linewidth", 1.0))
        dpg.set_value("grid_checkbox", self.current_overrides.get("grid", False))
        dpg.set_value(
            "transparent_checkbox", self.current_overrides.get("transparent", True)
        )

        self._update_preview(dpg)
        dpg.set_value("status_text", "Reset to original")

    def _export_png(self, sender=None, app_data=None, user_data=None):
        """Export current view to PNG."""
        import dearpygui.dearpygui as dpg
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        try:
            self._collect_overrides(dpg)
            output_path = self.json_path.with_suffix(".edited.png")

            # Full resolution render
            o = self.current_overrides
            fig_size = o.get("fig_size", [3.15, 2.68])
            dpi = o.get("dpi", 300)

            fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

            if self.csv_data is not None:
                self._plot_from_csv(ax, o)

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
