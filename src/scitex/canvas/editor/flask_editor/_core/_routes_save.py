#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_routes_save.py

"""Save-related Flask routes for the editor."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._core import WebEditor

__all__ = [
    "create_save_route",
    "create_save_layout_route",
    "create_save_element_position_route",
]


def create_save_route(app, editor: "WebEditor"):
    """Create the save route."""
    from flask import jsonify

    from ..edit import save_manual_overrides

    @app.route("/save", methods=["POST"])
    def save():
        try:
            manual_path = save_manual_overrides(
                editor.json_path, editor.current_overrides
            )
            return jsonify({"status": "saved", "path": str(manual_path)})
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return save


def create_save_layout_route(app, editor: "WebEditor"):
    """Create the save_layout route."""
    from flask import jsonify, request

    from ._export_helpers import export_composed_figure

    @app.route("/save_layout", methods=["POST"])
    def save_layout():
        try:
            data = request.get_json()
            layout = data.get("layout", {})

            if not layout:
                return jsonify({"success": False, "error": "No layout data provided"})

            if not editor.panel_info:
                return jsonify(
                    {
                        "success": False,
                        "error": "No panel info available (not a figure bundle)",
                    }
                )

            bundle_path = editor.panel_info.get("bundle_path")
            if not bundle_path:
                return jsonify({"success": False, "error": "Bundle path not available"})

            from scitex.canvas.io import ZipBundle

            bundle = ZipBundle(bundle_path)

            # Read existing layout or create new one
            try:
                existing_layout = bundle.read_json("layout.json")
            except:
                existing_layout = {}

            # Update layout with new positions
            for panel_name, pos in layout.items():
                if panel_name not in existing_layout:
                    existing_layout[panel_name] = {}
                if "position" not in existing_layout[panel_name]:
                    existing_layout[panel_name]["position"] = {}
                if "size" not in existing_layout[panel_name]:
                    existing_layout[panel_name]["size"] = {}

                existing_layout[panel_name]["position"]["x_mm"] = pos.get("x_mm", 0)
                existing_layout[panel_name]["position"]["y_mm"] = pos.get("y_mm", 0)

                if "width_mm" in pos:
                    existing_layout[panel_name]["size"]["width_mm"] = pos["width_mm"]
                if "height_mm" in pos:
                    existing_layout[panel_name]["size"]["height_mm"] = pos["height_mm"]

            bundle.write_json("layout.json", existing_layout)
            editor.panel_info["layout"] = existing_layout

            # Auto-export composed figure
            export_result = export_composed_figure(editor, formats=["png", "svg"])

            return jsonify(
                {
                    "success": True,
                    "layout": existing_layout,
                    "exported": export_result.get("exported", {}),
                }
            )

        except Exception as e:
            import traceback

            return jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return save_layout


def create_save_element_position_route(app, editor: "WebEditor"):
    """Create the save_element_position route.

    ONLY legends and panel letters can be repositioned to maintain
    scientific rigor. Data elements are never moved.
    """
    from flask import jsonify, request

    @app.route("/save_element_position", methods=["POST"])
    def save_element_position():
        try:
            data = request.get_json()
            element = data.get("element", "")
            panel = data.get("panel", "")
            element_type = data.get("element_type", "")
            position = data.get("position", {})
            snap_name = data.get("snap_name")

            # Validate element type (whitelist for scientific rigor)
            ALLOWED_TYPES = ["legend", "panel_letter"]
            if element_type not in ALLOWED_TYPES:
                return jsonify(
                    {
                        "success": False,
                        "error": f"Element type '{element_type}' cannot be repositioned (scientific rigor)",
                    }
                )

            if not editor.panel_info:
                return jsonify({"success": False, "error": "No panel info available"})

            bundle_path = editor.panel_info.get("bundle_path")
            if not bundle_path:
                return jsonify({"success": False, "error": "Bundle path not available"})

            from scitex.canvas.io import ZipBundle

            bundle = ZipBundle(bundle_path)

            try:
                style = bundle.read_json("style.json")
            except:
                style = {}

            if "elements" not in style:
                style["elements"] = {}
            if panel not in style["elements"]:
                style["elements"][panel] = {}

            style["elements"][panel][element] = {
                "type": element_type,
                "position": position,
                "snap_name": snap_name,
            }

            # For legends, update legend_location for matplotlib compatibility
            if element_type == "legend" and snap_name:
                loc_map = {
                    "upper left": "upper left",
                    "upper center": "upper center",
                    "upper right": "upper right",
                    "center left": "center left",
                    "center": "center",
                    "center right": "center right",
                    "lower left": "lower left",
                    "lower center": "lower center",
                    "lower right": "lower right",
                }
                if snap_name in loc_map:
                    if "legend" not in style:
                        style["legend"] = {}
                    style["legend"]["location"] = loc_map[snap_name]

            bundle.write_json("style.json", style)

            return jsonify(
                {
                    "success": True,
                    "element": element,
                    "position": position,
                    "snap_name": snap_name,
                }
            )

        except Exception as e:
            import traceback

            return jsonify(
                {
                    "success": False,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            )

    return save_element_position


# EOF
