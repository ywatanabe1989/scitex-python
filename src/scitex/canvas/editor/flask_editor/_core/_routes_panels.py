#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_routes_panels.py

"""Panel-related Flask routes for the editor."""

import base64
import copy
import json as json_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._core import WebEditor

__all__ = [
    "create_panels_route",
    "create_switch_panel_route",
]


def create_panels_route(app, editor: "WebEditor"):
    """Create the panels route for multi-panel figure bundles."""
    from flask import jsonify

    from .._bbox import (
        extract_bboxes_from_geometry_px,
        extract_bboxes_from_metadata,
    )
    from ..edit import load_panel_data

    @app.route("/panels")
    def panels():
        if not editor.panel_info:
            return jsonify({"error": "Not a multi-panel figure bundle"}), 400

        panel_names = editor.panel_info["panels"]
        panel_paths = editor.panel_info.get("panel_paths", [])
        panel_is_zip = editor.panel_info.get("panel_is_zip", [False] * len(panel_names))
        figure_dir = Path(editor.panel_info["figure_dir"])

        if not panel_paths:
            panel_paths = [str(figure_dir / name) for name in panel_names]

        # Load figz spec.json for panel layout
        figure_layout = {}
        spec_path = figure_dir / "spec.json"
        if spec_path.exists():
            with open(spec_path) as f:
                figure_spec = json_module.load(f)
                for panel_spec in figure_spec.get("panels", []):
                    panel_id = panel_spec.get("id", "")
                    figure_layout[panel_id] = {
                        "position": panel_spec.get("position", {}),
                        "size": panel_spec.get("size", {}),
                    }

        panel_images = []

        for idx, panel_name in enumerate(panel_names):
            panel_path = panel_paths[idx]
            is_zip = panel_is_zip[idx] if idx < len(panel_is_zip) else None
            display_name = panel_name.replace(".plot", "").replace(".plot", "")

            loaded = load_panel_data(panel_path, is_zip=is_zip)

            panel_data = {
                "name": display_name,
                "image": None,
                "bboxes": None,
                "img_size": None,
            }

            if display_name in figure_layout:
                panel_data["layout"] = figure_layout[display_name]

            if loaded:
                # Get image data
                if loaded.get("is_zip"):
                    png_bytes = loaded.get("png_bytes")
                    if png_bytes:
                        panel_data["image"] = base64.b64encode(png_bytes).decode(
                            "utf-8"
                        )
                else:
                    png_path = loaded.get("png_path")
                    if png_path and png_path.exists():
                        with open(png_path, "rb") as f:
                            panel_data["image"] = base64.b64encode(f.read()).decode(
                                "utf-8"
                            )

                # Get image size
                img_size = loaded.get("img_size")
                if img_size:
                    panel_data["img_size"] = img_size
                    panel_data["width"] = img_size["width"]
                    panel_data["height"] = img_size["height"]
                elif loaded.get("png_path"):
                    from PIL import Image

                    img = Image.open(loaded["png_path"])
                    panel_data["img_size"] = {
                        "width": img.size[0],
                        "height": img.size[1],
                    }
                    panel_data["width"], panel_data["height"] = img.size
                    img.close()

                # Extract bboxes
                if panel_data.get("img_size"):
                    geometry_data = loaded.get("geometry_data")
                    metadata = loaded.get("metadata", {})

                    if geometry_data:
                        panel_data["bboxes"] = extract_bboxes_from_geometry_px(
                            geometry_data,
                            panel_data["img_size"]["width"],
                            panel_data["img_size"]["height"],
                        )
                    elif metadata:
                        panel_data["bboxes"] = extract_bboxes_from_metadata(
                            metadata,
                            panel_data["img_size"]["width"],
                            panel_data["img_size"]["height"],
                        )

            panel_images.append(panel_data)

        return jsonify(
            {
                "panels": panel_images,
                "count": len(panel_images),
                "layout": figure_layout,
            }
        )

    return panels


def create_switch_panel_route(app, editor: "WebEditor"):
    """Create the switch_panel route."""
    from flask import jsonify

    from .._bbox import (
        extract_bboxes_from_geometry_px,
        extract_bboxes_from_metadata,
    )
    from ..edit import load_panel_data

    @app.route("/switch_panel/<int:panel_index>")
    def switch_panel(panel_index):
        if not editor.panel_info:
            return jsonify({"error": "Not a multi-panel figure bundle"}), 400

        panels = editor.panel_info["panels"]
        panel_paths = editor.panel_info.get("panel_paths", [])
        panel_is_zip = editor.panel_info.get("panel_is_zip", [False] * len(panels))

        if panel_index < 0 or panel_index >= len(panels):
            return jsonify({"error": f"Invalid panel index: {panel_index}"}), 400

        panel_name = panels[panel_index]
        panel_path = (
            panel_paths[panel_index]
            if panel_paths
            else str(Path(editor.panel_info["figure_dir"]) / panel_name)
        )
        is_zip = panel_is_zip[panel_index] if panel_index < len(panel_is_zip) else None

        try:
            loaded = load_panel_data(panel_path, is_zip=is_zip)

            if not loaded:
                return jsonify({"error": f"Could not load panel: {panel_name}"}), 400

            # Get image data
            img_data = None
            if loaded.get("is_zip"):
                png_bytes = loaded.get("png_bytes")
                if png_bytes:
                    img_data = base64.b64encode(png_bytes).decode("utf-8")
            else:
                png_path = loaded.get("png_path")
                if png_path and png_path.exists():
                    with open(png_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")

            if not img_data:
                return jsonify({"error": f"No PNG found for panel: {panel_name}"}), 400

            # Get image size
            img_size = loaded.get("img_size", {"width": 0, "height": 0})
            if not img_size and loaded.get("png_path"):
                from PIL import Image

                img = Image.open(loaded["png_path"])
                img_size = {"width": img.size[0], "height": img.size[1]}
                img.close()

            # Extract bboxes
            bboxes = {}
            geometry_data = loaded.get("geometry_data")
            metadata = loaded.get("metadata", {})

            if geometry_data and img_size:
                bboxes = extract_bboxes_from_geometry_px(
                    geometry_data, img_size["width"], img_size["height"]
                )
            elif metadata and img_size:
                bboxes = extract_bboxes_from_metadata(
                    metadata, img_size["width"], img_size["height"]
                )

            # Update editor state
            editor.metadata = metadata
            editor.panel_info["current_index"] = panel_index

            # Re-extract defaults
            from ..._defaults import extract_defaults_from_metadata, get_scitex_defaults

            editor.scitex_defaults = get_scitex_defaults()
            editor.metadata_defaults = extract_defaults_from_metadata(metadata)
            editor.current_overrides = copy.deepcopy(editor.scitex_defaults)
            editor.current_overrides.update(editor.metadata_defaults)
            editor.current_overrides.update(editor.manual_overrides)

            return jsonify(
                {
                    "success": True,
                    "panel_name": panel_name,
                    "panel_index": panel_index,
                    "image": img_data,
                    "bboxes": bboxes,
                    "img_size": img_size,
                    "overrides": editor.current_overrides,
                }
            )
        except Exception as e:
            import traceback

            return jsonify(
                {
                    "error": f"Failed to switch panel: {str(e)}",
                    "traceback": traceback.format_exc(),
                }
            ), 500

    return switch_panel


# EOF
