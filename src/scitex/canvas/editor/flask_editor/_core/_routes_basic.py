#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_routes_basic.py

"""Basic Flask routes for the editor."""

import base64
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._core import WebEditor

__all__ = [
    "create_index_route",
    "create_preview_route",
    "create_hitmap_route",
    "create_colormap_route",
    "create_update_route",
    "create_stats_route",
    "create_shutdown_route",
]


def create_index_route(app, editor: "WebEditor"):
    """Create the index route."""
    from flask import render_template_string

    from ..templates import build_html_template

    @app.route("/")
    def index():
        html_template = build_html_template()
        json_path_str = str(editor.json_path.resolve())
        figure_path = ""
        panel_path = ""

        if ".figure/" in json_path_str:
            parts = json_path_str.split(".figure/")
            figure_path = parts[0] + ".figure"
            panel_path = parts[1] if len(parts) > 1 else ""
        elif ".plot/" in json_path_str:
            parts = json_path_str.split(".plot/")
            figure_path = parts[0] + ".plot"
            panel_path = parts[1] if len(parts) > 1 else ""
        else:
            figure_path = json_path_str

        return render_template_string(
            html_template,
            filename=figure_path,
            panel_path=panel_path,
            overrides=json.dumps(editor.current_overrides),
        )

    return index


def create_preview_route(app, editor: "WebEditor"):
    """Create the preview route."""
    from flask import jsonify, request

    from .._renderer import render_preview_with_bboxes

    @app.route("/preview")
    def preview():
        dark_mode = request.args.get("dark_mode", "false").lower() == "true"
        img_data, bboxes, img_size = render_preview_with_bboxes(
            editor.csv_data,
            editor.current_overrides,
            metadata=editor.metadata,
            dark_mode=dark_mode,
        )
        return jsonify(
            {
                "image": img_data,
                "bboxes": bboxes,
                "img_size": img_size,
                "has_hitmap": editor.hitmap_path is not None
                and editor.hitmap_path.exists(),
                "format": "png",
                "panel_info": editor.panel_info,
            }
        )

    return preview


def create_hitmap_route(app, editor: "WebEditor"):
    """Create the hitmap route."""
    from flask import jsonify

    @app.route("/hitmap")
    def hitmap():
        if editor.hitmap_path and editor.hitmap_path.exists():
            with open(editor.hitmap_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            return jsonify(
                {
                    "image": img_data,
                    "color_map": editor.color_map,
                }
            )
        return jsonify({"error": "No hitmap available"}), 404

    return hitmap


def create_colormap_route(app, editor: "WebEditor"):
    """Create the color_map route."""
    from flask import jsonify

    @app.route("/color_map")
    def color_map():
        return jsonify(
            {
                "color_map": editor.color_map,
                "hit_regions": editor.hit_regions,
            }
        )

    return color_map


def create_update_route(app, editor: "WebEditor"):
    """Create the update route."""
    from flask import jsonify, request

    from .._renderer import render_preview_with_bboxes

    @app.route("/update", methods=["POST"])
    def update():
        data = request.json
        editor.current_overrides.update(data.get("overrides", {}))
        editor._user_modified = True
        dark_mode = data.get("dark_mode", False)

        img_data, bboxes, img_size = render_preview_with_bboxes(
            editor.csv_data,
            editor.current_overrides,
            metadata=editor.metadata,
            dark_mode=dark_mode,
        )
        return jsonify(
            {
                "image": img_data,
                "bboxes": bboxes,
                "img_size": img_size,
                "status": "updated",
            }
        )

    return update


def create_stats_route(app, editor: "WebEditor"):
    """Create the stats route."""
    from flask import jsonify

    @app.route("/stats")
    def stats():
        stats_data = editor.metadata.get("stats", [])
        stats_summary = editor.metadata.get("stats_summary", None)
        return jsonify(
            {
                "stats": stats_data,
                "stats_summary": stats_summary,
                "has_stats": len(stats_data) > 0,
            }
        )

    return stats


def create_shutdown_route(app, editor: "WebEditor"):
    """Create the shutdown route."""
    from flask import jsonify, request

    @app.route("/shutdown", methods=["POST"])
    def shutdown():
        func = request.environ.get("werkzeug.server.shutdown")
        if func is None:
            raise RuntimeError("Not running with Werkzeug Server")
        func()
        return jsonify({"status": "shutdown"})

    return shutdown


# EOF
