#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/core.py
"""Core WebEditor class for Flask-based figure editing."""

from pathlib import Path
from typing import Dict, Any, Optional
import base64
import copy
import json
import threading
import webbrowser

from ._utils import find_available_port, kill_process_on_port, check_port_available
from .templates import build_html_template


class WebEditor:
    """
    Browser-based figure editor using Flask.

    Features:
    - Displays existing PNG from pltz bundle (no re-rendering)
    - Hitmap-based element selection for precise clicking
    - Property editors with sliders and color pickers
    - Save to .manual.json
    - SciTeX style defaults pre-filled
    - Auto-finds available port if default is in use
    """

    def __init__(
        self,
        json_path: Path,
        metadata: Dict[str, Any],
        csv_data: Optional[Any] = None,
        png_path: Optional[Path] = None,
        hitmap_path: Optional[Path] = None,
        manual_overrides: Optional[Dict[str, Any]] = None,
        port: int = 5050,
        panel_info: Optional[Dict[str, Any]] = None,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.png_path = Path(png_path) if png_path else None
        self.hitmap_path = Path(hitmap_path) if hitmap_path else None
        self.manual_overrides = manual_overrides or {}
        self._requested_port = port
        self.port = port
        self.panel_info = panel_info  # For multi-panel figz bundles

        # Extract hit_regions from metadata for color-based element detection
        self.hit_regions = metadata.get("hit_regions", {})
        self.color_map = self.hit_regions.get("color_map", {})

        # Get SciTeX defaults and merge with metadata
        from .._defaults import get_scitex_defaults, extract_defaults_from_metadata

        self.scitex_defaults = get_scitex_defaults()
        self.metadata_defaults = extract_defaults_from_metadata(metadata)

        # Start with defaults, then overlay manual overrides
        self.current_overrides = copy.deepcopy(self.scitex_defaults)
        self.current_overrides.update(self.metadata_defaults)
        self.current_overrides.update(self.manual_overrides)

        # Track initial state to detect modifications
        self._initial_overrides = copy.deepcopy(self.current_overrides)
        self._user_modified = False

    def run(self):
        """Launch the web editor."""
        try:
            from flask import Flask, render_template_string, request, jsonify
        except ImportError:
            raise ImportError(
                "Flask is required for web editor. Install: pip install flask"
            )

        # Handle port conflicts - always use port 5050
        import time
        max_retries = 3
        for attempt in range(max_retries):
            if check_port_available(self._requested_port):
                self.port = self._requested_port
                break
            print(f"Port {self._requested_port} in use. Freeing... (attempt {attempt + 1}/{max_retries})")
            kill_process_on_port(self._requested_port)
            time.sleep(1.0)  # Wait for port release
        else:
            # After retries, use requested port anyway (Flask will error if unavailable)
            print(f"Warning: Port {self._requested_port} may still be in use")
            self.port = self._requested_port

        app = Flask(__name__)
        editor = self

        @app.route("/")
        def index():
            # Rebuild template each time for hot reload support
            html_template = build_html_template()

            # Extract figz and panel paths for display
            json_path_str = str(editor.json_path.resolve())
            figz_path = ""
            panel_path = ""

            # Check if this is inside a figz bundle
            if '.figz.d/' in json_path_str:
                parts = json_path_str.split('.figz.d/')
                figz_path = parts[0] + '.figz.d'
                panel_path = parts[1] if len(parts) > 1 else ""
            elif '.pltz.d/' in json_path_str:
                parts = json_path_str.split('.pltz.d/')
                figz_path = parts[0] + '.pltz.d'
                panel_path = parts[1] if len(parts) > 1 else ""
            else:
                figz_path = json_path_str

            return render_template_string(
                html_template,
                filename=figz_path,
                panel_path=panel_path,
                overrides=json.dumps(editor.current_overrides),
            )

        @app.route("/preview")
        def preview():
            """Render figure preview with current overrides (same logic as /update)."""
            from ._renderer import render_preview_with_bboxes

            # Always use renderer for consistency between initial and updated views
            dark_mode = request.args.get("dark_mode", "false").lower() == "true"
            img_data, bboxes, img_size = render_preview_with_bboxes(
                editor.csv_data, editor.current_overrides,
                metadata=editor.metadata,
                dark_mode=dark_mode,
            )
            return jsonify({
                "image": img_data,
                "bboxes": bboxes,
                "img_size": img_size,
                "has_hitmap": editor.hitmap_path is not None and editor.hitmap_path.exists(),
                "format": "png",
                "panel_info": editor.panel_info,
            })

        @app.route("/panels")
        def panels():
            """Return all panel images with bboxes for interactive grid view (figz bundles only)."""
            from PIL import Image
            from ._bbox import extract_bboxes_from_metadata, extract_bboxes_from_geometry_px

            if not editor.panel_info:
                return jsonify({"error": "Not a multi-panel figz bundle"}), 400

            figz_dir = Path(editor.panel_info["figz_dir"])
            panel_names = editor.panel_info["panels"]
            panel_images = []

            for panel_name in panel_names:
                panel_dir = figz_dir / panel_name
                panel_data = {"name": panel_name.replace(".pltz.d", ""), "image": None, "bboxes": None, "img_size": None}

                # Find PNG in exports/ or root
                png_path = None
                exports_dir = panel_dir / "exports"
                if exports_dir.exists():
                    for f in exports_dir.glob("*.png"):
                        if "_hitmap" not in f.name and "_overview" not in f.name:
                            png_path = f
                            break
                if not png_path:
                    for f in panel_dir.glob("*.png"):
                        if "_hitmap" not in f.name and "_overview" not in f.name:
                            png_path = f
                            break

                if png_path and png_path.exists():
                    with open(png_path, "rb") as f:
                        panel_data["image"] = base64.b64encode(f.read()).decode("utf-8")
                    img = Image.open(png_path)
                    panel_data["width"], panel_data["height"] = img.size
                    panel_data["img_size"] = {"width": img.size[0], "height": img.size[1]}
                    img.close()

                    # Try to load geometry_px.json from cache (has precise pixel coordinates)
                    geometry_path = panel_dir / "cache" / "geometry_px.json"
                    if geometry_path.exists():
                        import json
                        with open(geometry_path) as f:
                            geometry_data = json.load(f)
                        panel_data["bboxes"] = extract_bboxes_from_geometry_px(
                            geometry_data,
                            panel_data["img_size"]["width"],
                            panel_data["img_size"]["height"]
                        )
                    else:
                        # Fall back to spec.json extraction
                        spec_path = panel_dir / "spec.json"
                        if spec_path.exists():
                            import json
                            with open(spec_path) as f:
                                panel_metadata = json.load(f)
                            panel_data["bboxes"] = extract_bboxes_from_metadata(
                                panel_metadata,
                                panel_data["img_size"]["width"],
                                panel_data["img_size"]["height"]
                            )

                panel_images.append(panel_data)

            return jsonify({
                "panels": panel_images,
                "count": len(panel_images),
            })

        @app.route("/switch_panel/<int:panel_index>")
        def switch_panel(panel_index):
            """Switch to a different panel in the figz bundle.

            Loads the actual PNG from the panel's exports folder instead of re-rendering.
            """
            from PIL import Image
            from .._edit import _load_panel_data
            from ._bbox import extract_bboxes_from_metadata, extract_bboxes_from_geometry_px

            if not editor.panel_info:
                return jsonify({"error": "Not a multi-panel figz bundle"}), 400

            panels = editor.panel_info["panels"]
            if panel_index < 0 or panel_index >= len(panels):
                return jsonify({"error": f"Invalid panel index: {panel_index}"}), 400

            figz_dir = Path(editor.panel_info["figz_dir"])
            panel_name = panels[panel_index]
            panel_dir = figz_dir / panel_name

            # Load the panel's data
            try:
                panel_data = _load_panel_data(panel_dir)
                if not panel_data:
                    return jsonify({"error": f"Could not load panel: {panel_name}"}), 400

                # Update editor state
                editor.json_path = panel_data["json_path"]
                editor.metadata = panel_data["metadata"]
                editor.csv_data = panel_data.get("csv_data")
                editor.png_path = panel_data.get("png_path")
                editor.hitmap_path = panel_data.get("hitmap_path")
                editor.panel_info["current_index"] = panel_index

                # Re-extract defaults from new metadata
                from .._defaults import get_scitex_defaults, extract_defaults_from_metadata
                editor.scitex_defaults = get_scitex_defaults()
                editor.metadata_defaults = extract_defaults_from_metadata(editor.metadata)
                editor.current_overrides = copy.deepcopy(editor.scitex_defaults)
                editor.current_overrides.update(editor.metadata_defaults)
                editor.current_overrides.update(editor.manual_overrides)

                # Load actual PNG from panel instead of re-rendering
                img_data = None
                img_size = {"width": 0, "height": 0}
                png_path = panel_data.get("png_path")

                if png_path and png_path.exists():
                    with open(png_path, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")
                    img = Image.open(png_path)
                    img_size = {"width": img.size[0], "height": img.size[1]}
                    img.close()
                else:
                    # Fallback: look for any PNG in exports/
                    exports_dir = panel_dir / "exports"
                    if exports_dir.exists():
                        for f in exports_dir.glob("*.png"):
                            if "_hitmap" not in f.name and "_overview" not in f.name:
                                with open(f, "rb") as pf:
                                    img_data = base64.b64encode(pf.read()).decode("utf-8")
                                img = Image.open(f)
                                img_size = {"width": img.size[0], "height": img.size[1]}
                                img.close()
                                break

                if not img_data:
                    return jsonify({"error": f"No PNG found for panel: {panel_name}"}), 400

                # Extract bboxes - prefer geometry_px.json for precise coordinates
                bboxes = {}
                geometry_path = panel_dir / "cache" / "geometry_px.json"
                if geometry_path.exists():
                    with open(geometry_path) as f:
                        geometry_data = json.load(f)
                    bboxes = extract_bboxes_from_geometry_px(
                        geometry_data,
                        img_size["width"],
                        img_size["height"],
                    )
                else:
                    # Fall back to metadata extraction
                    bboxes = extract_bboxes_from_metadata(
                        editor.metadata,
                        img_size["width"],
                        img_size["height"],
                    )

                return jsonify({
                    "success": True,
                    "panel_name": panel_name,
                    "panel_index": panel_index,
                    "image": img_data,
                    "bboxes": bboxes,
                    "img_size": img_size,
                    "overrides": editor.current_overrides,
                })
            except Exception as e:
                import traceback
                return jsonify({
                    "error": f"Failed to switch panel: {str(e)}",
                    "traceback": traceback.format_exc(),
                }), 500

        @app.route("/hitmap")
        def hitmap():
            """Return hitmap PNG for element detection."""
            if editor.hitmap_path and editor.hitmap_path.exists():
                with open(editor.hitmap_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")
                return jsonify({
                    "image": img_data,
                    "color_map": editor.color_map,
                })
            return jsonify({"error": "No hitmap available"}), 404

        @app.route("/color_map")
        def color_map():
            """Return color map for hitmap element identification."""
            return jsonify({
                "color_map": editor.color_map,
                "hit_regions": editor.hit_regions,
            })

        @app.route("/update", methods=["POST"])
        def update():
            """Update overrides and re-render with updated properties."""
            from ._renderer import render_preview_with_bboxes

            data = request.json
            editor.current_overrides.update(data.get("overrides", {}))
            editor._user_modified = True

            # Check if dark mode is requested from POST data
            dark_mode = data.get("dark_mode", False)

            # Re-render the figure with updated overrides
            img_data, bboxes, img_size = render_preview_with_bboxes(
                editor.csv_data, editor.current_overrides,
                metadata=editor.metadata,
                dark_mode=dark_mode,
            )
            return jsonify({
                "image": img_data,
                "bboxes": bboxes,
                "img_size": img_size,
                "status": "updated",
            })

        @app.route("/save", methods=["POST"])
        def save():
            """Save to .manual.json."""
            from .._edit import save_manual_overrides

            try:
                manual_path = save_manual_overrides(
                    editor.json_path, editor.current_overrides
                )
                return jsonify({"status": "saved", "path": str(manual_path)})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @app.route("/shutdown", methods=["POST"])
        def shutdown():
            """Shutdown the server."""
            func = request.environ.get("werkzeug.server.shutdown")
            if func is None:
                raise RuntimeError("Not running with Werkzeug Server")
            func()
            return jsonify({"status": "shutdown"})

        @app.route("/stats")
        def stats():
            """Return statistical test results from figure metadata."""
            stats_data = editor.metadata.get("stats", [])
            stats_summary = editor.metadata.get("stats_summary", None)
            return jsonify({
                "stats": stats_data,
                "stats_summary": stats_summary,
                "has_stats": len(stats_data) > 0,
            })

        # Open browser after short delay
        def open_browser():
            import time

            time.sleep(0.5)
            webbrowser.open(f"http://127.0.0.1:{self.port}")

        threading.Thread(target=open_browser, daemon=True).start()

        print(f"Starting SciTeX Figure Editor at http://127.0.0.1:{self.port}")
        print("Press Ctrl+C to stop")

        # Note: use_reloader=False because the reloader re-runs the entire script
        # which causes infinite loops when the demo generates figures
        # Templates are rebuilt on each page refresh anyway
        app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)


def _extract_bboxes_from_metadata(
    metadata: Dict[str, Any],
    display_width: Optional[float] = None,
    display_height: Optional[float] = None
) -> Dict[str, Any]:
    """Extract element bounding boxes from pltz metadata.

    Builds bboxes from selectable_regions in the metadata for click detection.
    This allows the editor to highlight elements when clicked.

    Coordinate system (new layered format):
    - selectable_regions bbox_px: Already in final image space (figure_px)
    - Display size: Actual displayed image size (PNG pixels or SVG viewBox)
    - Scale = display_size / figure_px (usually 1:1, but may differ for scaled display)

    Parameters
    ----------
    metadata : dict
        The pltz JSON metadata containing selectable_regions
    display_width : float, optional
        Actual display image width (from PNG size or SVG viewBox)
    display_height : float, optional
        Actual display image height (from PNG size or SVG viewBox)

    Returns
    -------
    dict
        Mapping of element IDs to their bounding box coordinates (in display pixels)
    """
    bboxes = {}
    selectable = metadata.get("selectable_regions", {})

    # Figure dimensions from new layered format (bbox_px are in this space)
    figure_px = metadata.get("figure_px", [])
    if isinstance(figure_px, list) and len(figure_px) >= 2:
        fig_width = figure_px[0]
        fig_height = figure_px[1]
    else:
        # Fallback for old format: try hit_regions.path_data.figure
        hit_regions = metadata.get("hit_regions", {})
        path_data = hit_regions.get("path_data", {})
        orig_fig = path_data.get("figure", {})
        fig_width = orig_fig.get("width_px", 944)
        fig_height = orig_fig.get("height_px", 803)

    # Use actual display dimensions if provided, else use figure_px
    if display_width is None:
        display_width = fig_width
    if display_height is None:
        display_height = fig_height

    # Scale factor: display / figure_px
    # Usually 1:1 since display is the same PNG, but may differ for scaled display
    scale_x = display_width / fig_width if fig_width > 0 else 1
    scale_y = display_height / fig_height if fig_height > 0 else 1

    # Helper to convert coords to display pixels
    def to_display_bbox(bbox, is_list=True):
        """Convert bbox to display pixels (apply scaling if display != figure_px).

        Parameters
        ----------
        bbox : list or dict
            Bbox coordinates [x0, y0, x1, y1] or dict with keys
        is_list : bool
            Whether bbox is a list (True) or dict (False)
        """
        if is_list:
            x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        else:
            x0 = bbox.get("x0", 0)
            y0 = bbox.get("y0", 0)
            x1 = bbox.get("x1", bbox.get("x0", 0) + bbox.get("width", 0))
            y1 = bbox.get("y1", bbox.get("y0", 0) + bbox.get("height", 0))

        # Scale to display coords (usually 1:1)
        disp_x0 = x0 * scale_x
        disp_x1 = x1 * scale_x
        disp_y0 = y0 * scale_y
        disp_y1 = y1 * scale_y

        return {
            "x0": disp_x0,
            "y0": disp_y0,
            "x1": disp_x1,
            "y1": disp_y1,
            "x": disp_x0,
            "y": disp_y0,
            "width": disp_x1 - disp_x0,
            "height": disp_y1 - disp_y0,
        }

    # Extract from selectable_regions.axes
    axes_regions = selectable.get("axes", [])
    for ax_idx, ax in enumerate(axes_regions):
        ax_key = f"ax_{ax_idx:02d}"

        # Title
        title = ax.get("title", {})
        if title and "bbox_px" in title:
            bbox_disp = to_display_bbox(title["bbox_px"])
            bboxes[f"{ax_key}_title"] = {
                **bbox_disp,
                "type": "title",
                "text": title.get("text", ""),
            }

        # X label
        xlabel = ax.get("xlabel", {})
        if xlabel and "bbox_px" in xlabel:
            bbox_disp = to_display_bbox(xlabel["bbox_px"])
            bboxes[f"{ax_key}_xlabel"] = {
                **bbox_disp,
                "type": "xlabel",
                "text": xlabel.get("text", ""),
            }

        # Y label
        ylabel = ax.get("ylabel", {})
        if ylabel and "bbox_px" in ylabel:
            bbox_disp = to_display_bbox(ylabel["bbox_px"])
            bboxes[f"{ax_key}_ylabel"] = {
                **bbox_disp,
                "type": "ylabel",
                "text": ylabel.get("text", ""),
            }

        # Legend
        legend = ax.get("legend", {})
        if legend and "bbox_px" in legend:
            bbox_disp = to_display_bbox(legend["bbox_px"])
            bboxes[f"{ax_key}_legend"] = {
                **bbox_disp,
                "type": "legend",
            }

        # X-axis spine
        xaxis = ax.get("xaxis", {})
        if xaxis:
            spine = xaxis.get("spine", {})
            if spine and "bbox_px" in spine:
                bbox_disp = to_display_bbox(spine["bbox_px"])
                bboxes[f"{ax_key}_xaxis_spine"] = {
                    **bbox_disp,
                    "type": "xaxis",
                }

        # Y-axis spine
        yaxis = ax.get("yaxis", {})
        if yaxis:
            spine = yaxis.get("spine", {})
            if spine and "bbox_px" in spine:
                bbox_disp = to_display_bbox(spine["bbox_px"])
                bboxes[f"{ax_key}_yaxis_spine"] = {
                    **bbox_disp,
                    "type": "yaxis",
                }

    # Extract traces from artists (top-level in new format, or hit_regions.path_data in old)
    artists = metadata.get("artists", [])
    if not artists:
        # Fallback for old format
        hit_regions = metadata.get("hit_regions", {})
        path_data = hit_regions.get("path_data", {})
        artists = path_data.get("artists", [])

    for artist in artists:
        artist_id = artist.get("id", 0)
        artist_type = artist.get("type", "line")
        bbox_px = artist.get("bbox_px", {})
        if bbox_px:
            bbox_disp = to_display_bbox(bbox_px, is_list=False)
            trace_entry = {
                **bbox_disp,
                "type": artist_type,
                "label": artist.get("label", f"Trace {artist_id}"),
                "element_type": artist_type,
            }

            # Include scaled path points for line proximity detection
            path_px = artist.get("path_px", [])
            if path_px:
                scaled_points = [
                    [pt[0] * scale_x, pt[1] * scale_y]
                    for pt in path_px if len(pt) >= 2
                ]
                trace_entry["points"] = scaled_points

            bboxes[f"trace_{artist_id}"] = trace_entry

    # Add metadata for JavaScript to understand the coordinate system
    bboxes["_meta"] = {
        "display_width": display_width,
        "display_height": display_height,
        "figure_px_width": fig_width,
        "figure_px_height": fig_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        # Note: With new layered format, bbox_px are already in final image space
        # so scale is typically 1:1 (unless display is resized)
    }

    return bboxes


# EOF
