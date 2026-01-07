#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/core.py
"""Core WebEditor class for Flask-based figure editing."""

import base64
import copy
import json
import threading
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from ._utils import check_port_available, find_available_port, kill_process_on_port
from .templates import build_html_template


class WebEditor:
    """
    Browser-based figure editor using Flask.

    Features:
    - Displays existing PNG from plot bundle (no re-rendering)
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
        self.panel_info = panel_info  # For multi-panel figure bundles

        # Extract hit_regions from metadata for color-based element detection
        self.hit_regions = metadata.get("hit_regions", {})
        self.color_map = self.hit_regions.get("color_map", {})

        # Get SciTeX defaults and merge with metadata
        from .._defaults import extract_defaults_from_metadata, get_scitex_defaults

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
            from flask import Flask, jsonify, render_template_string, request
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
            print(
                f"Port {self._requested_port} in use. Freeing... (attempt {attempt + 1}/{max_retries})"
            )
            kill_process_on_port(self._requested_port)
            time.sleep(1.0)  # Wait for port release
        else:
            # After retries, use requested port anyway (Flask will error if unavailable)
            print(f"Warning: Port {self._requested_port} may still be in use")
            self.port = self._requested_port

        # Configure Flask with static folder path
        import os

        static_folder = os.path.join(os.path.dirname(__file__), "static")
        app = Flask(__name__, static_folder=static_folder, static_url_path="/static")
        editor = self

        def _export_composed_figure(editor, formats=["png", "svg"], dpi=150):
            """Helper to compose and export figure to bundle."""
            import matplotlib
            import numpy as np
            from PIL import Image

            from scitex.io import ZipBundle

            matplotlib.use("Agg")
            import io
            import json as json_module
            import zipfile

            import matplotlib.pyplot as plt

            if not editor.panel_info:
                return {"success": False, "error": "No panel info"}

            bundle_path = editor.panel_info.get("bundle_path")
            figure_dir = editor.panel_info.get("figure_dir")

            if not bundle_path and not figure_dir:
                return {"success": False, "error": "No bundle path"}

            figure_name = (
                Path(bundle_path).stem
                if bundle_path
                else (
                    Path(figure_dir).stem.replace(".figure", "")
                    if figure_dir
                    else "figure"
                )
            )

            # Read spec.json for layout and layout.json for position overrides
            spec = {}
            layout_overrides = {}
            if bundle_path:
                try:
                    with ZipBundle(bundle_path, mode="r") as bundle:
                        spec = bundle.read_json("spec.json")
                        try:
                            layout_overrides = bundle.read_json("layout.json")
                        except:
                            pass
                except:
                    pass
            elif figure_dir:
                spec_path = Path(figure_dir) / "spec.json"
                if spec_path.exists():
                    with open(spec_path) as f:
                        spec = json_module.load(f)
                layout_path = Path(figure_dir) / "layout.json"
                if layout_path.exists():
                    with open(layout_path) as f:
                        layout_overrides = json_module.load(f)

            # Also check in-memory layout overrides
            if editor.panel_info and editor.panel_info.get("layout"):
                layout_overrides = editor.panel_info.get("layout", {})

            # Get figure dimensions
            fig_width_mm = 180
            fig_height_mm = 120
            if "figure" in spec:
                fig_info = spec.get("figure", {})
                styles = fig_info.get("styles", {})
                size = styles.get("size", {})
                fig_width_mm = size.get("width_mm", 180)
                fig_height_mm = size.get("height_mm", 120)

            fig_width_in = fig_width_mm / 25.4
            fig_height_in = fig_height_mm / 25.4

            fig = plt.figure(
                figsize=(fig_width_in, fig_height_in), dpi=dpi, facecolor="white"
            )

            # Compose panels
            panels_spec = spec.get("panels", [])
            panel_paths = editor.panel_info.get("panel_paths", [])
            panel_is_zip = editor.panel_info.get("panel_is_zip", [])

            for panel_spec in panels_spec:
                panel_id = panel_spec.get("id", "")
                pos = panel_spec.get("position", {})
                size = panel_spec.get("size", {})

                # Skip overview/auxiliary panels (only compose main panels A-Z)
                panel_id_lower = panel_id.lower()
                if any(
                    skip in panel_id_lower
                    for skip in ["overview", "thumb", "preview", "aux"]
                ):
                    continue

                # Find panel path first (needed to check layout_overrides)
                panel_path = None
                is_zip = False
                panel_name = None
                for idx, pp in enumerate(panel_paths):
                    pp_name = Path(pp).stem.replace(".plot", "")
                    if (
                        pp_name == panel_id
                        or pp_name.startswith(f"panel_{panel_id}_")
                        or pp_name == f"panel_{panel_id}"
                        or f"_{panel_id}_" in pp_name
                    ):
                        panel_path = pp
                        panel_name = Path(pp).name  # e.g., "panel_A_twinx.plot"
                        is_zip = panel_is_zip[idx] if idx < len(panel_is_zip) else False
                        break

                if not panel_path:
                    continue

                # Check for layout overrides (from layout.json or in-memory)
                override = layout_overrides.get(panel_name, {})
                override_pos = override.get("position", {})
                override_size = override.get("size", {})

                # Use override positions if available, otherwise use spec
                x_mm = override_pos.get("x_mm", pos.get("x_mm", 0))
                y_mm = override_pos.get("y_mm", pos.get("y_mm", 0))
                w_mm = override_size.get("width_mm", size.get("width_mm", 60))
                h_mm = override_size.get("height_mm", size.get("height_mm", 40))

                x_frac = x_mm / fig_width_mm
                y_frac = 1 - (y_mm + h_mm) / fig_height_mm
                w_frac = w_mm / fig_width_mm
                h_frac = h_mm / fig_height_mm

                # Load panel preview
                try:
                    # Exclusion patterns for preview selection
                    exclude_patterns = ["hitmap", "overview", "thumb", "preview"]

                    if is_zip:
                        with ZipBundle(panel_path, mode="r") as plot_bundle:
                            with zipfile.ZipFile(panel_path, "r") as zf:
                                png_files = [
                                    n
                                    for n in zf.namelist()
                                    if n.endswith(".png")
                                    and "exports/" in n
                                    and not any(
                                        p in n.lower() for p in exclude_patterns
                                    )
                                ]
                                if png_files:
                                    preview_path = png_files[0]
                                    if ".plot/" in preview_path:
                                        preview_path = preview_path.split(".plot/")[-1]
                                    img_data = plot_bundle.read_bytes(preview_path)
                                    img = Image.open(io.BytesIO(img_data))
                                    ax = fig.add_axes([x_frac, y_frac, w_frac, h_frac])
                                    ax.imshow(np.array(img))
                                    ax.axis("off")
                    else:
                        plot_dir = Path(panel_path)
                        exports_dir = plot_dir / "exports"
                        if exports_dir.exists():
                            for png_file in exports_dir.glob("*.png"):
                                name_lower = png_file.name.lower()
                                if not any(p in name_lower for p in exclude_patterns):
                                    img = Image.open(png_file)
                                    ax = fig.add_axes([x_frac, y_frac, w_frac, h_frac])
                                    ax.imshow(np.array(img))
                                    ax.axis("off")
                                    break
                except Exception as e:
                    print(f"Could not load panel {panel_id}: {e}")

                # Draw panel letter
                if (
                    panel_id and len(panel_id) <= 2
                ):  # Only for short IDs like A, B, C...
                    # Position letter at top-left corner of panel
                    letter_x = x_frac + 0.01
                    letter_y = y_frac + h_frac - 0.02
                    fig.text(
                        letter_x,
                        letter_y,
                        panel_id,
                        fontsize=14,
                        fontweight="bold",
                        color="black",
                        ha="left",
                        va="top",
                        transform=fig.transFigure,
                        bbox=dict(
                            boxstyle="square,pad=0.1",
                            facecolor="white",
                            edgecolor="none",
                            alpha=0.8,
                        ),
                    )

            exported = {}

            # Save to bundle
            if bundle_path:
                with ZipBundle(bundle_path, mode="a") as bundle:
                    for fmt in formats:
                        buf = io.BytesIO()
                        fig.savefig(
                            buf,
                            format=fmt,
                            dpi=dpi,
                            bbox_inches="tight",
                            facecolor="white",
                            pad_inches=0.02,
                        )
                        buf.seek(0)
                        export_path = f"exports/{figure_name}.{fmt}"
                        bundle.write_bytes(export_path, buf.read())
                        exported[fmt] = export_path

            plt.close(fig)
            return {"success": True, "exported": exported}

        @app.route("/")
        def index():
            # Rebuild template each time for hot reload support
            html_template = build_html_template()

            # Extract figz and panel paths for display
            json_path_str = str(editor.json_path.resolve())
            figure_path = ""
            panel_path = ""

            # Check if this is inside a figure bundle
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

        @app.route("/preview")
        def preview():
            """Render figure preview with current overrides (same logic as /update)."""
            from ._renderer import render_preview_with_bboxes

            # Always use renderer for consistency between initial and updated views
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

        @app.route("/panels")
        def panels():
            """Return all panel images with bboxes for interactive grid view (figure bundles only).

            Uses smart load_panel_data helper for transparent zip/directory handling.
            Returns layout info from figz spec.json for unified canvas positioning.
            """
            import json as json_module

            from ..edit import load_panel_data
            from ._bbox import (
                extract_bboxes_from_geometry_px,
                extract_bboxes_from_metadata,
            )

            if not editor.panel_info:
                return jsonify({"error": "Not a multi-panel figure bundle"}), 400

            panel_names = editor.panel_info["panels"]
            panel_paths = editor.panel_info.get("panel_paths", [])
            panel_is_zip = editor.panel_info.get(
                "panel_is_zip", [False] * len(panel_names)
            )
            figure_dir = Path(editor.panel_info["figure_dir"])

            if not panel_paths:
                panel_paths = [str(figure_dir / name) for name in panel_names]

            # Load figz spec.json to get panel layout
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

                # Use smart helper to load panel data
                loaded = load_panel_data(panel_path, is_zip=is_zip)

                panel_data = {
                    "name": display_name,
                    "image": None,
                    "bboxes": None,
                    "img_size": None,
                }

                # Add layout info from figz spec
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

                    # Extract bboxes - prefer geometry_px.json
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

        @app.route("/switch_panel/<int:panel_index>")
        def switch_panel(panel_index):
            """Switch to a different panel in the figure bundle.

            Uses smart load_panel_data helper for transparent zip/directory handling.
            """
            from ..edit import load_panel_data
            from ._bbox import (
                extract_bboxes_from_geometry_px,
                extract_bboxes_from_metadata,
            )

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
            is_zip = (
                panel_is_zip[panel_index] if panel_index < len(panel_is_zip) else None
            )

            try:
                # Use smart helper to load panel data
                loaded = load_panel_data(panel_path, is_zip=is_zip)

                if not loaded:
                    return (
                        jsonify({"error": f"Could not load panel: {panel_name}"}),
                        400,
                    )

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
                    return (
                        jsonify({"error": f"No PNG found for panel: {panel_name}"}),
                        400,
                    )

                # Get image size
                img_size = loaded.get("img_size", {"width": 0, "height": 0})
                if not img_size and loaded.get("png_path"):
                    from PIL import Image

                    img = Image.open(loaded["png_path"])
                    img_size = {"width": img.size[0], "height": img.size[1]}
                    img.close()

                # Extract bboxes - prefer geometry_px.json
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

                # Re-extract defaults from new metadata
                from .._defaults import (
                    extract_defaults_from_metadata,
                    get_scitex_defaults,
                )

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

                return (
                    jsonify(
                        {
                            "error": f"Failed to switch panel: {str(e)}",
                            "traceback": traceback.format_exc(),
                        }
                    ),
                    500,
                )

        @app.route("/hitmap")
        def hitmap():
            """Return hitmap PNG for element detection."""
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

        @app.route("/color_map")
        def color_map():
            """Return color map for hitmap element identification."""
            return jsonify(
                {
                    "color_map": editor.color_map,
                    "hit_regions": editor.hit_regions,
                }
            )

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

        @app.route("/save", methods=["POST"])
        def save():
            """Save to .manual.json."""
            from ..edit import save_manual_overrides

            try:
                manual_path = save_manual_overrides(
                    editor.json_path, editor.current_overrides
                )
                return jsonify({"status": "saved", "path": str(manual_path)})
            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

        @app.route("/save_layout", methods=["POST"])
        def save_layout():
            """Save panel layout positions to figure bundle."""
            try:
                data = request.get_json()
                layout = data.get("layout", {})

                if not layout:
                    return jsonify(
                        {"success": False, "error": "No layout data provided"}
                    )

                # Check if we have panel_info (figure bundle)
                if not editor.panel_info:
                    return jsonify(
                        {
                            "success": False,
                            "error": "No panel info available (not a figure bundle)",
                        }
                    )

                bundle_path = editor.panel_info.get("bundle_path")
                if not bundle_path:
                    return jsonify(
                        {"success": False, "error": "Bundle path not available"}
                    )

                # Update layout in the figure bundle
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

                    # Update position
                    existing_layout[panel_name]["position"]["x_mm"] = pos.get("x_mm", 0)
                    existing_layout[panel_name]["position"]["y_mm"] = pos.get("y_mm", 0)

                    # Update size if provided
                    if "width_mm" in pos:
                        existing_layout[panel_name]["size"]["width_mm"] = pos[
                            "width_mm"
                        ]
                    if "height_mm" in pos:
                        existing_layout[panel_name]["size"]["height_mm"] = pos[
                            "height_mm"
                        ]

                # Save updated layout
                bundle.write_json("layout.json", existing_layout)

                # Update in-memory panel_info
                editor.panel_info["layout"] = existing_layout

                # Auto-export composed figure to bundle
                export_result = _export_composed_figure(editor, formats=["png", "svg"])

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

        @app.route("/save_element_position", methods=["POST"])
        def save_element_position():
            """Save element position (legend/panel_letter) to figure bundle.

            ONLY legends and panel letters can be repositioned to maintain
            scientific rigor. Data elements are never moved.
            """
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
                    return jsonify(
                        {"success": False, "error": "No panel info available"}
                    )

                bundle_path = editor.panel_info.get("bundle_path")
                if not bundle_path:
                    return jsonify(
                        {"success": False, "error": "Bundle path not available"}
                    )

                from scitex.canvas.io import ZipBundle

                bundle = ZipBundle(bundle_path)

                # Read or create style.json for element positions
                try:
                    style = bundle.read_json("style.json")
                except:
                    style = {}

                # Initialize structure
                if "elements" not in style:
                    style["elements"] = {}
                if panel not in style["elements"]:
                    style["elements"][panel] = {}

                # Save element position
                style["elements"][panel][element] = {
                    "type": element_type,
                    "position": position,
                    "snap_name": snap_name,
                }

                # For legends, also update legend_location for matplotlib compatibility
                if element_type == "legend" and snap_name:
                    # Convert snap name to matplotlib loc format
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

        @app.route("/export", methods=["POST"])
        def export_figure():
            """Export composed figure to various formats and update figure bundle."""
            try:
                data = request.get_json()
                formats = data.get("formats", ["png", "svg"])

                if not editor.panel_info:
                    return jsonify(
                        {"success": False, "error": "No panel info available"}
                    )

                bundle_path = editor.panel_info.get("bundle_path")
                if not bundle_path:
                    return jsonify(
                        {"success": False, "error": "Bundle path not available"}
                    )

                import io
                from pathlib import Path

                import matplotlib

                from scitex.io import ZipBundle

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                import numpy as np
                from PIL import Image

                figure_name = Path(bundle_path).stem
                dpi = data.get("dpi", 150)

                with ZipBundle(bundle_path, mode="a") as bundle:
                    # Read spec for figure size and panel positions
                    try:
                        spec = bundle.read_json("spec.json")
                    except:
                        spec = {}

                    # Get figure dimensions
                    fig_width_mm = 180
                    fig_height_mm = 120
                    if "figure" in spec:
                        fig_info = spec.get("figure", {})
                        styles = fig_info.get("styles", {})
                        size = styles.get("size", {})
                        fig_width_mm = size.get("width_mm", 180)
                        fig_height_mm = size.get("height_mm", 120)

                    # Convert mm to inches
                    fig_width_in = fig_width_mm / 25.4
                    fig_height_in = fig_height_mm / 25.4

                    # Create figure with white background
                    fig = plt.figure(
                        figsize=(fig_width_in, fig_height_in),
                        dpi=dpi,
                        facecolor="white",
                    )

                    # Get panels from spec or editor.panel_info
                    panels_spec = spec.get("panels", [])

                    # Compose panels onto figure
                    for panel_spec in panels_spec:
                        panel_id = panel_spec.get("id", "")
                        plot_name = panel_spec.get("plot", "")

                        # Get position and size from spec
                        pos = panel_spec.get("position", {})
                        size = panel_spec.get("size", {})

                        x_mm = pos.get("x_mm", 0)
                        y_mm = pos.get("y_mm", 0)
                        w_mm = size.get("width_mm", 60)
                        h_mm = size.get("height_mm", 40)

                        # Convert to figure coordinates (0-1)
                        x_frac = x_mm / fig_width_mm
                        y_frac = 1 - (y_mm + h_mm) / fig_height_mm  # Flip Y
                        w_frac = w_mm / fig_width_mm
                        h_frac = h_mm / fig_height_mm

                        # Try to read panel image from pltz exports
                        img_loaded = False
                        for plot_path in [
                            f"{panel_id}.plot",
                            plot_name.replace(".d", ""),
                        ]:
                            if img_loaded:
                                break
                            try:
                                # Read pltz as nested bundle
                                plot_bytes = bundle.read_bytes(plot_path)
                                import tempfile

                                with tempfile.NamedTemporaryFile(
                                    suffix=".plot", delete=False
                                ) as tmp:
                                    tmp.write(plot_bytes)
                                    tmp_path = tmp.name
                                try:
                                    with ZipBundle(tmp_path, mode="r") as plot_bundle:
                                        # Try various preview paths
                                        for preview_path in [
                                            "exports/preview.png",
                                            "preview.png",
                                            f"exports/{panel_id}.png",
                                        ]:
                                            try:
                                                img_data = plot_bundle.read_bytes(
                                                    preview_path
                                                )
                                                img = Image.open(io.BytesIO(img_data))
                                                img_array = np.array(img)

                                                # Create axes and add image
                                                ax = fig.add_axes(
                                                    [x_frac, y_frac, w_frac, h_frac]
                                                )
                                                ax.imshow(img_array)
                                                ax.axis("off")
                                                img_loaded = True
                                                break
                                            except:
                                                continue
                                finally:
                                    import os

                                    os.unlink(tmp_path)
                            except Exception as e:
                                print(f"Could not load plot {plot_path}: {e}")
                                continue

                    exported = {}

                    for fmt in formats:
                        buf = io.BytesIO()
                        if fmt in ["png", "jpeg", "jpg"]:
                            fig.savefig(
                                buf,
                                format="png" if fmt == "png" else "jpeg",
                                dpi=dpi,
                                bbox_inches="tight",
                                facecolor="white",
                                pad_inches=0.02,
                            )
                        elif fmt == "svg":
                            fig.savefig(
                                buf, format="svg", bbox_inches="tight", pad_inches=0.02
                            )
                        elif fmt == "pdf":
                            fig.savefig(
                                buf, format="pdf", bbox_inches="tight", pad_inches=0.02
                            )
                        else:
                            continue

                        buf.seek(0)
                        content = buf.read()

                        # Save to exports/ directory in bundle
                        export_path = f"exports/{figure_name}.{fmt}"
                        bundle.write_bytes(export_path, content)
                        exported[fmt] = export_path

                    plt.close(fig)

                return jsonify(
                    {
                        "success": True,
                        "exported": exported,
                        "bundle_path": str(bundle_path),
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

        @app.route("/download/<fmt>")
        def download_figure(fmt):
            """Download figure in specified format."""
            try:
                import io
                from pathlib import Path

                from flask import send_file

                mime_types = {
                    "png": "image/png",
                    "jpeg": "image/jpeg",
                    "jpg": "image/jpeg",
                    "svg": "image/svg+xml",
                    "pdf": "application/pdf",
                }

                if fmt not in mime_types:
                    return f"Unsupported format: {fmt}", 400

                # For figure bundles, download the composed figure
                if editor.panel_info:
                    bundle_path = editor.panel_info.get("bundle_path")
                    figure_dir = editor.panel_info.get("figure_dir")
                    figure_name = (
                        Path(bundle_path).stem
                        if bundle_path
                        else (
                            Path(figure_dir).stem.replace(".figure", "")
                            if figure_dir
                            else "figure"
                        )
                    )

                    if bundle_path or figure_dir:
                        import matplotlib
                        import numpy as np
                        from PIL import Image

                        from scitex.io import ZipBundle

                        matplotlib.use("Agg")
                        import json as json_module

                        import matplotlib.pyplot as plt

                        # Always compose on-demand to ensure current panel state
                        # (existing exports in bundle may be stale or blank)
                        # Read spec.json and layout.json for position overrides
                        spec = {}
                        layout_overrides = {}
                        if bundle_path:
                            try:
                                with ZipBundle(bundle_path, mode="r") as bundle:
                                    spec = bundle.read_json("spec.json")
                                    try:
                                        layout_overrides = bundle.read_json(
                                            "layout.json"
                                        )
                                    except:
                                        pass
                            except:
                                pass
                        elif figure_dir:
                            spec_path = Path(figure_dir) / "spec.json"
                            if spec_path.exists():
                                with open(spec_path) as f:
                                    spec = json_module.load(f)
                            layout_path = Path(figure_dir) / "layout.json"
                            if layout_path.exists():
                                with open(layout_path) as f:
                                    layout_overrides = json_module.load(f)

                        # Also check in-memory layout overrides (most current)
                        if editor.panel_info and editor.panel_info.get("layout"):
                            layout_overrides = editor.panel_info.get("layout", {})

                        # Get figure dimensions
                        fig_width_mm = 180
                        fig_height_mm = 120
                        if "figure" in spec:
                            fig_info = spec.get("figure", {})
                            styles = fig_info.get("styles", {})
                            size = styles.get("size", {})
                            fig_width_mm = size.get("width_mm", 180)
                            fig_height_mm = size.get("height_mm", 120)

                        fig_width_in = fig_width_mm / 25.4
                        fig_height_in = fig_height_mm / 25.4

                        dpi = 150 if fmt in ["jpeg", "jpg"] else 300
                        fig = plt.figure(
                            figsize=(fig_width_in, fig_height_in),
                            dpi=dpi,
                            facecolor="white",
                        )

                        # Compose panels
                        panels_spec = spec.get("panels", [])
                        panel_paths = editor.panel_info.get("panel_paths", [])
                        panel_is_zip = editor.panel_info.get("panel_is_zip", [])

                        for panel_spec in panels_spec:
                            panel_id = panel_spec.get("id", "")
                            pos = panel_spec.get("position", {})
                            size = panel_spec.get("size", {})

                            # Skip overview/auxiliary panels (only compose main panels A-Z)
                            panel_id_lower = panel_id.lower()
                            if any(
                                skip in panel_id_lower
                                for skip in ["overview", "thumb", "preview", "aux"]
                            ):
                                continue

                            # Find panel path first (needed to check layout_overrides)
                            panel_path = None
                            is_zip = False
                            panel_name = None
                            for idx, pp in enumerate(panel_paths):
                                pp_name = Path(pp).stem.replace(".plot", "")
                                # Match exact name, or name contains panel_id pattern
                                # e.g., "panel_A_twinx" matches panel_id "A"
                                if (
                                    pp_name == panel_id
                                    or pp_name.startswith(f"panel_{panel_id}_")
                                    or pp_name.startswith(f"panel_{panel_id}.")
                                    or pp_name == f"panel_{panel_id}"
                                    or pp_name == panel_id
                                    or f"_{panel_id}_" in pp_name
                                    or pp_name.endswith(f"_{panel_id}")
                                ):
                                    panel_path = pp
                                    panel_name = Path(
                                        pp
                                    ).name  # e.g., "panel_A_twinx.plot"
                                    is_zip = (
                                        panel_is_zip[idx]
                                        if idx < len(panel_is_zip)
                                        else False
                                    )
                                    break

                            if not panel_path:
                                print(
                                    f"Could not find panel path for id={panel_id}, available: {[Path(p).stem for p in panel_paths]}"
                                )
                                continue

                            # Check for layout overrides (from layout.json or in-memory)
                            override = layout_overrides.get(panel_name, {})
                            override_pos = override.get("position", {})
                            override_size = override.get("size", {})

                            # Use override positions if available, otherwise use spec
                            x_mm = override_pos.get("x_mm", pos.get("x_mm", 0))
                            y_mm = override_pos.get("y_mm", pos.get("y_mm", 0))
                            w_mm = override_size.get(
                                "width_mm", size.get("width_mm", 60)
                            )
                            h_mm = override_size.get(
                                "height_mm", size.get("height_mm", 40)
                            )

                            x_frac = x_mm / fig_width_mm
                            y_frac = 1 - (y_mm + h_mm) / fig_height_mm
                            w_frac = w_mm / fig_width_mm
                            h_frac = h_mm / fig_height_mm

                            # Load panel preview image
                            try:
                                img_loaded = False
                                # Exclusion patterns for preview selection
                                exclude_patterns = [
                                    "hitmap",
                                    "overview",
                                    "thumb",
                                    "preview",
                                ]

                                if is_zip:
                                    with ZipBundle(panel_path, mode="r") as plot_bundle:
                                        # Find PNG in exports (exclude hitmap, overview, thumbnails)
                                        import zipfile

                                        with zipfile.ZipFile(panel_path, "r") as zf:
                                            png_files = [
                                                n
                                                for n in zf.namelist()
                                                if n.endswith(".png")
                                                and "exports/" in n
                                                and not any(
                                                    p in n.lower()
                                                    for p in exclude_patterns
                                                )
                                            ]
                                            if png_files:
                                                # Use first matching PNG
                                                preview_path = png_files[0]
                                                # Extract the path relative to .d directory
                                                if ".plot/" in preview_path:
                                                    preview_path = preview_path.split(
                                                        ".plot/"
                                                    )[-1]
                                                try:
                                                    img_data = plot_bundle.read_bytes(
                                                        preview_path
                                                    )
                                                    img = Image.open(
                                                        io.BytesIO(img_data)
                                                    )
                                                    ax = fig.add_axes(
                                                        [x_frac, y_frac, w_frac, h_frac]
                                                    )
                                                    ax.imshow(np.array(img))
                                                    ax.axis("off")
                                                    img_loaded = True
                                                except Exception as e:
                                                    print(
                                                        f"Could not read {preview_path}: {e}"
                                                    )
                                else:
                                    # Directory-based pltz
                                    plot_dir = Path(panel_path)
                                    exports_dir = plot_dir / "exports"
                                    if exports_dir.exists():
                                        for png_file in exports_dir.glob("*.png"):
                                            name_lower = png_file.name.lower()
                                            if not any(
                                                p in name_lower
                                                for p in exclude_patterns
                                            ):
                                                img = Image.open(png_file)
                                                ax = fig.add_axes(
                                                    [x_frac, y_frac, w_frac, h_frac]
                                                )
                                                ax.imshow(np.array(img))
                                                ax.axis("off")
                                                img_loaded = True
                                                break
                                if not img_loaded:
                                    print(f"No preview found for panel {panel_id}")
                            except Exception as e:
                                print(f"Could not load panel {panel_id}: {e}")

                            # Draw panel letter
                            if (
                                panel_id and len(panel_id) <= 2
                            ):  # Only for short IDs like A, B, C...
                                # Position letter at top-left corner of panel
                                letter_x = x_frac + 0.01
                                letter_y = y_frac + h_frac - 0.02
                                fig.text(
                                    letter_x,
                                    letter_y,
                                    panel_id,
                                    fontsize=14,
                                    fontweight="bold",
                                    color="black",
                                    ha="left",
                                    va="top",
                                    transform=fig.transFigure,
                                    bbox=dict(
                                        boxstyle="square,pad=0.1",
                                        facecolor="white",
                                        edgecolor="none",
                                        alpha=0.8,
                                    ),
                                )

                        buf = io.BytesIO()
                        fig.savefig(
                            buf,
                            format=fmt if fmt != "jpg" else "jpeg",
                            dpi=dpi,
                            bbox_inches="tight",
                            facecolor="white",
                            pad_inches=0.02,
                        )
                        plt.close(fig)
                        buf.seek(0)

                        return send_file(
                            buf,
                            mimetype=mime_types[fmt],
                            as_attachment=True,
                            download_name=f"{figure_name}.{fmt}",
                        )

                # For single pltz files, render from csv_data
                import matplotlib

                from ._renderer import render_preview_with_bboxes

                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                figure_name = "figure"
                if editor.json_path:
                    figure_name = Path(editor.json_path).stem

                img_data, _, _ = render_preview_with_bboxes(
                    editor.csv_data,
                    editor.current_overrides,
                    metadata=editor.metadata,
                    dark_mode=False,
                )

                if fmt == "png":
                    import base64

                    content = base64.b64decode(img_data)
                    buf = io.BytesIO(content)
                    return send_file(
                        buf,
                        mimetype=mime_types[fmt],
                        as_attachment=True,
                        download_name=f"{figure_name}.{fmt}",
                    )

                # For other formats, re-render
                from ._plotter import plot_from_csv

                fig, ax = plt.subplots(figsize=(8, 6))
                plot_from_csv(ax, editor.csv_data, editor.current_overrides)

                buf = io.BytesIO()
                dpi = 150 if fmt in ["jpeg", "jpg"] else 300
                fig.savefig(
                    buf,
                    format=fmt if fmt != "jpg" else "jpeg",
                    dpi=dpi,
                    bbox_inches="tight",
                    facecolor="white" if fmt in ["jpeg", "jpg"] else None,
                )
                plt.close(fig)
                buf.seek(0)

                return send_file(
                    buf,
                    mimetype=mime_types[fmt],
                    as_attachment=True,
                    download_name=f"{figure_name}.{fmt}",
                )

            except Exception as e:
                import traceback

                return f"Error: {str(e)}\n{traceback.format_exc()}", 500

        @app.route("/download_figz")
        def download_figz():
            """Download as figure bundle (re-editable format)."""
            try:
                if not editor.panel_info:
                    return "No panel info available", 404

                bundle_path = editor.panel_info.get("bundle_path")
                if not bundle_path:
                    return "Bundle path not available", 404

                from pathlib import Path

                from flask import send_file

                # Send the figz file directly (it's already a pltz-compatible format)
                return send_file(
                    bundle_path,
                    mimetype="application/zip",
                    as_attachment=True,
                    download_name=Path(bundle_path).name,
                )

            except Exception as e:
                return str(e), 500

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
            return jsonify(
                {
                    "stats": stats_data,
                    "stats_summary": stats_summary,
                    "has_stats": len(stats_data) > 0,
                }
            )

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
    display_height: Optional[float] = None,
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
                    [pt[0] * scale_x, pt[1] * scale_y] for pt in path_px if len(pt) >= 2
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
