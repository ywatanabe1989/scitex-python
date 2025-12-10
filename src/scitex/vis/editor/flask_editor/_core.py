#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/flask_editor/core.py
"""Core WebEditor class for Flask-based figure editing."""

from pathlib import Path
from typing import Dict, Any, Optional
import copy
import json
import threading
import webbrowser

from ._utils import find_available_port, kill_process_on_port, check_port_available
from ._renderer import render_preview_with_bboxes
from .templates import build_html_template


class WebEditor:
    """
    Browser-based figure editor using Flask.

    Features:
    - Modern responsive UI
    - Real-time preview via WebSocket or polling
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
        manual_overrides: Optional[Dict[str, Any]] = None,
        port: int = 5050,
    ):
        self.json_path = Path(json_path)
        self.metadata = metadata
        self.csv_data = csv_data
        self.png_path = Path(png_path) if png_path else None
        self.manual_overrides = manual_overrides or {}
        self._requested_port = port
        self.port = port

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

        # Handle port conflicts
        if not check_port_available(self._requested_port):
            print(f"Port {self._requested_port} is in use. Attempting to free it...")
            if kill_process_on_port(self._requested_port):
                import time

                time.sleep(0.5)
                self.port = self._requested_port
                print(f"Successfully freed port {self.port}")
            else:
                self.port = find_available_port(self._requested_port + 1)
                print(f"Using alternative port: {self.port}")
        else:
            self.port = self._requested_port

        app = Flask(__name__)
        editor = self

        @app.route("/")
        def index():
            # Rebuild template each time for hot reload support
            html_template = build_html_template()
            return render_template_string(
                html_template,
                filename=str(editor.json_path.resolve()),
                overrides=json.dumps(editor.current_overrides),
            )

        @app.route("/preview")
        def preview():
            """Generate figure preview as base64 PNG with element bboxes."""
            img_data, bboxes, img_size = render_preview_with_bboxes(
                editor.csv_data, editor.current_overrides,
                metadata=editor.metadata
            )
            return jsonify({"image": img_data, "bboxes": bboxes, "img_size": img_size})

        @app.route("/update", methods=["POST"])
        def update():
            """Update overrides and return new preview."""
            data = request.json
            editor.current_overrides.update(data.get("overrides", {}))
            editor._user_modified = True
            img_data, bboxes, img_size = render_preview_with_bboxes(
                editor.csv_data, editor.current_overrides,
                metadata=editor.metadata
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

        print(f"Starting SciTeX Editor at http://127.0.0.1:{self.port}")
        print("Press Ctrl+C to stop")

        # Note: use_reloader=False because the reloader re-runs the entire script
        # which causes infinite loops when the demo generates figures
        # Templates are rebuilt on each page refresh anyway
        app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)


# EOF
