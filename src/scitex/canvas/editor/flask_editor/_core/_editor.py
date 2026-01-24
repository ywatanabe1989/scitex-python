#!/usr/bin/env python3
# Timestamp: "2026-01-24 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-python/src/scitex/canvas/editor/flask_editor/_core/_editor.py

"""Core WebEditor class for Flask-based figure editing."""

import copy
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from .._utils import check_port_available, kill_process_on_port

__all__ = ["WebEditor"]


class WebEditor:
    """Browser-based figure editor using Flask.

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
        self.panel_info = panel_info

        # Extract hit_regions from metadata
        self.hit_regions = metadata.get("hit_regions", {})
        self.color_map = self.hit_regions.get("color_map", {})

        # Get SciTeX defaults and merge with metadata
        from ..._defaults import extract_defaults_from_metadata, get_scitex_defaults

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
            from flask import Flask
        except ImportError:
            raise ImportError(
                "Flask is required for web editor. Install: pip install flask"
            )

        # Handle port conflicts
        self._setup_port()

        # Configure Flask
        import os

        static_folder = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "static"
        )
        app = Flask(__name__, static_folder=static_folder, static_url_path="/static")

        # Register all routes
        self._register_routes(app)

        # Open browser after short delay
        def open_browser():
            time.sleep(0.5)
            webbrowser.open(f"http://127.0.0.1:{self.port}")

        threading.Thread(target=open_browser, daemon=True).start()

        print(f"Starting SciTeX Figure Editor at http://127.0.0.1:{self.port}")
        print("Press Ctrl+C to stop")

        app.run(host="127.0.0.1", port=self.port, debug=False, use_reloader=False)

    def _setup_port(self):
        """Handle port conflicts."""
        max_retries = 3
        for attempt in range(max_retries):
            if check_port_available(self._requested_port):
                self.port = self._requested_port
                break
            print(
                f"Port {self._requested_port} in use. Freeing... "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            kill_process_on_port(self._requested_port)
            time.sleep(1.0)
        else:
            print(f"Warning: Port {self._requested_port} may still be in use")
            self.port = self._requested_port

    def _register_routes(self, app):
        """Register all Flask routes."""
        from ._routes_basic import (
            create_colormap_route,
            create_hitmap_route,
            create_index_route,
            create_preview_route,
            create_shutdown_route,
            create_stats_route,
            create_update_route,
        )
        from ._routes_export import (
            create_download_figz_route,
            create_download_route,
            create_export_route,
        )
        from ._routes_panels import (
            create_panels_route,
            create_switch_panel_route,
        )
        from ._routes_save import (
            create_save_element_position_route,
            create_save_layout_route,
            create_save_route,
        )

        # Basic routes
        create_index_route(app, self)
        create_preview_route(app, self)
        create_hitmap_route(app, self)
        create_colormap_route(app, self)
        create_update_route(app, self)
        create_stats_route(app, self)
        create_shutdown_route(app, self)

        # Panel routes
        create_panels_route(app, self)
        create_switch_panel_route(app, self)

        # Save routes
        create_save_route(app, self)
        create_save_layout_route(app, self)
        create_save_element_position_route(app, self)

        # Export routes
        create_export_route(app, self)
        create_download_route(app, self)
        create_download_figz_route(app, self)


# EOF
