#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/_app.py

"""Flask application factory for the dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns
    -------
    Flask
        Configured Flask application.
    """
    try:
        from flask import Flask
    except ImportError as e:
        raise ImportError(
            "Flask is required for the dashboard. Install with: pip install flask"
        ) from e

    from pathlib import Path

    static_folder = Path(__file__).parent / "static"
    app = Flask(__name__, static_folder=str(static_folder), static_url_path="/static")
    app.config["JSON_SORT_KEYS"] = False

    # Register routes
    from ._routes import register_routes

    register_routes(app)

    return app


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
) -> None:
    """Run the Flask dashboard server.

    Parameters
    ----------
    host : str
        Host to bind to. Default "127.0.0.1".
    port : int
        Port to listen on. Default 5000.
    debug : bool
        Enable Flask debug mode.
    open_browser : bool
        Open browser automatically.
    """
    app = create_app()

    url = f"http://{host}:{port}"
    print(f"Starting SciTeX Version Dashboard at {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        import threading
        import webbrowser

        def open_url():
            import time

            time.sleep(1)  # Wait for server to start
            webbrowser.open(url)

        threading.Thread(target=open_url, daemon=True).start()

    try:
        app.run(host=host, port=port, debug=debug, threaded=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")


# EOF
