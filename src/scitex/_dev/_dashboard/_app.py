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

    # Disable JSON key sorting to preserve insertion order (Flask 2.2+)
    app.json.sort_keys = False

    # Register routes
    from ._routes import register_routes

    register_routes(app)

    return app


def _kill_process_on_port(port: int) -> None:
    """Kill any process using the specified port.

    Parameters
    ----------
    port : int
        Port number to free up.
    """
    import subprocess
    import sys

    try:
        if sys.platform == "win32":
            # Windows: use netstat and taskkill
            result = subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    pid = line.strip().split()[-1]
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True,
                        check=False,
                    )
                    print(f"Killed process {pid} on port {port}")
        else:
            # Unix: use lsof
            result = subprocess.run(
                ["lsof", "-ti", f":{port}"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                for pid in pids:
                    subprocess.run(
                        ["kill", "-9", pid], capture_output=True, check=False
                    )
                    print(f"Killed process {pid} on port {port}")
    except Exception as e:
        print(f"Warning: Could not kill process on port {port}: {e}")


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
    force: bool = False,
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
    force : bool
        Kill existing process using the port if any.
    """
    if force:
        _kill_process_on_port(port)

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
