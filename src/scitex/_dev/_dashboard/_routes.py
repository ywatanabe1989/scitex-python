#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_dashboard/_routes.py

"""Flask routes for the dashboard."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from flask import Flask


def register_routes(app: Flask) -> None:
    """Register dashboard routes with Flask app."""
    from flask import jsonify, request

    from ._templates import get_dashboard_html, get_error_html

    @app.route("/")
    def index():
        """Serve the main dashboard page."""
        try:
            return get_dashboard_html()
        except Exception as e:
            return get_error_html(str(e)), 500

    @app.route("/api/versions")
    def api_versions():
        """Get version data as JSON."""
        try:
            data = _get_all_version_data()
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/packages")
    def api_packages():
        """Get local package versions only (fast)."""
        try:
            from .._versions import list_versions

            return jsonify(list_versions())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/config")
    def api_config():
        """Get current configuration."""
        try:
            from .._config import get_config_path, load_config

            config = load_config()
            return jsonify(
                {
                    "config_path": str(get_config_path()),
                    "packages": [
                        {
                            "name": p.name,
                            "local_path": p.local_path,
                            "pypi_name": p.pypi_name,
                        }
                        for p in config.packages
                    ],
                    "hosts": [
                        {
                            "name": h.name,
                            "hostname": h.hostname,
                            "role": h.role,
                            "enabled": h.enabled,
                        }
                        for h in config.hosts
                    ],
                    "github_remotes": [
                        {"name": r.name, "org": r.org, "enabled": r.enabled}
                        for r in config.github_remotes
                    ],
                    "branches": config.branches,
                }
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/refresh", methods=["POST"])
    def api_refresh():
        """Trigger a data refresh."""
        try:
            data = _get_all_version_data(force_refresh=True)
            return jsonify({"status": "ok", "data": data})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/hosts")
    def api_hosts():
        """Get host version data."""
        try:
            packages = request.args.getlist("package") or None
            hosts = request.args.getlist("host") or None
            from .._ssh import check_all_hosts

            data = check_all_hosts(packages=packages, hosts=hosts)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/remotes")
    def api_remotes():
        """Get GitHub remote version data."""
        try:
            packages = request.args.getlist("package") or None
            remotes = request.args.getlist("remote") or None
            from .._github import check_all_remotes

            data = check_all_remotes(packages=packages, remotes=remotes)
            return jsonify(data)
        except Exception as e:
            return jsonify({"error": str(e)}), 500


def _get_all_version_data(force_refresh: bool = False) -> dict[str, Any]:
    """Get all version data from all sources.

    Parameters
    ----------
    force_refresh : bool
        If True, bypass any caching.

    Returns
    -------
    dict
        Combined version data.
    """
    from .._config import get_enabled_hosts, get_enabled_remotes, load_config
    from .._github import check_all_remotes
    from .._ssh import check_all_hosts
    from .._versions import list_versions

    config = load_config()

    # Get local versions
    packages_data = list_versions()

    # Get host versions (if any hosts configured)
    hosts_data = {}
    enabled_hosts = get_enabled_hosts(config)
    if enabled_hosts:
        try:
            hosts_data = check_all_hosts(config=config)
        except Exception:
            pass

    # Get remote versions (if any remotes configured)
    remotes_data = {}
    enabled_remotes = get_enabled_remotes(config)
    if enabled_remotes:
        try:
            remotes_data = check_all_remotes(config=config)
        except Exception:
            pass

    return {
        "packages": packages_data,
        "hosts": hosts_data,
        "remotes": remotes_data,
    }


# EOF
