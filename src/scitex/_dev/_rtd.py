#!/usr/bin/env python3
# Timestamp: 2026-02-03
# File: scitex/_dev/_rtd.py

"""Read the Docs build status checking for scitex ecosystem."""

from __future__ import annotations

import urllib.request
from typing import Any

from ._ecosystem import ECOSYSTEM

# RTD project slugs (if different from package name)
RTD_SLUGS: dict[str, str] = {
    "scitex": "scitex-python",
}


def get_rtd_slug(package: str) -> str:
    """Get RTD project slug for a package."""
    return RTD_SLUGS.get(package, package)


def check_rtd_status(package: str, version: str = "latest") -> dict[str, Any]:
    """Check Read the Docs build status for a package.

    Parameters
    ----------
    package : str
        Package name.
    version : str
        RTD version to check (latest, stable, etc.).

    Returns
    -------
    dict
        Status info with keys: status, version, url, error (if any).
    """
    slug = get_rtd_slug(package)
    badge_url = f"https://readthedocs.org/projects/{slug}/badge/?version={version}"
    docs_url = f"https://{slug}.readthedocs.io/en/{version}/"

    try:
        # Fetch the badge SVG content to determine status
        req = urllib.request.Request(badge_url)
        req.add_header("User-Agent", "scitex-dev-tools/1.0")

        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")

            # Parse SVG content for status
            if "passing" in content.lower():
                status = "passing"
            elif "failing" in content.lower():
                status = "failing"
            elif "unknown" in content.lower():
                status = "unknown"
            else:
                status = "unknown"

            return {
                "status": status,
                "version": version,
                "url": docs_url,
            }

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                "status": "not_found",
                "version": version,
                "error": f"Project '{slug}' not found on RTD",
            }
        return {
            "status": "error",
            "version": version,
            "error": f"HTTP {e.code}: {e.reason}",
        }
    except Exception as e:
        return {
            "status": "error",
            "version": version,
            "error": str(e),
        }


def check_all_rtd(
    packages: list[str] | None = None,
    versions: list[str] | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Check RTD status for all ecosystem packages.

    Parameters
    ----------
    packages : list[str] | None
        List of package names. If None, uses ecosystem packages.
    versions : list[str] | None
        List of versions to check. Default: ["latest", "stable"].

    Returns
    -------
    dict
        Mapping: version -> package_name -> status_info
    """
    if packages is None:
        packages = list(ECOSYSTEM.keys())

    if versions is None:
        versions = ["latest", "stable"]

    results: dict[str, dict[str, dict[str, Any]]] = {}

    for version in versions:
        results[version] = {}
        for package in packages:
            results[version][package] = check_rtd_status(package, version)

    return results


# EOF
