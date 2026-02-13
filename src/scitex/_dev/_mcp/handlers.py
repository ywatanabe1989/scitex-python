#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_mcp/handlers.py

"""MCP handler implementations for developer utilities."""

from __future__ import annotations

from typing import Any


async def list_versions_handler(
    packages: list[str] | None = None,
) -> dict[str, Any]:
    """List versions across the scitex ecosystem.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.

    Returns
    -------
    dict
        Version information for each package.
    """
    from scitex._dev import list_versions

    return list_versions(packages)


async def check_versions_handler(
    packages: list[str] | None = None,
) -> dict[str, Any]:
    """Check version consistency across the scitex ecosystem.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.

    Returns
    -------
    dict
        Detailed version check results with summary.
    """
    from scitex._dev import check_versions

    return check_versions(packages)


async def check_hosts_handler(
    packages: list[str] | None = None,
    hosts: list[str] | None = None,
) -> dict[str, Any]:
    """Check versions on SSH hosts.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    hosts : list[str] | None
        List of host names to check. If None, checks all enabled hosts.

    Returns
    -------
    dict
        Host name -> package name -> version info mapping.
    """
    from scitex._dev import check_all_hosts

    return check_all_hosts(packages=packages, hosts=hosts)


async def check_remotes_handler(
    packages: list[str] | None = None,
    remotes: list[str] | None = None,
) -> dict[str, Any]:
    """Check versions on GitHub remotes.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    remotes : list[str] | None
        List of remote names to check. If None, checks all enabled remotes.

    Returns
    -------
    dict
        Remote name -> package name -> version info mapping.
    """
    from scitex._dev import check_all_remotes

    return check_all_remotes(packages=packages, remotes=remotes)


async def get_config_handler() -> dict[str, Any]:
    """Get current developer configuration.

    Returns
    -------
    dict
        Configuration including packages, hosts, remotes, branches.
    """
    from scitex._dev import get_config_path, load_config

    config = load_config()
    return {
        "config_path": str(get_config_path()),
        "packages": [
            {
                "name": p.name,
                "local_path": p.local_path,
                "pypi_name": p.pypi_name,
                "github_repo": p.github_repo,
            }
            for p in config.packages
        ],
        "hosts": [
            {
                "name": h.name,
                "hostname": h.hostname,
                "user": h.user,
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


async def get_full_versions_handler(
    packages: list[str] | None = None,
    include_hosts: bool = False,
    include_remotes: bool = False,
) -> dict[str, Any]:
    """Get comprehensive version data from all sources.

    Parameters
    ----------
    packages : list[str] | None
        List of package names to check. If None, checks all ecosystem packages.
    include_hosts : bool
        Include SSH host version checks.
    include_remotes : bool
        Include GitHub remote version checks.

    Returns
    -------
    dict
        Combined version data from local, hosts, and remotes.
    """
    from scitex._dev import check_all_hosts, check_all_remotes, list_versions

    result = {
        "packages": list_versions(packages),
        "hosts": {},
        "remotes": {},
    }

    if include_hosts:
        try:
            result["hosts"] = check_all_hosts(packages=packages)
        except Exception as e:
            result["hosts"] = {"error": str(e)}

    if include_remotes:
        try:
            result["remotes"] = check_all_remotes(packages=packages)
        except Exception as e:
            result["remotes"] = {"error": str(e)}

    return result


# EOF
