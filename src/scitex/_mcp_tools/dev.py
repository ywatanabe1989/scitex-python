#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_mcp_tools/dev.py

"""MCP tool registration for developer utilities."""

from __future__ import annotations

import json


def _json(obj) -> str:
    """Serialize object to JSON string."""
    return json.dumps(obj, indent=2, default=str)


def register_dev_tools(mcp) -> None:
    """Register developer tools with FastMCP server."""

    @mcp.tool()
    async def dev_list_versions(
        packages: list[str] | None = None,
    ) -> str:
        """[dev] List versions across the scitex ecosystem.

        Shows version information from multiple sources:
        - pyproject.toml (local source)
        - installed package (importlib.metadata)
        - git tag (latest version tag)
        - git branch (current branch)
        - PyPI (remote published version)

        Parameters
        ----------
        packages : list[str] | None
            List of package names to check. If None, checks all ecosystem packages.
            Available packages: scitex, scitex-cloud, scitex-writer, scitex-dataset,
            figrecipe, crossref-local, openalex-local, socialia

        Returns
        -------
        str
            JSON with version information for each package.
        """
        from scitex._dev._mcp.handlers import list_versions_handler

        result = await list_versions_handler(packages)
        return _json(result)

    @mcp.tool()
    async def dev_check_versions(
        packages: list[str] | None = None,
    ) -> str:
        """[dev] Check version consistency across the scitex ecosystem.

        Checks for version mismatches between sources and returns a summary.

        Status values:
        - ok: All versions consistent
        - mismatch: Local sources disagree
        - unreleased: Local > PyPI (ready to release)
        - outdated: Local < PyPI (should update)
        - unavailable: Package not found locally

        Parameters
        ----------
        packages : list[str] | None
            List of package names to check. If None, checks all ecosystem packages.

        Returns
        -------
        str
            JSON with detailed version check results and summary.
        """
        from scitex._dev._mcp.handlers import check_versions_handler

        result = await check_versions_handler(packages)
        return _json(result)

    @mcp.tool()
    async def dev_check_hosts(
        packages: list[str] | None = None,
        hosts: list[str] | None = None,
    ) -> str:
        """[dev] Check package versions on SSH hosts.

        Connects to configured SSH hosts and checks installed package versions.
        Hosts are configured in ~/.scitex/dev_config.yaml or via SCITEX_DEV_HOSTS.

        Parameters
        ----------
        packages : list[str] | None
            List of package names to check. If None, checks all ecosystem packages.
        hosts : list[str] | None
            List of host names to check. If None, checks all enabled hosts.

        Returns
        -------
        str
            JSON with host -> package -> version mapping.
        """
        from scitex._dev._mcp.handlers import check_hosts_handler

        result = await check_hosts_handler(packages, hosts)
        return _json(result)

    @mcp.tool()
    async def dev_check_remotes(
        packages: list[str] | None = None,
        remotes: list[str] | None = None,
    ) -> str:
        """[dev] Check package versions on GitHub remotes.

        Fetches tags and releases from configured GitHub organizations.
        Remotes are configured in ~/.scitex/dev_config.yaml.

        Parameters
        ----------
        packages : list[str] | None
            List of package names to check. If None, checks all ecosystem packages.
        remotes : list[str] | None
            List of remote names to check. If None, checks all enabled remotes.

        Returns
        -------
        str
            JSON with remote -> package -> version mapping.
        """
        from scitex._dev._mcp.handlers import check_remotes_handler

        result = await check_remotes_handler(packages, remotes)
        return _json(result)

    @mcp.tool()
    async def dev_get_config() -> str:
        """[dev] Get current developer configuration.

        Returns the configuration from ~/.scitex/dev_config.yaml including:
        - Packages to track
        - SSH hosts
        - GitHub remotes
        - Branches to track

        Returns
        -------
        str
            JSON with current configuration.
        """
        from scitex._dev._mcp.handlers import get_config_handler

        result = await get_config_handler()
        return _json(result)

    @mcp.tool()
    async def dev_full_versions(
        packages: list[str] | None = None,
        include_hosts: bool = False,
        include_remotes: bool = True,
    ) -> str:
        """[dev] Get comprehensive version data from all sources.

        Combines local, host, and GitHub remote version information.

        Parameters
        ----------
        packages : list[str] | None
            List of package names to check. If None, checks all ecosystem packages.
        include_hosts : bool
            Include SSH host version checks (slower, requires SSH access).
        include_remotes : bool
            Include GitHub remote version checks.

        Returns
        -------
        str
            JSON with combined version data.
        """
        from scitex._dev._mcp.handlers import get_full_versions_handler

        result = await get_full_versions_handler(
            packages, include_hosts, include_remotes
        )
        return _json(result)


# EOF
