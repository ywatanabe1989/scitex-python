#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/__init__.py

"""
SciTeX Developer Utilities (Internal Module).

This module provides internal developer tools for managing the scitex ecosystem.

Functions
---------
list_versions : List versions across ecosystem packages
check_versions : Check version consistency
load_config : Load developer configuration
check_all_hosts : Check versions on SSH hosts
check_all_remotes : Check versions on GitHub remotes
run_dashboard : Run the Flask version dashboard

Examples
--------
>>> from scitex._dev import list_versions
>>> versions = list_versions()
>>> versions["scitex"]["local"]["pyproject_toml"]
'2.17.1'

>>> from scitex._dev import check_versions
>>> result = check_versions(["scitex", "figrecipe"])
>>> result["summary"]["ok"]
2

>>> from scitex._dev import load_config
>>> config = load_config()
>>> len(config.packages)
8
"""

from ._config import (
    DevConfig,
    GitHubRemote,
    HostConfig,
    PackageConfig,
    PyPIAccount,
    create_default_config,
    get_config_path,
    get_enabled_hosts,
    get_enabled_remotes,
    load_config,
)
from ._ecosystem import ECOSYSTEM, get_all_packages, get_local_path
from ._github import (
    check_all_remotes,
    compare_with_local,
    get_github_latest_tag,
    get_github_release,
    get_github_tags,
)
from ._ssh import (
    check_all_hosts,
    get_remote_version,
    get_remote_versions,
    test_host_connection,
)
from ._versions import check_versions, list_versions

__all__ = [
    # Versions
    "list_versions",
    "check_versions",
    # Ecosystem
    "ECOSYSTEM",
    "get_all_packages",
    "get_local_path",
    # Config
    "load_config",
    "get_config_path",
    "create_default_config",
    "get_enabled_hosts",
    "get_enabled_remotes",
    "DevConfig",
    "HostConfig",
    "GitHubRemote",
    "PackageConfig",
    "PyPIAccount",
    # SSH
    "check_all_hosts",
    "get_remote_version",
    "get_remote_versions",
    "test_host_connection",
    # GitHub
    "check_all_remotes",
    "compare_with_local",
    "get_github_tags",
    "get_github_latest_tag",
    "get_github_release",
]


def run_dashboard(
    host: str = "127.0.0.1",
    port: int = 5000,
    debug: bool = False,
    open_browser: bool = True,
    force: bool = False,
) -> None:
    """Run the Flask version dashboard.

    Parameters
    ----------
    host : str
        Host to bind to.
    port : int
        Port to listen on.
    debug : bool
        Enable debug mode.
    open_browser : bool
        Open browser automatically.
    force : bool
        Kill existing process using the port if any.
    """
    from ._dashboard import run_dashboard as _run

    _run(host=host, port=port, debug=debug, open_browser=open_browser, force=force)


# EOF
