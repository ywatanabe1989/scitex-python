#!/usr/bin/env python3
# File: src/scitex/cli/cloud.py
"""
SciTeX Cloud Commands - Delegates to scitex-cloud package.

This module provides cloud/git operations by delegating to scitex_cloud.cli.gitea.
"""

import click

try:
    from scitex_cloud.cli.gitea import gitea as _gitea_group

    HAS_CLOUD = True
except ImportError:
    HAS_CLOUD = False
    _gitea_group = None


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cloud():
    r"""Cloud and Git operations (delegates to scitex-cloud).

    \b
    Provides standard git hosting operations:
    - Repository management (create, list, delete)
    - Cloning and forking
    - Pull requests and issues

    \b
    Backend: Gitea (git.scitex.ai)
    Requires: pip install scitex-cloud
    """
    if not HAS_CLOUD:
        click.echo("Error: scitex-cloud package not installed", err=True)
        click.echo("Install: pip install scitex-cloud", err=True)
        raise SystemExit(1)


# Delegate all commands from scitex_cloud.cli.gitea
if HAS_CLOUD and _gitea_group is not None:
    for name, cmd in _gitea_group.commands.items():
        cloud.add_command(cmd, name)


# EOF
