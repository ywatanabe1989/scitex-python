#!/usr/bin/env python3
# Timestamp: 2026-01-29
# File: src/scitex/cli/scholar/_openalex_scitex.py
"""OpenAlex-SciTeX CLI - Thin wrapper delegating to openalex-local.

This module provides access to the local OpenAlex database (284M+ works)
by delegating directly to openalex-local CLI without any modifications.
"""

from __future__ import annotations

import sys

import click


@click.command(
    "openalex-scitex",
    context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
)
@click.option("--help-recursive", is_flag=True, help="Show help for all commands")
@click.pass_context
def openalex_scitex(ctx, help_recursive):
    """
    OpenAlex-SciTeX database search (284M+ works)

    \b
    Thin wrapper for openalex-local. All arguments passed directly.
    Run 'openalex-local --help' for full options.

    \b
    Examples:
        scitex scholar openalex-scitex search "machine learning"
        scitex scholar openalex-scitex search-by-doi 10.1038/nature12373
        scitex scholar openalex-scitex status
    """
    try:
        from openalex_local.cli import cli as openalex_cli
    except ImportError:
        click.secho(
            "openalex-local not installed. Install with: pip install openalex-local",
            fg="red",
        )
        sys.exit(1)

    # Handle --help-recursive by delegating to openalex-local
    args = ctx.args
    if help_recursive:
        args = ["--help-recursive"]

    # Delegate all arguments to openalex-local CLI
    sys.exit(openalex_cli.main(args, standalone_mode=False))


# EOF
