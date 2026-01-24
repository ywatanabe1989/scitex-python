#!/usr/bin/env python3
# Timestamp: 2026-01-14
# File: src/scitex/cli/scholar/__init__.py
# ----------------------------------------

"""Scholar CLI commands.

This module provides the command-line interface for SciTeX Scholar.

Usage:
    scitex scholar fetch "10.1038/nature12373"
    scitex scholar fetch --from-bibtex papers.bib --project myresearch
    scitex scholar fetch "10.1038/nature12373" --async
    scitex scholar library
    scitex scholar config
    scitex scholar jobs list
    scitex scholar jobs status <job_id>

CrossRef database (167M+ papers via crossref-local):
    scitex scholar crossref-scitex search "deep learning"
    scitex scholar crossref-scitex get 10.1038/nature12373
    scitex scholar crossref-scitex count "epilepsy seizure"
    scitex scholar crossref-scitex info
"""

from __future__ import annotations

import click

from ._crossref_scitex import crossref_scitex
from ._fetch import fetch
from ._jobs import jobs
from ._library import config, library


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--help-recursive", is_flag=True, help="Show help for all subcommands")
@click.pass_context
def scholar(ctx, help_recursive):
    """
    Scientific paper management

    \b
    Fetch papers, manage your library, and track background jobs.

    \b
    Examples:
        scitex scholar fetch "10.1038/nature12373"
        scitex scholar fetch --from-bibtex refs.bib -p myproject
        scitex scholar library
        scitex scholar jobs list
    """
    if help_recursive:
        _print_help_recursive(ctx)
        ctx.exit(0)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


def _print_help_recursive(ctx):
    """Print help for all commands recursively."""
    fake_parent = click.Context(click.Group(), info_name="scitex")
    parent_ctx = click.Context(scholar, info_name="scholar", parent=fake_parent)

    click.secho("━━━ scitex scholar ━━━", fg="cyan", bold=True)
    click.echo(scholar.get_help(parent_ctx))

    for name in sorted(scholar.list_commands(ctx) or []):
        cmd = scholar.get_command(ctx, name)
        if cmd is None:
            continue
        click.echo()
        click.secho(f"━━━ scitex scholar {name} ━━━", fg="cyan", bold=True)
        with click.Context(cmd, info_name=name, parent=parent_ctx) as sub_ctx:
            click.echo(cmd.get_help(sub_ctx))
            if isinstance(cmd, click.Group):
                for sub_name in sorted(cmd.list_commands(sub_ctx) or []):
                    sub_cmd = cmd.get_command(sub_ctx, sub_name)
                    if sub_cmd is None:
                        continue
                    click.echo()
                    click.secho(
                        f"━━━ scitex scholar {name} {sub_name} ━━━",
                        fg="cyan",
                        bold=True,
                    )
                    with click.Context(
                        sub_cmd, info_name=sub_name, parent=sub_ctx
                    ) as sub_sub_ctx:
                        click.echo(sub_cmd.get_help(sub_sub_ctx))


scholar.add_command(crossref_scitex)
scholar.add_command(fetch)
scholar.add_command(library)
scholar.add_command(config)
scholar.add_command(jobs)

__all__ = ["scholar"]

# EOF
