#!/usr/bin/env python3
"""
SciTeX CLI - Command-line interface for SciTeX platform

Provides unified interface for:
- Cloud operations (wraps tea for Gitea)
- Scholar operations (Django API)
- Code operations (Django API)
- Viz operations (Django API)
- Writer operations (Django API)
- Project operations (integrated workflows)
"""

import click

from .main import cli


def print_help_recursive(ctx, group: click.Group) -> None:
    """Print help for a group and all its subcommands.

    Args:
        ctx: The click context
        group: The click group to print help for
    """
    # Create a fake parent context to show correct command path in Usage
    fake_parent = click.Context(click.Group(), info_name="scitex")
    parent_ctx = click.Context(group, info_name=group.name, parent=fake_parent)

    click.secho(f"━━━ scitex {group.name} ━━━", fg="cyan", bold=True)
    click.echo(group.get_help(parent_ctx))

    for name in sorted(group.list_commands(ctx) or []):
        cmd = group.get_command(ctx, name)
        if cmd is None:
            continue
        click.echo()
        click.secho(f"━━━ scitex {group.name} {name} ━━━", fg="cyan", bold=True)
        with click.Context(cmd, info_name=name, parent=parent_ctx) as sub_ctx:
            click.echo(cmd.get_help(sub_ctx))
            # Handle nested subgroups
            if isinstance(cmd, click.Group):
                for sub_name in sorted(cmd.list_commands(sub_ctx) or []):
                    sub_cmd = cmd.get_command(sub_ctx, sub_name)
                    if sub_cmd is None:
                        continue
                    click.echo()
                    click.secho(
                        f"━━━ scitex {group.name} {name} {sub_name} ━━━",
                        fg="cyan",
                        bold=True,
                    )
                    with click.Context(
                        sub_cmd, info_name=sub_name, parent=sub_ctx
                    ) as sub_sub_ctx:
                        click.echo(sub_cmd.get_help(sub_sub_ctx))


__all__ = ["cli", "print_help_recursive"]
