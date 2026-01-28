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


@scholar.group(invoke_without_command=True)
@click.pass_context
def mcp(ctx):
    """
    MCP (Model Context Protocol) server operations

    \b
    Commands:
      start      - Start the MCP server
      doctor     - Check MCP server health
      list-tools - List available MCP tools

    \b
    Examples:
      scitex scholar mcp start
      scitex scholar mcp list-tools
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp.command()
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "sse", "http"]),
    default="stdio",
    help="Transport protocol (default: stdio)",
)
@click.option("--host", default="0.0.0.0", help="Host for HTTP/SSE (default: 0.0.0.0)")
@click.option(
    "--port", default=8097, type=int, help="Port for HTTP/SSE (default: 8097)"
)
def start(transport, host, port):
    """
    Start the scholar MCP server

    \b
    Examples:
      scitex scholar mcp start
      scitex scholar mcp start -t http --port 8097
    """
    import sys

    try:
        from scitex.scholar.mcp_server import main as run_server

        if transport != "stdio":
            click.secho(f"Starting scholar MCP server ({transport})", fg="cyan")
            click.echo(f"  Host: {host}")
            click.echo(f"  Port: {port}")

        run_server()

    except ImportError as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        click.echo("\nInstall dependencies: pip install fastmcp")
        sys.exit(1)
    except Exception as e:
        click.secho(f"Error: {e}", fg="red", err=True)
        sys.exit(1)


@mcp.command()
def doctor():
    """
    Check MCP server health and dependencies

    \b
    Example:
      scitex scholar mcp doctor
    """
    click.secho("Scholar MCP Server Health Check", fg="cyan", bold=True)
    click.echo()

    click.echo("Checking FastMCP... ", nl=False)
    try:
        import fastmcp  # noqa: F401

        click.secho("OK", fg="green")
    except ImportError:
        click.secho("NOT INSTALLED", fg="red")
        click.echo("  Install with: pip install fastmcp")

    click.echo("Checking scholar module... ", nl=False)
    try:
        from scitex import scholar as _  # noqa: F401

        click.secho("OK", fg="green")
    except ImportError as e:
        click.secho(f"FAIL ({e})", fg="red")

    click.echo("Checking crossref-local... ", nl=False)
    try:
        import crossref_local  # noqa: F401

        click.secho("OK", fg="green")
    except ImportError:
        click.secho("NOT INSTALLED (optional)", fg="yellow")


@mcp.command("list-tools")
def list_tools():
    """
    List available MCP tools

    \b
    Example:
      scitex scholar mcp list-tools
    """
    click.secho("Scholar MCP Tools", fg="cyan", bold=True)
    click.echo()
    tools = [
        ("scholar_search_papers", "Search for papers by query"),
        ("scholar_resolve_dois", "Resolve DOIs to metadata"),
        ("scholar_enrich_bibtex", "Enrich BibTeX with abstracts/DOIs"),
        ("scholar_download_pdf", "Download PDF for a paper"),
        ("scholar_download_pdfs_batch", "Batch download PDFs"),
        ("scholar_get_library_status", "Get library status"),
        ("scholar_parse_bibtex", "Parse BibTeX file"),
        ("scholar_validate_pdfs", "Validate downloaded PDFs"),
        ("scholar_authenticate", "Authenticate with institution"),
        ("scholar_check_auth_status", "Check authentication status"),
        ("scholar_fetch_papers", "Fetch papers by DOIs"),
        ("scholar_crossref_search", "Search CrossRef database"),
        ("scholar_crossref_get", "Get paper by DOI from CrossRef"),
    ]
    for name, desc in tools:
        click.echo(f"  {name}: {desc}")


scholar.add_command(crossref_scitex)
scholar.add_command(fetch)
scholar.add_command(library)
scholar.add_command(config)
scholar.add_command(jobs)

__all__ = ["scholar"]

# EOF
