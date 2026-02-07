#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/cli/dev.py

"""
SciTeX Developer CLI Commands (Internal).

Commands for managing and inspecting the scitex ecosystem.
"""

import click


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option("--help-recursive", is_flag=True, help="Show help for all subcommands")
@click.pass_context
def dev(ctx, help_recursive):
    """
    Developer utilities (internal).

    \b
    Examples:
      scitex dev versions              # List all ecosystem versions
      scitex dev versions --check      # Check version consistency
      scitex dev versions --json       # Output as JSON
      scitex dev versions -p scitex    # Check specific package
    """
    if help_recursive:
        from . import print_help_recursive

        print_help_recursive(ctx, dev)
        ctx.exit(0)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@dev.command("versions")
@click.option(
    "--check",
    is_flag=True,
    help="Check version consistency and show summary",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
@click.option(
    "-p",
    "--package",
    multiple=True,
    help="Filter to specific package(s)",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Skip remote (PyPI) version checks",
)
@click.option(
    "--host",
    multiple=True,
    help="Check specific SSH host(s)",
)
@click.option(
    "--all-hosts",
    is_flag=True,
    help="Check all configured SSH hosts",
)
@click.option(
    "--remote",
    multiple=True,
    help="Check specific GitHub remote(s)",
)
@click.option(
    "--all-remotes",
    is_flag=True,
    help="Check all configured GitHub remotes",
)
def versions(check, as_json, package, local_only, host, all_hosts, remote, all_remotes):
    """
    List versions across the scitex ecosystem.

    \b
    Shows version information from multiple sources:
      - pyproject.toml (local source)
      - installed package (pip/importlib.metadata)
      - git tag (latest version tag)
      - git branch (current branch)
      - PyPI (remote published version)
      - SSH hosts (if configured)
      - GitHub remotes (if configured)

    \b
    Examples:
      scitex dev versions                    # List all versions
      scitex dev versions --check            # Check consistency
      scitex dev versions --json             # JSON output
      scitex dev versions -p scitex          # Single package
      scitex dev versions -p scitex -p figrecipe  # Multiple packages
      scitex dev versions --local-only       # Skip PyPI checks
      scitex dev versions --host myhost      # Check specific host
      scitex dev versions --all-hosts        # Check all hosts
      scitex dev versions --remote ywatanabe1989  # Check GitHub remote
      scitex dev versions --all-remotes      # Check all remotes
    """
    import json as json_module

    from scitex._dev import check_versions, list_versions

    packages = list(package) if package else None

    if check:
        result = check_versions(packages)
        if local_only:
            for pkg_info in result["packages"].values():
                pkg_info.get("remote", {}).pop("pypi", None)
    else:
        result = list_versions(packages)
        if local_only:
            for pkg_info in result.values():
                pkg_info.get("remote", {}).pop("pypi", None)

    # Add host data if requested
    if host or all_hosts:
        from scitex._dev import check_all_hosts

        hosts_filter = list(host) if host else None
        try:
            result["hosts"] = check_all_hosts(packages=packages, hosts=hosts_filter)
        except Exception as e:
            result["hosts"] = {"error": str(e)}

    # Add remote data if requested
    if remote or all_remotes:
        from scitex._dev import check_all_remotes

        remotes_filter = list(remote) if remote else None
        try:
            result["remotes"] = check_all_remotes(
                packages=packages, remotes=remotes_filter
            )
        except Exception as e:
            result["remotes"] = {"error": str(e)}

    if as_json:
        click.echo(json_module.dumps(result, indent=2))
        return

    # Human-readable output
    if check:
        _print_check_result(result)
    else:
        _print_versions(result)

    # Print host data
    if "hosts" in result and result["hosts"]:
        _print_hosts(result["hosts"])

    # Print remote data
    if "remotes" in result and result["remotes"]:
        _print_remotes(result["remotes"])


def _print_versions(versions: dict) -> None:
    """Print version information in human-readable format."""
    click.secho("SciTeX Ecosystem Versions", fg="cyan", bold=True)
    click.echo("=" * 60)
    click.echo()

    for pkg, info in versions.items():
        status = info.get("status", "unknown")
        status_color = {
            "ok": "green",
            "unreleased": "yellow",
            "mismatch": "red",
            "outdated": "magenta",
            "unavailable": "white",
            "unknown": "white",
        }.get(status, "white")

        click.secho(f"{pkg}", fg="cyan", bold=True, nl=False)
        click.echo("  ", nl=False)
        click.secho(f"[{status}]", fg=status_color)

        # Local versions
        local = info.get("local", {})
        if local.get("pyproject_toml"):
            click.echo(f"    toml:      {local['pyproject_toml']}")
        if local.get("installed"):
            click.echo(f"    installed: {local['installed']}")

        # Git info
        git = info.get("git", {})
        if git.get("latest_tag"):
            click.echo(f"    git tag:   {git['latest_tag']}")
        if git.get("branch"):
            click.echo(f"    branch:    {git['branch']}")

        # Remote
        remote = info.get("remote", {})
        if remote.get("pypi"):
            click.echo(f"    pypi:      {remote['pypi']}")

        # Issues
        issues = info.get("issues", [])
        if issues:
            for issue in issues:
                click.secho(f"    ! {issue}", fg="yellow")

        click.echo()


def _print_check_result(result: dict) -> None:
    """Print version check result with summary."""
    _print_versions(result.get("packages", {}))

    summary = result.get("summary", {})
    click.secho("Summary", fg="cyan", bold=True)
    click.echo("-" * 30)

    total = summary.get("total", 0)
    ok = summary.get("ok", 0)
    unreleased = summary.get("unreleased", 0)
    mismatch = summary.get("mismatch", 0)
    outdated = summary.get("outdated", 0)
    unavailable = summary.get("unavailable", 0)

    click.echo(f"  Total:       {total}")
    click.secho(f"  OK:          {ok}", fg="green" if ok else "white")
    if unreleased:
        click.secho(f"  Unreleased:  {unreleased}", fg="yellow")
    if mismatch:
        click.secho(f"  Mismatch:    {mismatch}", fg="red")
    if outdated:
        click.secho(f"  Outdated:    {outdated}", fg="magenta")
    if unavailable:
        click.secho(f"  Unavailable: {unavailable}", fg="white")

    click.echo()
    if mismatch > 0:
        click.secho("Some packages have version mismatches!", fg="red", bold=True)
    elif unreleased > 0:
        click.secho("Some packages are ready to release.", fg="yellow")
    else:
        click.secho("All versions are consistent.", fg="green", bold=True)


def _print_hosts(hosts_data: dict) -> None:
    """Print host version data."""
    click.echo()
    click.secho("SSH Hosts", fg="cyan", bold=True)
    click.echo("-" * 40)

    if "error" in hosts_data:
        click.secho(f"  Error: {hosts_data['error']}", fg="red")
        return

    for host_name, host_info in hosts_data.items():
        if host_name.startswith("_"):
            continue
        click.secho(f"  {host_name}", fg="yellow", bold=True)
        meta = host_info.get("_host", {})
        if meta:
            click.echo(f"    ({meta.get('hostname', '')} - {meta.get('role', '')})")
        for pkg, pkg_info in host_info.items():
            if pkg.startswith("_"):
                continue
            status = pkg_info.get("status", "unknown")
            installed = pkg_info.get("installed", "-")
            color = (
                "green" if status == "ok" else "red" if status == "error" else "yellow"
            )
            click.echo(f"    {pkg}: ", nl=False)
            click.secho(f"{installed}", fg=color)


def _print_remotes(remotes_data: dict) -> None:
    """Print GitHub remote version data."""
    click.echo()
    click.secho("GitHub Remotes", fg="cyan", bold=True)
    click.echo("-" * 40)

    if "error" in remotes_data:
        click.secho(f"  Error: {remotes_data['error']}", fg="red")
        return

    for remote_name, remote_info in remotes_data.items():
        if remote_name.startswith("_"):
            continue
        click.secho(f"  {remote_name}", fg="yellow", bold=True)
        meta = remote_info.get("_remote", {})
        if meta:
            click.echo(f"    (org: {meta.get('org', '')})")
        for pkg, pkg_info in remote_info.items():
            if pkg.startswith("_"):
                continue
            tag = pkg_info.get("latest_tag", "-")
            release = pkg_info.get("release", "-")
            click.echo(f"    {pkg}: tag={tag}, release={release}")


# MCP subgroup
@dev.group(invoke_without_command=True)
@click.pass_context
def mcp(ctx):
    """
    MCP (Model Context Protocol) server operations.

    \b
    Commands:
      list-tools - List available MCP tools

    \b
    Examples:
      scitex dev mcp list-tools
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@mcp.command("list-tools")
@click.option("-v", "--verbose", count=True, help="-v params, -vv returns")
def list_tools(verbose):
    """List available MCP tools for dev module."""
    click.secho("Dev MCP Tools", fg="cyan", bold=True)
    click.echo()
    tools = [
        ("dev_list_versions", "List versions across ecosystem", "packages", "JSON"),
        ("dev_check_versions", "Check version consistency", "packages", "JSON"),
        ("dev_check_hosts", "Check versions on SSH hosts", "packages, hosts", "JSON"),
        ("dev_check_remotes", "Check versions on GitHub", "packages, remotes", "JSON"),
        ("dev_get_config", "Get current configuration", "", "JSON"),
        (
            "dev_full_versions",
            "Get comprehensive data",
            "packages, hosts, remotes",
            "JSON",
        ),
    ]
    for name, desc, params, returns in tools:
        click.secho(f"  {name}", fg="green", bold=True, nl=False)
        click.echo(f": {desc}")
        if verbose >= 1 and params:
            click.echo(f"    params: {params}")
        if verbose >= 2 and returns:
            click.echo(f"    returns: {returns}")


@dev.command("list-python-apis")
@click.option("-v", "--verbose", count=True, help="Verbosity: -v +doc, -vv full doc")
@click.option("-d", "--max-depth", type=int, default=5, help="Max recursion depth")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.pass_context
def list_python_apis(ctx, verbose, max_depth, as_json):
    """List Python APIs (alias for: scitex introspect api scitex._dev)."""
    from scitex.cli.introspect import api

    ctx.invoke(
        api,
        dotted_path="scitex._dev",
        verbose=verbose,
        max_depth=max_depth,
        as_json=as_json,
    )


@dev.command("dashboard")
@click.option("--host", default="127.0.0.1", help="Host to bind to")
@click.option("--port", "-p", default=5000, type=int, help="Port to listen on")
@click.option("--debug", is_flag=True, help="Enable debug mode")
@click.option("--no-browser", is_flag=True, help="Don't open browser")
@click.option("--force", is_flag=True, help="Kill existing process using the port")
def dashboard(host, port, debug, no_browser, force):
    """
    Start the Flask version dashboard.

    \b
    Examples:
      scitex dev dashboard              # Start on localhost:5000
      scitex dev dashboard --port 5001  # Custom port
      scitex dev dashboard --no-browser # Don't open browser
      scitex dev dashboard --force      # Kill existing and restart
    """
    from scitex._dev import run_dashboard

    run_dashboard(
        host=host, port=port, debug=debug, open_browser=not no_browser, force=force
    )


# Config subgroup
@dev.group(invoke_without_command=True)
@click.pass_context
def config(ctx):
    """
    Configuration management.

    \b
    Commands:
      show     - Show current configuration
      validate - Validate configuration file
      create   - Create default config file

    \b
    Examples:
      scitex dev config show
      scitex dev config create
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@config.command("show")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def config_show(as_json):
    """Show current configuration."""
    import json as json_module

    from scitex._dev import get_config_path, load_config

    config_path = get_config_path()
    cfg = load_config()

    if as_json:
        data = {
            "config_path": str(config_path),
            "exists": config_path.exists(),
            "packages": [p.name for p in cfg.packages],
            "hosts": [{"name": h.name, "enabled": h.enabled} for h in cfg.hosts],
            "remotes": [
                {"name": r.name, "enabled": r.enabled} for r in cfg.github_remotes
            ],
            "branches": cfg.branches,
        }
        click.echo(json_module.dumps(data, indent=2))
        return

    click.secho("Configuration", fg="cyan", bold=True)
    click.echo(f"  Path: {config_path}")
    click.echo(f"  Exists: {config_path.exists()}")
    click.echo()
    click.secho("Packages:", fg="yellow")
    for p in cfg.packages:
        click.echo(f"  - {p.name} ({p.pypi_name})")
    click.echo()
    click.secho("Hosts:", fg="yellow")
    for h in cfg.hosts:
        status = "enabled" if h.enabled else "disabled"
        click.echo(f"  - {h.name} ({h.hostname}) [{status}]")
    click.echo()
    click.secho("GitHub Remotes:", fg="yellow")
    for r in cfg.github_remotes:
        status = "enabled" if r.enabled else "disabled"
        click.echo(f"  - {r.name} (org: {r.org}) [{status}]")


@config.command("create")
@click.option("--force", is_flag=True, help="Overwrite existing config")
def config_create(force):
    """Create default configuration file."""
    from scitex._dev import create_default_config, get_config_path

    config_path = get_config_path()
    if config_path.exists() and not force:
        click.secho(f"Config already exists: {config_path}", fg="yellow")
        click.echo("Use --force to overwrite.")
        return

    path = create_default_config()
    click.secho(f"Created config: {path}", fg="green")


@config.command("validate")
def config_validate():
    """Validate configuration file."""
    from scitex._dev import get_config_path, load_config

    config_path = get_config_path()
    if not config_path.exists():
        click.secho(f"Config not found: {config_path}", fg="red")
        click.echo("Run 'scitex dev config create' to create one.")
        return

    try:
        cfg = load_config()
        click.secho("Configuration is valid.", fg="green")
        click.echo(f"  Packages: {len(cfg.packages)}")
        click.echo(f"  Hosts: {len(cfg.hosts)}")
        click.echo(f"  Remotes: {len(cfg.github_remotes)}")
    except Exception as e:
        click.secho(f"Configuration error: {e}", fg="red")


# Register clone command from separate module
from ._dev_clone import clone

dev.add_command(clone)


# EOF
