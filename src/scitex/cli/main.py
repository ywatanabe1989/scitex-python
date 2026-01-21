#!/usr/bin/env python3
"""
SciTeX CLI Main Entry Point
"""

import os
import sys

import click

from . import (
    audio,
    browser,
    capture,
    cloud,
    config,
    convert,
    introspect,
    mcp,
    repro,
    resource,
    scholar,
    security,
    social,
    stats,
    template,
    tex,
    web,
    writer,
)


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.version_option()
@click.option("--help-recursive", is_flag=True, help="Show help for all commands")
@click.pass_context
def cli(ctx, help_recursive):
    """
    SciTeX - Integrated Scientific Research Platform

    \b
    Examples:
      scitex config list                    # Show configured paths
      scitex cloud clone user/project       # Clone from cloud
      scitex scholar bibtex papers.bib      # Manage papers
      scitex audio speak "Hello"            # Text-to-speech
      scitex mcp list-tools                 # List MCP tools
      scitex mcp start                      # Start MCP server

    \b
    Enable tab-completion:
      scitex completion          # Auto-install for your shell
      scitex completion --show   # Show installation instructions
    """
    if help_recursive:
        _print_help_recursive(ctx)
        ctx.exit(0)
    elif ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# Add command groups
cli.add_command(audio.audio)
cli.add_command(browser.browser)
cli.add_command(capture.capture)
cli.add_command(cloud.cloud)
cli.add_command(config.config)
cli.add_command(convert.convert)
cli.add_command(introspect.introspect)
cli.add_command(mcp.mcp)
cli.add_command(repro.repro)
cli.add_command(resource.resource)
cli.add_command(scholar.scholar)
cli.add_command(security.security)
cli.add_command(social.social)
cli.add_command(stats.stats)
cli.add_command(template.template)
cli.add_command(tex.tex)
cli.add_command(web.web)
cli.add_command(writer.writer)


def _get_all_command_paths(group, prefix=""):
    """Recursively get all command paths from a Click group."""
    paths = []
    for name in sorted(group.list_commands(None) or []):
        cmd = group.get_command(None, name)
        if cmd is None:
            continue
        full_path = f"{prefix} {name}".strip() if prefix else name
        paths.append((full_path, cmd))
        if isinstance(cmd, click.MultiCommand):
            paths.extend(_get_all_command_paths(cmd, full_path))
    return paths


def _print_help_recursive(ctx):
    """Print help for all commands recursively."""
    # Show main CLI help first
    click.secho("━━━ scitex ━━━", fg="cyan", bold=True)
    click.echo(cli.get_help(ctx))

    # Show help for each command using Click's internal help generation
    command_paths = _get_all_command_paths(cli)
    for cmd_path, cmd in command_paths:
        click.echo()
        click.secho(f"━━━ scitex {cmd_path} ━━━", fg="cyan", bold=True)
        # Create a new context for this command
        with click.Context(cmd, info_name=cmd_path, parent=ctx) as sub_ctx:
            click.echo(cmd.get_help(sub_ctx))


def _detect_shell() -> str | None:
    """Auto-detect current shell."""
    shell_env = os.environ.get("SHELL", "")
    if "bash" in shell_env:
        return "bash"
    elif "zsh" in shell_env:
        return "zsh"
    elif "fish" in shell_env:
        return "fish"
    return None


def _get_rc_file(shell: str) -> str:
    """Get shell config file path."""
    if shell == "bash":
        return os.path.expanduser("~/.bashrc")
    elif shell == "zsh":
        return os.path.expanduser("~/.zshrc")
    elif shell == "fish":
        return os.path.expanduser("~/.config/fish/config.fish")
    return ""


def _get_ecosystem_clis() -> list[dict]:
    """Get info about SciTeX ecosystem CLIs."""
    import shutil

    clis = [
        {"name": "scitex", "type": "click", "env_var": "_SCITEX_COMPLETE"},
        {"name": "figrecipe", "type": "click", "env_var": "_FIGRECIPE_COMPLETE"},
        {
            "name": "crossref-local",
            "type": "click",
            "env_var": "_CROSSREF_LOCAL_COMPLETE",
        },
        {"name": "socialia", "type": "argcomplete", "env_var": None},
    ]
    for cli_info in clis:
        cli_info["available"] = shutil.which(cli_info["name"]) is not None
    return clis


def _generate_completion_lines(shell: str) -> list[str]:
    """Generate completion lines for all ecosystem CLIs."""
    import shutil

    lines = ["# SciTeX ecosystem tab completion"]
    clis = _get_ecosystem_clis()

    for cli_info in clis:
        if not cli_info["available"]:
            continue

        cli_path = shutil.which(cli_info["name"])
        if cli_info["type"] == "click":
            if shell == "bash":
                lines.append(f'eval "$({cli_info["env_var"]}=bash_source {cli_path})"')
            elif shell == "zsh":
                lines.append(f'eval "$({cli_info["env_var"]}=zsh_source {cli_path})"')
            elif shell == "fish":
                lines.append(f"eval (env {cli_info['env_var']}=fish_source {cli_path})")
        elif cli_info["type"] == "argcomplete":
            if shell in ("bash", "zsh"):
                lines.append(
                    f'eval "$(register-python-argcomplete {cli_info["name"]})"'
                )

    return lines


@cli.group(invoke_without_command=True)
@click.pass_context
def completion(ctx):
    """
    Shell completion for SciTeX ecosystem.

    \b
    Commands:
      scitex completion install   # Install completion (default)
      scitex completion status    # Check installation status
      scitex completion bash      # Show bash completion script
      scitex completion zsh       # Show zsh completion script

    \b
    Quick install:
      scitex completion install
    """
    if ctx.invoked_subcommand is None:
        # Default to install
        ctx.invoke(completion_install)


@completion.command("install")
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    help="Shell type (auto-detected if not provided).",
)
def completion_install(shell):
    """
    Install shell completion for SciTeX ecosystem.

    \b
    Examples:
      scitex completion install           # Auto-detect shell
      scitex completion install --shell bash
    """
    if not shell:
        shell = _detect_shell()
        if not shell:
            click.secho(
                "Could not auto-detect shell. Please specify with --shell option.",
                fg="red",
                err=True,
            )
            sys.exit(1)

    shell = shell.lower()
    rc_file = _get_rc_file(shell)
    lines = _generate_completion_lines(shell)

    if len(lines) <= 1:
        click.secho("No SciTeX ecosystem CLIs found.", fg="yellow")
        sys.exit(1)

    completion_block = "\n".join(lines)

    # Check if already installed
    if os.path.exists(rc_file):
        with open(rc_file) as f:
            content = f.read()
            if "SciTeX ecosystem tab completion" in content:
                click.secho(f"Completion already installed in {rc_file}", fg="yellow")
                click.echo(
                    "\nTo reinstall, first remove the existing block, then run again."
                )
                click.echo("\nTo reload, run:")
                click.secho(f"  source {rc_file}", fg="cyan")
                sys.exit(0)

    # Install
    try:
        os.makedirs(os.path.dirname(rc_file), exist_ok=True)
        with open(rc_file, "a") as f:
            f.write(f"\n{completion_block}\n")

        click.secho(f"Installed completion to {rc_file}", fg="green")
        click.echo("\nCLIs with completion:")
        for cli_info in _get_ecosystem_clis():
            status = (
                click.style("OK", fg="green")
                if cli_info["available"]
                else click.style("not found", fg="yellow")
            )
            click.echo(f"  {cli_info['name']}: {status}")
        click.echo("\nTo activate, run:")
        click.secho(f"  source {rc_file}", fg="cyan")

    except Exception as e:
        click.secho(f"ERROR: {e}", fg="red", err=True)
        click.echo("\nManually add to your shell config:")
        click.echo(completion_block)
        sys.exit(1)


@completion.command("status")
def completion_status():
    """
    Check shell completion installation status.

    \b
    Shows:
      - Current shell
      - Installed completions
      - Available ecosystem CLIs
    """
    shell = _detect_shell() or "unknown"
    rc_file = _get_rc_file(shell) if shell != "unknown" else "N/A"

    click.secho("Shell Completion Status", fg="cyan", bold=True)
    click.echo(f"  Shell: {shell}")
    click.echo(f"  Config: {rc_file}")

    # Check if installed
    installed = False
    if rc_file != "N/A" and os.path.exists(rc_file):
        with open(rc_file) as f:
            if "SciTeX ecosystem tab completion" in f.read():
                installed = True

    status = (
        click.style("installed", fg="green")
        if installed
        else click.style("not installed", fg="yellow")
    )
    click.echo(f"  Status: {status}")

    click.echo("\nEcosystem CLIs:")
    for cli_info in _get_ecosystem_clis():
        avail = (
            click.style("OK", fg="green")
            if cli_info["available"]
            else click.style("missing", fg="red")
        )
        click.echo(f"  {cli_info['name']}: {avail}")

    if not installed:
        click.echo("\nTo install completion:")
        click.secho("  scitex completion install", fg="cyan")


@completion.command("bash")
def completion_bash():
    """Show bash completion script."""
    lines = _generate_completion_lines("bash")
    click.echo("\n".join(lines))


@completion.command("zsh")
def completion_zsh():
    """Show zsh completion script."""
    lines = _generate_completion_lines("zsh")
    click.echo("\n".join(lines))


@completion.command("fish")
def completion_fish():
    """Show fish completion script."""
    lines = _generate_completion_lines("fish")
    click.echo("\n".join(lines))


if __name__ == "__main__":
    cli()
