#!/usr/bin/env python3
"""
SciTeX CLI Main Entry Point
"""

import os
import sys

import click

from . import (
    audio,
    capture,
    cloud,
    config,
    convert,
    repro,
    resource,
    scholar,
    security,
    stats,
    template,
    tex,
    web,
    writer,
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option()
def cli():
    """
    SciTeX - Integrated Scientific Research Platform

    \b
    Examples:
      scitex config list                    # Show all configured paths
      scitex config init                    # Initialize directories
      scitex cloud login
      scitex cloud clone ywatanabe/my-project
      scitex scholar bibtex papers.bib --project myresearch
      scitex scholar single --doi "10.1038/nature12373"
      scitex security check --save
      scitex web get-urls https://example.com
      scitex web download-images https://example.com --output ./downloads
      scitex audio speak "Hello world"
      scitex capture snap --output screenshot.jpg
      scitex resource usage
      scitex stats recommend --data data.csv

    \b
    Enable tab-completion:
      scitex completion          # Auto-install for your shell
      scitex completion --show   # Show installation instructions
    """
    pass


# Add command groups
cli.add_command(audio.audio)
cli.add_command(capture.capture)
cli.add_command(cloud.cloud)
cli.add_command(config.config)
cli.add_command(convert.convert)
cli.add_command(repro.repro)
cli.add_command(resource.resource)
cli.add_command(scholar.scholar)
cli.add_command(security.security)
cli.add_command(stats.stats)
cli.add_command(template.template)
cli.add_command(tex.tex)
cli.add_command(web.web)
cli.add_command(writer.writer)


@cli.command()
@click.option(
    "--shell",
    type=click.Choice(["bash", "zsh", "fish"], case_sensitive=False),
    help="Shell type (auto-detected if not provided)",
)
@click.option(
    "--show", is_flag=True, help="Show completion script instead of installing"
)
def completion(shell, show):
    """
    Install or show shell completion for scitex commands.

    \b
    Supported shells: bash, zsh, fish

    \b
    Installation:
      # Auto-detect shell and install
      scitex completion

      # Specify shell
      scitex completion --shell bash
      scitex completion --shell zsh

      # Show completion script
      scitex completion --show

    \b
    After installation, restart your shell or run:
      source ~/.bashrc    # for bash
      source ~/.zshrc     # for zsh
    """
    # Auto-detect shell if not provided
    if not shell:
        shell_env = os.environ.get("SHELL", "")
        if "bash" in shell_env:
            shell = "bash"
        elif "zsh" in shell_env:
            shell = "zsh"
        elif "fish" in shell_env:
            shell = "fish"
        else:
            click.secho(
                "Could not auto-detect shell. Please specify with --shell option.",
                fg="red",
                err=True,
            )
            sys.exit(1)

    shell = shell.lower()

    # Get full path to scitex executable
    scitex_path = sys.argv[0]
    if not os.path.isabs(scitex_path):
        # If relative path, find the full path
        import shutil

        scitex_full = shutil.which("scitex") or scitex_path
    else:
        scitex_full = scitex_path

    # Generate completion script
    if shell == "bash":
        rc_file = os.path.expanduser("~/.bashrc")
        eval_line = f'eval "$(_SCITEX_COMPLETE=bash_source {scitex_full})"'
    elif shell == "zsh":
        rc_file = os.path.expanduser("~/.zshrc")
        eval_line = f'eval "$(_SCITEX_COMPLETE=zsh_source {scitex_full})"'
    elif shell == "fish":
        rc_file = os.path.expanduser("~/.config/fish/config.fish")
        eval_line = f"eval (env _SCITEX_COMPLETE=fish_source {scitex_full})"

    if show:
        # Just show the completion script
        click.echo(f"Add this line to your {rc_file}:")
        click.echo()
        click.secho(eval_line, fg="green")
        sys.exit(0)

    # Check if already installed (and not commented out)
    if os.path.exists(rc_file):
        with open(rc_file) as f:
            for line in f:
                # Check if the line exists and is not commented
                stripped = line.strip()
                if stripped == eval_line and not stripped.startswith("#"):
                    click.secho(
                        f"Tab completion is already installed in {rc_file}", fg="yellow"
                    )
                    click.echo()
                    click.echo("To reload, run:")
                    click.secho(f"  source {rc_file}", fg="cyan")
                    sys.exit(0)

    # Install completion
    try:
        # Create config directory if it doesn't exist (for fish)
        os.makedirs(os.path.dirname(rc_file), exist_ok=True)

        with open(rc_file, "a") as f:
            f.write("\n# SciTeX tab completion\n")
            f.write(f"{eval_line}\n")

        click.secho(f"Successfully installed tab completion to {rc_file}", fg="green")
        click.echo()
        click.echo("To activate completion in current shell, run:")
        click.secho(f"  source {rc_file}", fg="cyan")
        click.echo()
        click.echo("Or restart your shell.")
        sys.exit(0)

    except Exception as e:
        click.secho(f"ERROR: Failed to install completion: {e}", fg="red", err=True)
        click.echo()
        click.echo("You can manually add this line to your shell config:")
        click.secho(eval_line, fg="green")
        sys.exit(1)


if __name__ == "__main__":
    cli()
