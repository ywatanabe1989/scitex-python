#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX CLI Main Entry Point
"""

import os
import sys
import click
from . import cloud, scholar, security, web


@click.group(context_settings={'help_option_names': ['-h', '--help']})
@click.version_option()
def cli():
    """
    SciTeX - Integrated Scientific Research Platform

    \b
    Commands:
      cloud      Git operations (via Gitea) - AVAILABLE NOW
      scholar    Literature management - AVAILABLE NOW
      security   GitHub security alerts - AVAILABLE NOW
      web        Web scraping utilities - AVAILABLE NOW
      completion Install shell tab-completion - AVAILABLE NOW
      code       Analysis execution (coming soon)
      viz        Visualization (coming soon)
      writer     Manuscript writing (coming soon)
      project    Integrated workflows (coming soon)

    \b
    Examples:
      scitex cloud login
      scitex cloud clone ywatanabe/my-project
      scitex scholar bibtex papers.bib --project myresearch
      scitex scholar single --doi "10.1038/nature12373"
      scitex security check --save
      scitex web get-urls https://example.com
      scitex web download-images https://example.com --output ./downloads

    \b
    Enable tab-completion:
      scitex completion          # Auto-install for your shell
      scitex completion --show   # Show installation instructions
    """
    pass


# Add command groups
cli.add_command(cloud.cloud)
cli.add_command(scholar.scholar)
cli.add_command(security.security)
cli.add_command(web.web)


@cli.command()
@click.option(
    '--shell',
    type=click.Choice(['bash', 'zsh', 'fish'], case_sensitive=False),
    help='Shell type (auto-detected if not provided)'
)
@click.option(
    '--show',
    is_flag=True,
    help='Show completion script instead of installing'
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
        shell_env = os.environ.get('SHELL', '')
        if 'bash' in shell_env:
            shell = 'bash'
        elif 'zsh' in shell_env:
            shell = 'zsh'
        elif 'fish' in shell_env:
            shell = 'fish'
        else:
            click.secho(
                "Could not auto-detect shell. Please specify with --shell option.",
                fg='red',
                err=True
            )
            sys.exit(1)

    shell = shell.lower()

    # Generate completion script
    if shell == 'bash':
        completion_script = '_SCITEX_COMPLETE=bash_source scitex'
        rc_file = os.path.expanduser('~/.bashrc')
        eval_line = 'eval "$(_SCITEX_COMPLETE=bash_source scitex)"'
    elif shell == 'zsh':
        completion_script = '_SCITEX_COMPLETE=zsh_source scitex'
        rc_file = os.path.expanduser('~/.zshrc')
        eval_line = 'eval "$(_SCITEX_COMPLETE=zsh_source scitex)"'
    elif shell == 'fish':
        completion_script = '_SCITEX_COMPLETE=fish_source scitex'
        rc_file = os.path.expanduser('~/.config/fish/config.fish')
        eval_line = 'eval (env _SCITEX_COMPLETE=fish_source scitex)'

    if show:
        # Just show the completion script
        click.echo(f"Add this line to your {rc_file}:")
        click.echo()
        click.secho(eval_line, fg='green')
        sys.exit(0)

    # Check if already installed
    if os.path.exists(rc_file):
        with open(rc_file, 'r') as f:
            content = f.read()
            if eval_line in content:
                click.secho(
                    f"Tab completion is already installed in {rc_file}",
                    fg='yellow'
                )
                click.echo()
                click.echo("To reload, run:")
                click.secho(f"  source {rc_file}", fg='cyan')
                sys.exit(0)

    # Install completion
    try:
        # Create config directory if it doesn't exist (for fish)
        os.makedirs(os.path.dirname(rc_file), exist_ok=True)

        with open(rc_file, 'a') as f:
            f.write(f"\n# SciTeX tab completion\n")
            f.write(f"{eval_line}\n")

        click.secho(f"Successfully installed tab completion to {rc_file}", fg='green')
        click.echo()
        click.echo("To activate completion in current shell, run:")
        click.secho(f"  source {rc_file}", fg='cyan')
        click.echo()
        click.echo("Or restart your shell.")
        sys.exit(0)

    except Exception as e:
        click.secho(f"ERROR: Failed to install completion: {e}", fg='red', err=True)
        click.echo()
        click.echo("You can manually add this line to your shell config:")
        click.secho(eval_line, fg='green')
        sys.exit(1)


if __name__ == '__main__':
    cli()
