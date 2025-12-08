#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SciTeX CLI - Security Commands
"""

import sys
import click
from pathlib import Path

from scitex.security import (
    check_github_alerts,
    save_alerts_to_file,
    format_alerts_report,
    get_latest_alerts_file,
    GitHubSecurityError,
)


@click.group()
def security():
    """
    Security utilities - Check GitHub security alerts

    \b
    Examples:
      scitex security check                    # Check current repo
      scitex security check --repo owner/repo  # Check specific repo
      scitex security check --save             # Save to file
      scitex security latest                   # Show latest report
    """
    pass


@security.command()
@click.option(
    "--repo", help='Repository in format "owner/repo" (default: current repo)'
)
@click.option("--save", is_flag=True, help="Save report to file")
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory (default: ./logs/security)",
)
def check(repo, save, output_dir):
    """Check GitHub security alerts."""
    try:
        click.echo("Checking GitHub security alerts...")
        alerts = check_github_alerts(repo)

        # Count open alerts
        total = sum(
            len([a for a in alerts[key] if a.get("state") == "open"]) for key in alerts
        )

        if save:
            output_path = Path(output_dir) if output_dir else None
            file_path = save_alerts_to_file(alerts, output_path)
            click.echo(f"\nReport saved to: {file_path}")
            click.echo(f"Latest symlink: {file_path.parent / 'security-latest.txt'}")

        # Print report
        click.echo("\n" + format_alerts_report(alerts))

        # Exit with error code if alerts found
        if total > 0:
            click.secho(f"\n❌ Found {total} open security alert(s)", fg="red")
            sys.exit(1)
        else:
            click.secho("\n✓ No security alerts found", fg="green")
            sys.exit(0)

    except GitHubSecurityError as e:
        click.secho(f"ERROR: {e}", fg="red", err=True)
        sys.exit(1)


@security.command()
@click.option(
    "--dir",
    "security_dir",
    type=click.Path(),
    help="Security directory (default: ./logs/security)",
)
def latest(security_dir):
    """Show the latest security alerts file."""
    try:
        dir_path = Path(security_dir) if security_dir else None
        latest_file = get_latest_alerts_file(dir_path)

        if latest_file:
            click.echo(latest_file.read_text())
        else:
            click.secho("No security alerts files found", fg="yellow")
            sys.exit(1)

    except Exception as e:
        click.secho(f"ERROR: {e}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    security()
