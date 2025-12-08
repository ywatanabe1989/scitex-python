#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ~/proj/scitex-code/src/scitex/security/cli.py

"""
Command-line interface for SciTeX security utilities.

Usage:
    scitex-security check                    # Check current repo
    scitex-security check --repo owner/repo  # Check specific repo
    scitex-security check --save             # Save to file
"""

import sys
from pathlib import Path
from typing import Optional

try:
    import click
except ImportError:
    # Fallback if click not installed
    click = None

from .github import (
    check_github_alerts,
    save_alerts_to_file,
    format_alerts_report,
    get_latest_alerts_file,
    GitHubSecurityError,
)


def check_command(
    repo: Optional[str] = None,
    save: bool = False,
    output_dir: Optional[str] = None,
):
    """Check GitHub security alerts."""
    try:
        print("Checking GitHub security alerts...")
        alerts = check_github_alerts(repo)

        # Count open alerts
        total = sum(
            len([a for a in alerts[key] if a.get("state") == "open"]) for key in alerts
        )

        if save:
            output_path = Path(output_dir) if output_dir else None
            file_path = save_alerts_to_file(alerts, output_path)
            print(f"\nReport saved to: {file_path}")
            print(f"Latest symlink: {file_path.parent / 'security-latest.txt'}")

        # Print report
        print("\n" + format_alerts_report(alerts))

        # Exit with error code if alerts found
        if total > 0:
            print(f"\n❌ Found {total} open security alert(s)")
            sys.exit(1)
        else:
            print("\n✓ No security alerts found")
            sys.exit(0)

    except GitHubSecurityError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def latest_command(security_dir: Optional[str] = None):
    """Show the latest security alerts file."""
    try:
        dir_path = Path(security_dir) if security_dir else None
        latest_file = get_latest_alerts_file(dir_path)

        if latest_file:
            print(latest_file.read_text())
        else:
            print("No security alerts files found")
            sys.exit(1)

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SciTeX Security - GitHub security alerts checker"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Check command
    check_parser = subparsers.add_parser("check", help="Check GitHub security alerts")
    check_parser.add_argument(
        "--repo", help="Repository in format 'owner/repo' (default: current repo)"
    )
    check_parser.add_argument("--save", action="store_true", help="Save report to file")
    check_parser.add_argument(
        "--output-dir", help="Output directory (default: ./logs/security)"
    )

    # Latest command
    latest_parser = subparsers.add_parser("latest", help="Show latest security alerts")
    latest_parser.add_argument(
        "--dir",
        dest="security_dir",
        help="Security directory (default: ./logs/security)",
    )

    args = parser.parse_args()

    if args.command == "check":
        check_command(args.repo, args.save, args.output_dir)
    elif args.command == "latest":
        latest_command(args.security_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
