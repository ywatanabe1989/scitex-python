#!/usr/bin/env python3
"""Check version synchronization across GitHub tags, releases, and PyPI."""

import json
import subprocess
import sys
import urllib.request
from typing import NamedTuple


class VersionStatus(NamedTuple):
    version: str
    git_tag: bool
    gh_release: bool
    pypi: bool

    @property
    def synced(self) -> bool:
        return self.git_tag and self.gh_release and self.pypi


def get_git_tags() -> set[str]:
    """Get all version tags from git."""
    result = subprocess.run(
        ["git", "tag", "-l", "v*"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {tag.lstrip("v") for tag in result.stdout.strip().split("\n") if tag}


def get_gh_releases() -> set[str]:
    """Get all GitHub release versions."""
    result = subprocess.run(
        ["gh", "release", "list", "--limit", "100", "--json", "tagName"],
        capture_output=True,
        text=True,
        check=True,
    )
    releases = json.loads(result.stdout)
    return {r["tagName"].lstrip("v") for r in releases}


def get_pypi_versions(package: str = "scitex") -> set[str]:
    """Get all PyPI versions."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read())
    return set(data["releases"].keys())


def get_local_version() -> str:
    """Get version from pyproject.toml."""
    result = subprocess.run(
        ["grep", "^version = ", "pyproject.toml"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().split('"')[1]


def check_sync(recent_only: int = 10) -> list[VersionStatus]:
    """Check version sync status."""
    git_tags = get_git_tags()
    gh_releases = get_gh_releases()
    pypi_versions = get_pypi_versions()

    all_versions = git_tags | gh_releases | pypi_versions

    # Sort by version (newest first)
    def version_key(v: str):
        parts = v.replace("-", ".").split(".")
        return tuple(int(p) if p.isdigit() else 0 for p in parts)

    sorted_versions = sorted(all_versions, key=version_key, reverse=True)

    if recent_only:
        sorted_versions = sorted_versions[:recent_only]

    return [
        VersionStatus(
            version=v,
            git_tag=v in git_tags,
            gh_release=v in gh_releases,
            pypi=v in pypi_versions,
        )
        for v in sorted_versions
    ]


def print_status(statuses: list[VersionStatus]) -> bool:
    """Print status table and return True if all synced."""
    print("\n┌──────────┬─────────┬────────────┬───────┬────────┐")
    print("│ Version  │ Git Tag │ GH Release │ PyPI  │ Status │")
    print("├──────────┼─────────┼────────────┼───────┼────────┤")

    all_synced = True
    for s in statuses:
        tag = "✓" if s.git_tag else "✗"
        rel = "✓" if s.gh_release else "✗"
        pypi = "✓" if s.pypi else "✗"
        status = "✓ OK" if s.synced else "⚠ SYNC"

        if not s.synced:
            all_synced = False

        print(
            f"│ {s.version:8} │    {tag}    │     {rel}      │   {pypi}   │ {status:6} │"
        )

    print("└──────────┴─────────┴────────────┴───────┴────────┘\n")

    return all_synced


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check version sync status")
    parser.add_argument(
        "--recent", "-n", type=int, default=10, help="Number of recent versions to show"
    )
    parser.add_argument(
        "--check", action="store_true", help="Exit with error if out of sync"
    )
    parser.add_argument(
        "--local", action="store_true", help="Check if local version is released"
    )
    args = parser.parse_args()

    statuses = check_sync(args.recent)

    if args.local:
        local = get_local_version()
        print(f"Local version: {local}")
        status = next((s for s in statuses if s.version == local), None)
        if status:
            if status.synced:
                print(f"✓ Version {local} is fully released")
            else:
                print(f"⚠ Version {local} is NOT fully synced:")
                print(f"  Git tag: {'✓' if status.git_tag else '✗'}")
                print(f"  GH Release: {'✓' if status.gh_release else '✗'}")
                print(f"  PyPI: {'✓' if status.pypi else '✗'}")
                if args.check:
                    sys.exit(1)
        else:
            print(f"✗ Version {local} not released anywhere yet")
        return

    all_synced = print_status(statuses)

    if args.check and not all_synced:
        print("ERROR: Versions are out of sync!")
        sys.exit(1)

    if all_synced:
        print("✓ All versions are synchronized")


if __name__ == "__main__":
    main()
