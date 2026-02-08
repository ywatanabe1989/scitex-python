#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_github.py

"""GitHub API integration for version checking."""

from __future__ import annotations

import os
import urllib.request
from typing import Any

from ._config import DevConfig, GitHubRemote, get_enabled_remotes, load_config


def _get_github_token() -> str | None:
    """Get GitHub token from environment."""
    return os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")


def _github_api_request(url: str) -> dict[str, Any] | list[Any] | None:
    """Make a GitHub API request.

    Parameters
    ----------
    url : str
        API URL to fetch.

    Returns
    -------
    dict | list | None
        JSON response or None on error.
    """
    import json

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "scitex-dev",
    }

    token = _get_github_token()
    if token:
        headers["Authorization"] = f"token {token}"

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


def get_github_tags(org: str, repo: str) -> list[str]:
    """Get tags from a GitHub repository.

    Parameters
    ----------
    org : str
        GitHub organization or user.
    repo : str
        Repository name.

    Returns
    -------
    list[str]
        List of tag names (most recent first).
    """
    url = f"https://api.github.com/repos/{org}/{repo}/tags?per_page=10"
    data = _github_api_request(url)

    if not data or not isinstance(data, list):
        return []

    return [tag.get("name", "") for tag in data if tag.get("name")]


def get_github_latest_tag(org: str, repo: str) -> str | None:
    """Get the latest version tag from a GitHub repository.

    Parameters
    ----------
    org : str
        GitHub organization or user.
    repo : str
        Repository name.

    Returns
    -------
    str | None
        Latest version tag or None.
    """
    tags = get_github_tags(org, repo)

    # Filter to version tags (v*)
    version_tags = [t for t in tags if t.startswith("v")]
    if version_tags:
        return version_tags[0]

    # Fallback: any tags
    return tags[0] if tags else None


def get_github_release(org: str, repo: str) -> dict[str, Any] | None:
    """Get the latest release from a GitHub repository.

    Parameters
    ----------
    org : str
        GitHub organization or user.
    repo : str
        Repository name.

    Returns
    -------
    dict | None
        Release info with keys: tag_name, name, published_at, prerelease.
    """
    url = f"https://api.github.com/repos/{org}/{repo}/releases/latest"
    data = _github_api_request(url)

    if not data or not isinstance(data, dict):
        return None

    return {
        "tag_name": data.get("tag_name"),
        "name": data.get("name"),
        "published_at": data.get("published_at"),
        "prerelease": data.get("prerelease", False),
    }


def get_github_repo_info(org: str, repo: str) -> dict[str, Any] | None:
    """Get basic info about a GitHub repository.

    Parameters
    ----------
    org : str
        GitHub organization or user.
    repo : str
        Repository name.

    Returns
    -------
    dict | None
        Repository info with keys: default_branch, description, stars, forks.
    """
    url = f"https://api.github.com/repos/{org}/{repo}"
    data = _github_api_request(url)

    if not data or not isinstance(data, dict):
        return None

    return {
        "default_branch": data.get("default_branch"),
        "description": data.get("description"),
        "stars": data.get("stargazers_count", 0),
        "forks": data.get("forks_count", 0),
        "private": data.get("private", False),
    }


def check_github_remote(
    remote: GitHubRemote,
    packages: list[str],
    config: DevConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Check versions for packages on a GitHub remote.

    Parameters
    ----------
    remote : GitHubRemote
        GitHub remote configuration.
    packages : list[str]
        List of package names to check.
    config : DevConfig | None
        Configuration for package -> repo mapping.

    Returns
    -------
    dict
        Package name -> version info mapping.
    """
    if config is None:
        config = load_config()

    from ._ecosystem import ECOSYSTEM

    results = {}

    for pkg in packages:
        # Get repo name from ecosystem or config
        repo = None
        if pkg in ECOSYSTEM:
            github_repo = ECOSYSTEM[pkg].get("github_repo", "")
            if github_repo:
                # Extract repo name (last part after /)
                repo = github_repo.split("/")[-1]

        # Check config packages
        if not repo:
            for pkg_conf in config.packages:
                if pkg_conf.name == pkg and pkg_conf.github_repo:
                    repo = pkg_conf.github_repo.split("/")[-1]
                    break

        if not repo:
            # Try package name as repo name
            repo = pkg

        # Get tag and release info
        try:
            latest_tag = get_github_latest_tag(remote.org, repo)
            release = get_github_release(remote.org, repo)

            results[pkg] = {
                "org": remote.org,
                "repo": repo,
                "latest_tag": latest_tag,
                "release": release.get("tag_name") if release else None,
                "release_date": release.get("published_at") if release else None,
                "status": "ok" if latest_tag else "no_tags",
            }
        except Exception as e:
            results[pkg] = {
                "org": remote.org,
                "repo": repo,
                "latest_tag": None,
                "release": None,
                "status": "error",
                "error": str(e),
            }

    return results


def check_all_remotes(
    packages: list[str] | None = None,
    remotes: list[str] | None = None,
    config: DevConfig | None = None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Check versions on all enabled GitHub remotes.

    Parameters
    ----------
    packages : list[str] | None
        List of package names. If None, uses ecosystem packages.
    remotes : list[str] | None
        List of remote names to check. If None, checks all enabled remotes.
    config : DevConfig | None
        Configuration to use. If None, loads default config.

    Returns
    -------
    dict
        Mapping: remote_name -> package_name -> version_info
    """
    if config is None:
        config = load_config()

    if packages is None:
        from ._ecosystem import get_all_packages

        packages = get_all_packages()

    enabled_remotes = get_enabled_remotes(config)
    if remotes:
        enabled_remotes = [r for r in enabled_remotes if r.name in remotes]

    results = {}

    for remote in enabled_remotes:
        results[remote.name] = check_github_remote(remote, packages, config)
        results[remote.name]["_remote"] = {
            "org": remote.org,
        }

    return results


def compare_with_local(
    packages: list[str] | None = None,
    config: DevConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Compare local versions with GitHub remotes.

    Parameters
    ----------
    packages : list[str] | None
        List of package names. If None, uses ecosystem packages.
    config : DevConfig | None
        Configuration to use.

    Returns
    -------
    dict
        Comparison results with local vs remote versions.
    """
    if config is None:
        config = load_config()

    if packages is None:
        from ._ecosystem import get_all_packages

        packages = get_all_packages()

    from ._versions import list_versions

    local_versions = list_versions(packages)
    remote_versions = check_all_remotes(packages, config=config)

    results = {}

    for pkg in packages:
        local_info = local_versions.get(pkg, {})
        local_tag = local_info.get("git", {}).get("latest_tag")
        local_toml = local_info.get("local", {}).get("pyproject_toml")

        pkg_result = {
            "local": {
                "tag": local_tag,
                "toml": local_toml,
            },
            "remotes": {},
            "sync_status": "ok",
            "issues": [],
        }

        for remote_name, remote_data in remote_versions.items():
            if remote_name.startswith("_"):
                continue

            pkg_remote = remote_data.get(pkg, {})
            remote_tag = pkg_remote.get("latest_tag")

            pkg_result["remotes"][remote_name] = {
                "tag": remote_tag,
                "release": pkg_remote.get("release"),
            }

            # Check sync status
            if local_tag and remote_tag and local_tag != remote_tag:
                pkg_result["issues"].append(
                    f"local tag ({local_tag}) != {remote_name} ({remote_tag})"
                )
                pkg_result["sync_status"] = "out_of_sync"

        if pkg_result["issues"]:
            pkg_result["sync_status"] = "out_of_sync"

        results[pkg] = pkg_result

    return results


# EOF
