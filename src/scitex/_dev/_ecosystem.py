#!/usr/bin/env python3
# Timestamp: 2026-02-02
# File: scitex/_dev/_ecosystem.py

"""SciTeX ecosystem package registry."""

from pathlib import Path
from typing import TypedDict


class PackageInfo(TypedDict, total=False):
    """Package information structure."""

    local_path: str
    pypi_name: str
    github_repo: str
    import_name: str


ECOSYSTEM: dict[str, PackageInfo] = {
    "scitex": {
        "local_path": "~/proj/scitex-python",
        "pypi_name": "scitex",
        "github_repo": "ywatanabe1989/scitex-python",
        "import_name": "scitex",
    },
    "scitex-cloud": {
        "local_path": "~/proj/scitex-cloud",
        "pypi_name": "scitex-cloud",
        "github_repo": "ywatanabe1989/scitex-cloud",
        "import_name": "scitex_cloud",
    },
    "scitex-writer": {
        "local_path": "~/proj/scitex-writer",
        "pypi_name": "scitex-writer",
        "github_repo": "ywatanabe1989/scitex-writer",
        "import_name": "scitex_writer",
    },
    "scitex-dataset": {
        "local_path": "~/proj/scitex-dataset",
        "pypi_name": "scitex-dataset",
        "github_repo": "ywatanabe1989/scitex-dataset",
        "import_name": "scitex_dataset",
    },
    "figrecipe": {
        "local_path": "~/proj/figrecipe",
        "pypi_name": "figrecipe",
        "github_repo": "ywatanabe1989/figrecipe",
        "import_name": "figrecipe",
    },
    "crossref-local": {
        "local_path": "~/proj/crossref-local",
        "pypi_name": "crossref-local",
        "github_repo": "ywatanabe1989/crossref-local",
        "import_name": "crossref_local",
    },
    "openalex-local": {
        "local_path": "~/proj/openalex-local",
        "pypi_name": "openalex-local",
        "github_repo": "ywatanabe1989/openalex-local",
        "import_name": "openalex_local",
    },
    "socialia": {
        "local_path": "~/proj/socialia",
        "pypi_name": "socialia",
        "github_repo": "ywatanabe1989/socialia",
        "import_name": "socialia",
    },
}


def get_local_path(package: str) -> Path | None:
    """Get expanded local path for a package."""
    if package not in ECOSYSTEM:
        return None
    return Path(ECOSYSTEM[package]["local_path"]).expanduser()


def get_all_packages() -> list[str]:
    """Get list of all ecosystem package names."""
    return list(ECOSYSTEM.keys())


# EOF
