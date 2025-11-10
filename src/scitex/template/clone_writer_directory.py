#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-30 08:47:48 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/clone_writer_directory.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/template/clone_writer_directory.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Create a new paper directory from the scitex-writer template.
"""

import sys
from typing import Optional

from ._clone_project import clone_project

TEMPLATE_REPO_URL = "https://github.com/ywatanabe1989/scitex-writer.git"


def clone_writer_directory(
    project_name: str,
    target_dir: Optional[str] = None,
    git_strategy: Optional[str] = "child",
    branch: Optional[str] = None,
) -> bool:
    """
    Create a new paper directory from the scitex-writer template repository.

    Parameters
    ----------
    project_name : str
        Name of the new paper directory/project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.
    git_strategy : str, optional
        Git initialization strategy ('child', 'parent', None). Default is 'child'.
    branch : str, optional
        Specific branch of the template repository to clone. If None, clones the default branch.

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from scitex.template import clone_writer_directory
    >>> clone_writer_directory("my_paper")
    >>> clone_writer_directory("my_paper", branch="develop")
    """
    return clone_project(
        project_name,
        target_dir,
        TEMPLATE_REPO_URL,
        "scitex-writer",
        git_strategy,
        branch,
    )


def main(args: list = None) -> None:
    """
    Command-line interface for clone_writer_directory.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If None, uses sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print(
            "Usage: python -m scitex clone_writer_directory <project-name> [target-dir]"
        )
        print("")
        print("Arguments:")
        print("  project-name  Name of the new paper directory/project")
        print(
            "  target-dir    Optional: Directory where project will be created (default: current directory)"
        )
        print("")
        print("Example:")
        print("  python -m scitex clone_writer_directory my_paper")
        print("  python -m scitex clone_writer_directory my_paper ~/papers")
        sys.exit(1)

    project_name = args[0]
    target_dir = args[1] if len(args) > 1 else None

    success = clone_writer_directory(project_name, target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# EOF
