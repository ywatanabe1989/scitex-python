#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 05:56:36 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/clone_pip_project.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/template/clone_pip_project.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Create a new pip project from the pip-project-template.
"""

import sys
from typing import Optional

from ._clone_project import clone_project

TEMPLATE_REPO_URL = "https://github.com/ywatanabe1989/pip-project-template.git"


def clone_pip_project(
    project_name: str,
    target_dir: Optional[str] = None,
    git_strategy: Optional[str] = "child",
    branch: Optional[str] = None,
    tag: Optional[str] = None,
) -> bool:
    """
    Create a new pip project from the template repository.

    Parameters
    ----------
    project_name : str
        Name of the new pip project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.
    git_strategy : str, optional
        Git initialization strategy ('child', 'parent', None). Default is 'child'.
    branch : str, optional
        Specific branch of the template repository to clone. If None, clones the default branch.
        Mutually exclusive with tag parameter.
    tag : str, optional
        Specific tag/release of the template repository to clone. If None, clones the default branch.
        Mutually exclusive with branch parameter.

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from scitex.template import clone_pip_project
    >>> clone_pip_project("my_pip_project")
    >>> clone_pip_project("my_project", branch="develop")
    >>> clone_pip_project("my_project", tag="v1.0.0")
    """
    return clone_project(
        project_name,
        target_dir,
        TEMPLATE_REPO_URL,
        "pip-project-template",
        git_strategy,
        branch,
        tag,
    )


def main(args: list = None) -> None:
    """
    Command-line interface for clone_pip_project.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If None, uses sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print(
            "Usage: python -m scitex clone_pip_project <project-name> [target-dir]"
        )
        print("")
        print("Arguments:")
        print("  project-name  Name of the new pip project")
        print(
            "  target-dir    Optional: Directory where project will be created (default: current directory)"
        )
        print("")
        print("Example:")
        print("  python -m scitex clone_pip_project my_pip_project")
        print("  python -m scitex clone_pip_project my_project ~/projects")
        sys.exit(1)

    project_name = args[0]
    target_dir = args[1] if len(args) > 1 else None

    success = clone_pip_project(project_name, target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# EOF
