#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/template/create_singularity.py

"""
Create a new singularity project from the singularity_template.
"""

import sys
from typing import Optional

from ._create_project import create_project

TEMPLATE_REPO_URL = "https://github.com/ywatanabe1989/singularity_template.git"


def create_singularity(project_name: str, target_dir: Optional[str] = None) -> bool:
    """
    Create a new singularity project from the template repository.

    Parameters
    ----------
    project_name : str
        Name of the new singularity project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from scitex.template import create_singularity
    >>> create_singularity("my_singularity_project")
    """
    return create_project(
        project_name,
        target_dir,
        TEMPLATE_REPO_URL,
        "singularity_template",
    )


def main(args: list = None) -> None:
    """
    Command-line interface for create_singularity.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If None, uses sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python -m scitex create_singularity_project <project-name> [target-dir]")
        print("")
        print("Arguments:")
        print("  project-name  Name of the new singularity project")
        print("  target-dir    Optional: Directory where project will be created (default: current directory)")
        print("")
        print("Example:")
        print("  python -m scitex create_singularity_project my_singularity_project")
        print("  python -m scitex create_singularity_project my_project ~/projects")
        sys.exit(1)

    project_name = args[0]
    target_dir = args[1] if len(args) > 1 else None

    success = create_singularity(project_name, target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# EOF
