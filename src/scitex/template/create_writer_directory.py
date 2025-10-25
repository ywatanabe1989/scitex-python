#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/create_writer_directory.py

"""
Create a new paper directory from the scitex-writer template.
"""

import sys
from typing import Optional

from ._create_project import create_project

TEMPLATE_REPO_URL = "git@github.com:ywatanabe1989/scitex-writer.git"


def create_writer_directory(project_name: str, target_dir: Optional[str] = None) -> bool:
    """
    Create a new paper directory from the scitex-writer template repository.

    Parameters
    ----------
    project_name : str
        Name of the new paper directory/project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from scitex.template import create_writer_directory
    >>> create_writer_directory("my_paper")
    """
    return create_project(
        project_name,
        target_dir,
        TEMPLATE_REPO_URL,
        "scitex-writer",
    )


def main(args: list = None) -> None:
    """
    Command-line interface for create_writer_directory.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If None, uses sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python -m scitex create_writer_directory <project-name> [target-dir]")
        print("")
        print("Arguments:")
        print("  project-name  Name of the new paper directory/project")
        print("  target-dir    Optional: Directory where project will be created (default: current directory)")
        print("")
        print("Example:")
        print("  python -m scitex create_writer_directory my_paper")
        print("  python -m scitex create_writer_directory my_paper ~/papers")
        sys.exit(1)

    project_name = args[0]
    target_dir = args[1] if len(args) > 1 else None

    success = create_writer_directory(project_name, target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# EOF
