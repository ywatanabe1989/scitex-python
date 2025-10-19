#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/template/create_research.py

"""
Create a new research project from the scitex_template_research template.
"""

import sys
from typing import Optional

from ._create_project import create_project

TEMPLATE_REPO_URL = "https://github.com/ywatanabe1989/scitex_template_research.git"


def create_research(project_name: str, target_dir: Optional[str] = None) -> bool:
    """
    Create a new research project from the template repository.

    Parameters
    ----------
    project_name : str
        Name of the new research project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.

    Returns
    -------
    bool
        True if successful, False otherwise

    Example
    -------
    >>> from scitex.template import create_research
    >>> create_research("my_research_project")
    """
    return create_project(
        project_name,
        target_dir,
        TEMPLATE_REPO_URL,
        "scitex_template_research",
    )


def main(args: list = None) -> None:
    """
    Command-line interface for create_research.

    Parameters
    ----------
    args : list, optional
        Command-line arguments. If None, uses sys.argv[1:]
    """
    if args is None:
        args = sys.argv[1:]

    if len(args) < 1:
        print("Usage: python -m scitex create_research_project <project-name> [target-dir]")
        print("")
        print("Arguments:")
        print("  project-name  Name of the new research project")
        print("  target-dir    Optional: Directory where project will be created (default: current directory)")
        print("")
        print("Example:")
        print("  python -m scitex create_research_project my_research_project")
        print("  python -m scitex create_research_project my_project ~/projects")
        sys.exit(1)

    project_name = args[0]
    target_dir = args[1] if len(args) > 1 else None

    success = create_research(project_name, target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# EOF
