#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 05:50:51 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_init_directory.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/_init_directory.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Writer project template management.

Provides functions to create and copy writer project templates.
"""

from typing import Optional

from scitex.logging import getLogger

logger = getLogger(__name__)


def clone_writer_project(
    project_name: str,
    target_dir: Optional[str] = None,
    git_strategy: Optional[str] = "child",
) -> bool:
    """
    Initialize a new writer project directory from scitex-writer template.

    Convenience wrapper for scitex.template.clone_writer_directory.
    Handles template cloning and project setup automatically.

    Args:
        project_name: Name of the new paper directory/project
        target_dir: Directory where the project will be created (optional)
        git_strategy: Git initialization strategy (optional)
            - 'child': Create isolated git in project directory (default)
            - 'parent': Use parent git repository
            - 'origin': Preserve template's original git history
            - None: Disable git initialization

    Returns:
        True if successful, False otherwise

    Examples:
        >>> from scitex.writer import clone_writer_project
        >>> clone_writer_project("my_paper")
        >>> clone_writer_project("my_paper", target_dir="/path/to/papers")
        >>> clone_writer_project("my_paper", git_strategy="parent")
    """
    from scitex.template import clone_writer_directory

    try:
        result = clone_writer_directory(project_name, target_dir, git_strategy)
        return result
    except Exception as e:
        logger.error(f"Failed to initialize writer directory: {e}")
        return False


__all__ = [
    "clone_writer_project",
]

# EOF
