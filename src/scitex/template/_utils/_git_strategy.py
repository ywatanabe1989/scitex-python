#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_git_strategy.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/template/_git_strategy.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Git strategy handling for template projects.

Manages different git initialization strategies:
- child: Create isolated git in project directory
- parent: Use parent git repository
- origin: Preserve template's original git history
- None: No git initialization
"""

import shutil
from pathlib import Path
from typing import Optional

from scitex.logging import getLogger
import scitex.git

from ._logging_helpers import log_group

logger = getLogger(__name__)


def remove_template_git(project_path: Path) -> None:
    """
    Remove .git directory from cloned template.

    Parameters
    ----------
    project_path : Path
        Path to project directory
    """
    git_dir = project_path / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir)
        logger.debug("Removed template .git directory")


def apply_git_strategy(
    project_path: Path,
    git_strategy: Optional[str],
    template_name: str,
) -> None:
    """
    Apply git initialization strategy to project.

    Parameters
    ----------
    project_path : Path
        Path to project directory
    git_strategy : str or None
        Git strategy to apply:
        - 'child': Create isolated git in project directory
        - 'parent': Use parent git repository
        - 'origin': Preserve template's original git history
        - None: No git initialization
    template_name : str
        Name of template (for commit messages)

    Raises
    ------
    ValueError
        If invalid git strategy provided
    """
    if git_strategy is None:
        logger.debug("Git initialization disabled (git_strategy=None)")
        remove_template_git(project_path)
        return

    if git_strategy == "origin":
        logger.debug("Using 'origin' git strategy, preserving template git history")
        git_dir = project_path / ".git"
        if not git_dir.exists():
            logger.warning("No .git directory found, cannot preserve origin history")
        return

    # Git initialization group
    with log_group("Initializing git repository", "📝") as ctx:
        if git_strategy == "parent":
            # Remove template git first
            remove_template_git(project_path)

            parent_git = scitex.git.find_parent_git(project_path)
            if parent_git:
                ctx.step(f"Using parent repository: {parent_git}")
                return
            else:
                logger.warning(
                    "No parent git repository found. Using 'child' strategy."
                )
                git_strategy = "child"

        if git_strategy == "child":
            # Remove template git first
            remove_template_git(project_path)

            if not scitex.git.git_init(project_path, verbose=False):
                ctx.step("Failed to initialize repository", success=False)
            else:
                scitex.git.setup_branches(project_path, template_name, verbose=False)
                ctx.step("Created branch: main")
                ctx.step("Created branch: develop")
                ctx.step("Initial commit complete")
            return

    raise ValueError(f"Invalid git strategy: {git_strategy}")


__all__ = ["remove_template_git", "apply_git_strategy"]

# EOF
