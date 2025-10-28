#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_clone_project.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = "./src/scitex/template/_clone_project.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Orchestration for creating projects from template repositories.

This module provides high-level orchestration only, delegating to:
- _copy: Directory operations
- _rename: Package renaming
- _customize: Reference updates
- _git_strategy: Git handling
"""

import tempfile
from pathlib import Path
from typing import Optional

from scitex.logging import getLogger
import scitex.git

from ._copy import copy_template
from ._rename import rename_package_directories
from ._customize import update_references
from ._git_strategy import apply_git_strategy, remove_template_git

logger = getLogger(__name__)


def clone_project(
    project_name: str,
    target_dir: Optional[str],
    template_url: str,
    template_name: str,
    git_strategy: Optional[str] = "child",
) -> bool:
    """
    Create a project from a template repository.

    This function orchestrates the entire project creation process:
    1. Validates target directory
    2. Clones template to temporary location
    3. Copies template to target location
    4. Customizes package names and references
    5. Applies git strategy

    Parameters
    ----------
    project_name : str
        Name of the new project
    target_dir : str, optional
        Directory where the project will be created. If None, uses current directory.
    template_url : str
        Git repository URL of the template
    template_name : str
        Name of the template (for logging purposes)
    git_strategy : str, optional
        Git initialization strategy:
        - 'child': Create isolated git in project directory (default)
        - 'parent': Use parent git repository
        - 'origin': Preserve template's original git history
        - None: No git initialization

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Determine target directory
        if target_dir is None:
            target_dir = os.getcwd()

        target_path = Path(target_dir) / project_name

        # Check if target directory already exists
        if target_path.exists():
            if scitex.git.is_cloned_from(target_path, template_url):
                logger.info(f"Project already cloned at: {target_path}")
                return True
            logger.error(f"Directory exists but not from template: {target_path}")
            logger.error(
                "Please choose a different project name or remove the existing directory"
            )
            return False

        logger.info(
            f"Creating new project from {template_name}: {project_name}"
        )
        logger.info(f"Target directory: {target_path}")

        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "template"

            logger.info(f"Cloning template from {template_url}")

            # Clone the template repository
            if not scitex.git.clone_repo(template_url, temp_path):
                logger.error("Failed to clone template")
                return False

            logger.info("Template cloned successfully")

            # Handle git directory based on strategy
            if git_strategy != "origin":
                remove_template_git(temp_path)

            # Copy template to target location
            copy_template(temp_path, target_path)

        # Customize template for project
        logger.info("Customizing template for project")
        rename_package_directories(target_path, project_name)
        update_references(target_path, project_name)

        # Apply git strategy
        apply_git_strategy(target_path, git_strategy, template_name)

        # Success summary
        logger.success(f"Successfully created project: {project_name}")
        logger.info(f"Project location: {target_path}")

        if git_strategy == "child":
            logger.info(f"Current branch: develop")
            logger.info(f"Next steps:")
            logger.info(f"  cd {target_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to create project: {str(e)}")
        return False


__all__ = ["clone_project"]

# EOF
