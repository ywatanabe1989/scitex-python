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
from ._logging_helpers import log_group, log_step, log_final

logger = getLogger(__name__)


def clone_project(
    project_dir: str,
    template_url: str,
    template_name: str,
    git_strategy: Optional[str] = "child",
    branch: Optional[str] = None,
    tag: Optional[str] = None,
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
    project_dir : str
        Path to project directory (will be created). Can be a simple name like "my_paper"
        or a full path like "./papers/my_paper"
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
    """
    try:
        # Parse project_dir into name and parent directory
        project_path = Path(project_dir)
        project_name = project_path.name
        target_dir_path = (
            project_path.parent if project_path.parent != Path(".") else Path.cwd()
        )
        target_path = target_dir_path / project_name

        # Check if target directory already exists
        if target_path.exists():
            if scitex.git.is_cloned_from(target_path, template_url):
                log_final(f"Project already exists at {target_path}")
                return True
            logger.error(f"Directory already exists: {target_path}")
            logger.error(f"Cannot clone from {template_url}")
            logger.error(
                "Please choose a different project name or remove the existing directory"
            )
            return False

        # Setup project structure
        with log_group("Setting up project structure", "📦") as ctx:
            target_dir_path.mkdir(parents=True, exist_ok=True)
            ctx.step(f"Target directory: {target_dir_path}")

            # Create temporary directory for cloning
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir) / "template"

                ref_info = ""
                if branch:
                    ref_info = f" (branch: {branch})"
                elif tag:
                    ref_info = f" (tag: {tag})"

                # Clone the template repository
                ctx.substep(f"Cloning from {template_url}{ref_info}...")
                if not scitex.git.clone_repo(
                    template_url, temp_path, branch=branch, tag=tag, verbose=False
                ):
                    ctx.step(f"Failed to clone to {target_path}", success=False)
                    return False
                ctx.step(f"Cloned to {target_path}")

                # Handle git directory based on strategy
                if git_strategy != "origin":
                    remove_template_git(temp_path)

                # Copy template to target location
                copy_template(temp_path, target_path, quiet=True)
                ctx.step("Copied template files")

        # Customize template for project
        with log_group("Customizing template", "🔧") as ctx:
            rename_package_directories(target_path, project_name)
            updated_count = update_references(target_path, project_name)
            if updated_count > 0:
                ctx.step(f"Updated {updated_count} references to {project_name}")
            else:
                ctx.step("No references to update")

        # Apply git strategy
        apply_git_strategy(target_path, git_strategy, template_name)

        # Success summary
        log_final(f"Successfully created project at {target_path}")
        logger.info("")
        logger.info("Next steps:")
        logger.info(f"  cd {target_path}")
        if git_strategy == "child":
            logger.info(f"  # Edit your manuscript in 01_manuscript/contents/")
            logger.info(f"  scitex writer compile manuscript")

        return True

    except Exception as e:
        logger.error(f"Failed to create project: {str(e)}")
        return False


__all__ = ["clone_project"]

# EOF
