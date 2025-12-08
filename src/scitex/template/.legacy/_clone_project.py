#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 05:56:37 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_clone_project.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/template/_clone_project.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Base functionality for creating projects from template repositories.
"""

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from scitex.logging import getLogger
import scitex.git

logger = getLogger(__name__)


def _copy_tree_skip_broken_symlinks(src: Path, dst: Path) -> None:
    """
    Copy directory tree, skipping broken symlinks.

    Parameters
    ----------
    src : Path
        Source directory
    dst : Path
        Destination directory
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        src_item = src / item.name
        dst_item = dst / item.name

        # Check if it's a symlink and if it's broken
        if src_item.is_symlink():
            try:
                # Try to resolve the symlink
                src_item.resolve(strict=True)
                # If successful, copy it
                if src_item.is_dir():
                    _copy_tree_skip_broken_symlinks(src_item, dst_item)
                else:
                    shutil.copy2(src_item, dst_item)
            except (OSError, FileNotFoundError):
                # Broken symlink - skip it
                logger.warning(f"Skipping broken symlink: {src_item}")
                continue
        elif src_item.is_dir():
            _copy_tree_skip_broken_symlinks(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)


def _rename_template_package(
    target_path: Path,
    project_name: str,
    template_package_name: str = "pip_project_template",
) -> None:
    """
    Rename template package directories and update all references.

    Parameters
    ----------
    target_path : Path
        Path to the project directory
    project_name : str
        New name for the project/package
    template_package_name : str
        Original template package name to be replaced
    """
    # Rename directories
    src_template_dir = target_path / "src" / template_package_name
    tests_template_dir = target_path / "tests" / template_package_name

    if src_template_dir.exists():
        src_new_dir = target_path / "src" / project_name
        logger.info(f"Renaming {src_template_dir} to {src_new_dir}")
        src_template_dir.rename(src_new_dir)

    if tests_template_dir.exists():
        tests_new_dir = target_path / "tests" / project_name
        logger.info(f"Renaming {tests_template_dir} to {tests_new_dir}")
        tests_template_dir.rename(tests_new_dir)

    # Update file contents
    logger.info(f"Updating references from {template_package_name} to {project_name}")

    # Files that typically contain package name references
    files_to_update = [
        target_path / "pyproject.toml",
        target_path / "README.md",
        target_path / "Makefile",
    ]

    # Also update all Python files
    for py_file in target_path.rglob("*.py"):
        files_to_update.append(py_file)

    for file_path in files_to_update:
        if not file_path.exists() or file_path.is_dir():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Replace template package name with project name
            updated_content = content.replace(template_package_name, project_name)

            # Only write if content changed
            if updated_content != content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(updated_content)
                logger.info(
                    f"Updated references in {file_path.relative_to(target_path)}"
                )

        except Exception as e:
            logger.warning(f"Failed to update {file_path}: {str(e)}")


def clone_project(
    project_name: str,
    target_dir: Optional[str],
    template_url: str,
    template_name: str,
    git_strategy: Optional[str] = "child",
) -> bool:
    """
    Generic function to create a project from a template repository.

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

    Returns
    -------
    bool
        True if successful, False otherwise

    Raises
    ------
    FileExistsError
        If the target project directory already exists
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

        logger.info(f"Creating new project from {template_name}: {project_name}")
        logger.info(f"Target directory: {target_path}")

        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "template"

            logger.info(f"Cloning template from {template_url}")

            # Clone the template repository using safe git operations
            if not scitex.git.clone_repo(template_url, temp_path):
                logger.error("Failed to clone template")
                return False

            logger.info("Template cloned successfully")

            # Handle git directory based on strategy
            git_dir = temp_path / ".git"
            if git_strategy == "origin" and git_dir.exists():
                logger.info(
                    "Using 'origin' git strategy, preserving template git history..."
                )
            else:
                if git_dir.exists():
                    shutil.rmtree(git_dir)
                    logger.info("Removed template .git directory")

            # Copy template to target location
            logger.info(f"Copying template to {target_path}")
            _copy_tree_skip_broken_symlinks(temp_path, target_path)

            logger.info("Template copied successfully")

        # Rename template package and update references
        logger.info("Customizing template for project")
        _rename_template_package(target_path, project_name)

        if git_strategy == "origin":
            logger.success(f"Successfully created project: {project_name}")
            logger.info(f"Project location: {target_path}")
            logger.info(f"Preserved original git history from {template_name}")
            return True

        if git_strategy is None:
            logger.info("Git initialization disabled (git_strategy=None)")
            logger.success(f"Successfully created project: {project_name}")
            logger.info(f"Project location: {target_path}")
            return True

        if git_strategy == "parent":
            parent_git = scitex.git.find_parent_git(target_path)
            if parent_git:
                logger.info(f"Found parent git repository: {parent_git}")
                logger.success(f"Successfully created project: {project_name}")
                logger.info(f"Project location: {target_path}")
                logger.info(f"Using parent git at: {parent_git}")
                return True
            else:
                logger.warning(
                    "No parent git repository found. Degrading to 'child' strategy."
                )
                git_strategy = "child"

        if git_strategy == "child":
            logger.info("Initializing new git repository")
            if not scitex.git.git_init(target_path):
                logger.warning("Failed to initialize git repository")
            else:
                logger.info("Git repository initialized successfully")
                logger.info("Setting up branches and initial commit")
                scitex.git.setup_branches(target_path, template_name)

        logger.info(f"Successfully created project: {project_name}")
        logger.info(f"Project location: {target_path}")
        logger.info(f"Current branch: develop")
        logger.info(f"Next steps:")
        logger.info(f"  cd {target_path}")

        return True

    except Exception as e:
        logger.error(f"Failed to create project: {str(e)}")
        return False


# EOF
