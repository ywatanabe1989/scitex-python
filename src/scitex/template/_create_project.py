#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex_repo/src/scitex/template/_create_project.py

"""
Base functionality for creating projects from template repositories.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from scitex.logging import getLogger

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


def _rename_template_package(target_path: Path, project_name: str, template_package_name: str = "pip_project_template") -> None:
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
                logger.info(f"Updated references in {file_path.relative_to(target_path)}")

        except Exception as e:
            logger.warning(f"Failed to update {file_path}: {str(e)}")


def create_project(
    project_name: str,
    target_dir: Optional[str],
    template_url: str,
    template_name: str,
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
            logger.error(f"Directory already exists: {target_path}")
            logger.error("Please choose a different project name or remove the existing directory")
            return False

        logger.info(f"Creating new project from {template_name}: {project_name}")
        logger.info(f"Target directory: {target_path}")

        # Create temporary directory for cloning
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "template"

            logger.info(f"Cloning template from {template_url}")

            # Clone the template repository
            result = subprocess.run(
                ["git", "clone", template_url, str(temp_path)],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.error(f"Failed to clone template: {result.stderr}")
                return False

            logger.info("Template cloned successfully")

            # Remove .git directory from template to avoid conflicts
            git_dir = temp_path / ".git"
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

        # Initialize new git repository
        logger.info("Initializing new git repository")
        result = subprocess.run(
            ["git", "init"],
            cwd=str(target_path),
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning(f"Failed to initialize git repository: {result.stderr}")
        else:
            logger.info("Git repository initialized successfully")

        # Create initial commit
        logger.info("Creating initial commit")
        subprocess.run(
            ["git", "add", "."],
            cwd=str(target_path),
            capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", f"Initial commit from {template_name}"],
            cwd=str(target_path),
            capture_output=True,
        )

        # Rename branch to main
        logger.info("Renaming branch to main")
        result = subprocess.run(
            ["git", "branch", "-M", "main"],
            cwd=str(target_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to rename branch to main: {result.stderr}")
        else:
            logger.info("Branch renamed to main successfully")

        # Create and switch to develop branch
        logger.info("Creating develop branch")
        result = subprocess.run(
            ["git", "checkout", "-b", "develop"],
            cwd=str(target_path),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to create develop branch: {result.stderr}")
        else:
            logger.info("Switched to develop branch successfully")

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
