#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_customize.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/template/_customize.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Reference update logic for template customization.

Handles updating all references from template package name to new project name.
"""

from pathlib import Path

from scitex.logging import getLogger

logger = getLogger(__name__)


def update_file_references(
    file_path: Path,
    old_name: str,
    new_name: str,
) -> bool:
    """
    Update references in a single file.

    Parameters
    ----------
    file_path : Path
        Path to file to update
    old_name : str
        Old package name to replace
    new_name : str
        New package name

    Returns
    -------
    bool
        True if file was updated, False otherwise
    """
    if not file_path.exists() or file_path.is_dir():
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Replace template package name with project name
        updated_content = content.replace(old_name, new_name)

        # Only write if content changed
        if updated_content != content:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(updated_content)
            return True
        return False

    except Exception as e:
        logger.warning(f"Failed to update {file_path}: {str(e)}")
        return False


def update_references(
    target_path: Path,
    new_name: str,
    template_package_name: str = "pip_project_template",
) -> int:
    """
    Update all references from template package name to new project name.

    Parameters
    ----------
    target_path : Path
        Path to the project directory
    new_name : str
        New name for the project/package
    template_package_name : str
        Original template package name to be replaced

    Returns
    -------
    int
        Number of files updated
    """
    logger.debug(f"Updating references from {template_package_name} to {new_name}")

    # Files that typically contain package name references
    files_to_update = [
        target_path / "pyproject.toml",
        target_path / "README.md",
        target_path / "Makefile",
    ]

    # Also update all Python files
    for py_file in target_path.rglob("*.py"):
        files_to_update.append(py_file)

    updated_count = 0
    for file_path in files_to_update:
        if update_file_references(file_path, template_package_name, new_name):
            logger.debug(f"Updated references in {file_path.relative_to(target_path)}")
            updated_count += 1

    return updated_count


__all__ = ["update_file_references", "update_references"]

# EOF
