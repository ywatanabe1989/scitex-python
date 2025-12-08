#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_rename.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/template/_rename.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Package renaming logic for template customization.

Handles renaming template package directories (e.g., pip_project_template -> my_project).
"""

from pathlib import Path

from scitex.logging import getLogger

logger = getLogger(__name__)


def rename_package_directories(
    target_path: Path,
    new_name: str,
    template_package_name: str = "pip_project_template",
) -> None:
    """
    Rename template package directories.

    Parameters
    ----------
    target_path : Path
        Path to the project directory
    new_name : str
        New name for the project/package
    template_package_name : str
        Original template package name to be replaced

    Raises
    ------
    OSError
        If rename operation fails
    """
    # Rename src directory
    src_template_dir = target_path / "src" / template_package_name
    if src_template_dir.exists():
        src_new_dir = target_path / "src" / new_name
        logger.info(f"Renaming {src_template_dir} to {src_new_dir}")
        src_template_dir.rename(src_new_dir)

    # Rename tests directory
    tests_template_dir = target_path / "tests" / template_package_name
    if tests_template_dir.exists():
        tests_new_dir = target_path / "tests" / new_name
        logger.info(f"Renaming {tests_template_dir} to {tests_new_dir}")
        tests_template_dir.rename(tests_new_dir)


__all__ = ["rename_package_directories"]

# EOF
