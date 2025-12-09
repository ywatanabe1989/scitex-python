#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_create.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_project/_create.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Project creation logic for writer module.

Handles creating new writer projects from template.
"""

from pathlib import Path
from typing import Optional

from scitex.logging import getLogger
from scitex.writer._clone_writer_project import (
    clone_writer_project as _clone_writer_project,
)

logger = getLogger(__name__)


def ensure_project_exists(
    project_dir: Path,
    project_name: str,
    git_strategy: Optional[str] = "child",
    branch: Optional[str] = None,
    tag: Optional[str] = None,
) -> Path:
    """
    Ensure project directory exists, creating it if necessary.

    Parameters
    ----------
    project_dir : Path
        Path to project directory
    project_name : str
        Name of the project
    git_strategy : str or None
        Git initialization strategy
    branch : str, optional
        Specific branch of template repository to clone. If None, clones the default branch.
        Mutually exclusive with tag parameter.
    tag : str, optional
        Specific tag/release of template repository to clone. If None, clones the default branch.
        Mutually exclusive with branch parameter.

    Returns
    -------
    Path
        Path to the project directory

    Raises
    ------
    RuntimeError
        If project creation fails
    """
    if project_dir.exists():
        logger.info(f"Attached to existing project at {project_dir.absolute()}")
        return project_dir

    logger.info(f"Creating new project '{project_name}' at {project_dir.absolute()}")

    # Initialize project directory structure
    success = _clone_writer_project(str(project_dir), git_strategy, branch, tag)

    if not success:
        logger.error(f"Failed to initialize project directory for {project_name}")
        raise RuntimeError(f"Could not create project directory at {project_dir}")

    # Verify project directory was created
    if not project_dir.exists():
        logger.error(f"Project directory {project_dir} was not created")
        raise RuntimeError(f"Project directory {project_dir} was not created")

    logger.success(f"Successfully created project at {project_dir.absolute()}")
    return project_dir


__all__ = ["ensure_project_exists"]

# EOF
