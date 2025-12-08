#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_validate.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_project/_validate.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Project structure validation for writer module.

Verifies that writer projects have expected directory structure.
"""

from pathlib import Path

from scitex.logging import getLogger

logger = getLogger(__name__)


def validate_structure(project_dir: Path) -> None:
    """
    Verify attached project has expected structure.

    Checks:
    - Required directories exist (01_manuscript, 02_supplementary, 03_revision)

    Parameters
    ----------
    project_dir : Path
        Path to project directory

    Raises
    ------
    RuntimeError
        If structure is invalid
    """
    required_dirs = [
        project_dir / "01_manuscript",
        project_dir / "02_supplementary",
        project_dir / "03_revision",
    ]

    for dir_path in required_dirs:
        if not dir_path.exists():
            logger.error(f"Expected directory missing: {dir_path}")
            raise RuntimeError(
                f"Project structure invalid: missing {dir_path.name} directory (expected at: {dir_path})"
            )

    logger.success(f"Project structure verified at {project_dir.absolute()}")


__all__ = ["validate_structure"]

# EOF
