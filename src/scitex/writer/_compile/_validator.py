#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_validator.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/_validator.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Pre-compile validation for writer projects.

Validates project structure before attempting compilation.
"""

from pathlib import Path

from scitex.logging import getLogger
from scitex.writer._validate_tree_structures import validate_tree_structures

logger = getLogger(__name__)


def validate_before_compile(project_dir: Path) -> None:
    """
    Validate project structure before compilation.

    Parameters
    ----------
    project_dir : Path
        Path to project directory

    Raises
    ------
    RuntimeError
        If validation fails
    """
    validate_tree_structures(project_dir)


__all__ = ["validate_before_compile"]

# EOF
