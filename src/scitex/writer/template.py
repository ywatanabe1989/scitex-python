#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/template.py

"""
Writer project template management.

Provides functions to initialize writer project directories.
"""

from pathlib import Path
from typing import Optional

from scitex.logging import getLogger

logger = getLogger(__name__)


def init_directory(project_name: str, target_dir: Optional[str] = None) -> bool:
    """
    Initialize a new writer project directory from scitex-writer template.

    Convenience wrapper for scitex.template.create_writer_directory.
    Handles template cloning and project setup automatically.

    Args:
        project_name: Name of the new paper directory/project
        target_dir: Directory where the project will be created (optional)

    Returns:
        True if successful, False otherwise

    Examples:
        >>> from scitex.writer import init_directory
        >>> init_directory("my_paper")
        >>> init_directory("my_paper", target_dir="/path/to/papers")
    """
    from scitex.template import create_writer_directory as _create

    try:
        result = _create(project_name, target_dir)
        return result
    except Exception as e:
        logger.error(f"Failed to initialize writer directory: {e}")
        return False


__all__ = [
    'init_directory',
]

# EOF
