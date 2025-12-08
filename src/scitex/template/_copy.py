#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/template/_copy.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/template/_copy.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Directory copy operations for template cloning.

Handles copying template directories with proper symlink handling.
"""

import shutil
from pathlib import Path

from scitex.logging import getLogger

logger = getLogger(__name__)


def copy_tree_skip_broken_symlinks(src: Path, dst: Path) -> None:
    """
    Copy directory tree, preserving symlinks as-is.

    Symlinks are copied with their original target paths (relative or absolute).
    Relative symlinks will work correctly once the full directory tree is copied.

    Parameters
    ----------
    src : Path
        Source directory
    dst : Path
        Destination directory

    Raises
    ------
    OSError
        If copy operation fails
    """
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        src_item = src / item.name
        dst_item = dst / item.name

        # Check if it's a symlink
        if src_item.is_symlink():
            try:
                # Read the symlink target (preserve relative/absolute paths as-is)
                link_target = src_item.readlink()

                # Create the symlink at destination
                # Don't verify target exists - relative symlinks will work once full tree is copied
                dst_item.symlink_to(link_target)
                logger.debug(f"Copied symlink: {src_item.name} -> {link_target}")
            except (OSError, FileNotFoundError) as e:
                # Failed to copy symlink
                logger.warning(f"Failed to copy symlink {src_item}: {e}")
                continue
        elif src_item.is_dir():
            copy_tree_skip_broken_symlinks(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)


def copy_template(src: Path, dst: Path, quiet: bool = False) -> Path:
    """
    Copy template directory to destination.

    Parameters
    ----------
    src : Path
        Source template directory
    dst : Path
        Destination directory
    quiet : bool
        If True, suppress logging messages

    Returns
    -------
    Path
        Path to copied template

    Raises
    ------
    OSError
        If copy operation fails
    """
    if not quiet:
        logger.info(f"Copying template from {src} to {dst}")
    copy_tree_skip_broken_symlinks(src, dst)
    if not quiet:
        logger.info("Template copied successfully")
    return dst


__all__ = ["copy_tree_skip_broken_symlinks", "copy_template"]

# EOF
