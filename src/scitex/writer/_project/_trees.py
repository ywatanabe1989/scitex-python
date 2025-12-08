#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_project/_trees.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_project/_trees.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Document tree creation for writer module.

Creates and initializes document tree objects for manuscript, supplementary,
revision, and scripts.
"""

from pathlib import Path
from typing import Optional, Tuple

from scitex.logging import getLogger
from scitex.writer.dataclasses import ManuscriptTree, SupplementaryTree, RevisionTree
from scitex.writer.dataclasses.tree import ScriptsTree

logger = getLogger(__name__)


def create_document_trees(
    project_dir: Path,
    git_root: Optional[Path],
) -> Tuple[ManuscriptTree, SupplementaryTree, RevisionTree, ScriptsTree]:
    """
    Create document tree objects for writer project.

    Parameters
    ----------
    project_dir : Path
        Path to project directory
    git_root : Path or None
        Path to git root (for efficiency)

    Returns
    -------
    tuple
        (manuscript, supplementary, revision, scripts) tree objects
    """
    manuscript = ManuscriptTree(project_dir / "01_manuscript", git_root=git_root)
    supplementary = SupplementaryTree(
        project_dir / "02_supplementary", git_root=git_root
    )
    revision = RevisionTree(project_dir / "03_revision", git_root=git_root)
    scripts = ScriptsTree(project_dir / "scripts", git_root=git_root)

    logger.success("Document trees initialized")

    return manuscript, supplementary, revision, scripts


__all__ = ["create_document_trees"]

# EOF
