#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:16:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/_RevisionTree.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/dataclasses/tree/_RevisionTree.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
RevisionTree - dataclass for revision directory structure.

Represents the 03_revision/ directory with all subdirectories.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .._RevisionContents import RevisionContents
from .._DocumentSection import DocumentSection


@dataclass
class RevisionTree:
    """Revision directory structure (03_revision/)."""

    root: Path
    git_root: Optional[Path] = None

    # Contents subdirectory
    contents: RevisionContents = None

    # Root level files
    base: DocumentSection = None
    revision: DocumentSection = None
    readme: DocumentSection = None

    # Directories
    archive: Path = None
    docs: Path = None

    def __post_init__(self):
        """Initialize all instances."""
        if self.contents is None:
            self.contents = RevisionContents(
                self.root / "contents", self.git_root
            )
        if self.base is None:
            self.base = DocumentSection(
                self.root / "base.tex", self.git_root
            )
        if self.revision is None:
            self.revision = DocumentSection(
                self.root / "revision.tex", self.git_root
            )
        if self.readme is None:
            self.readme = DocumentSection(
                self.root / "README.md", self.git_root
            )
        if self.archive is None:
            self.archive = self.root / "archive"
        if self.docs is None:
            self.docs = self.root / "docs"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify revision structure has required components.

        Returns:
            (is_valid, list_of_missing_items)
        """
        missing = []

        # Check contents structure
        contents_valid, contents_issues = self.contents.verify_structure()
        if not contents_valid:
            missing.extend([f"contents/{item}" for item in contents_issues])

        # Check root level files
        if not self.base.path.exists():
            missing.append("base.tex")
        if not self.revision.path.exists():
            missing.append("revision.tex")

        return len(missing) == 0, missing


__all__ = ["RevisionTree"]

# EOF
