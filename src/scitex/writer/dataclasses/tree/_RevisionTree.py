#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:16:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/_RevisionTree.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/tree/_RevisionTree.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
RevisionTree - dataclass for revision directory structure.

Represents the 03_revision/ directory with all subdirectories.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..contents import RevisionContents
from ..core import DocumentSection


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
            self.contents = RevisionContents(self.root / "contents", self.git_root)
        if self.base is None:
            self.base = DocumentSection(self.root / "base.tex", self.git_root)
        if self.revision is None:
            self.revision = DocumentSection(self.root / "revision.tex", self.git_root)
        if self.readme is None:
            self.readme = DocumentSection(self.root / "README.md", self.git_root)
        if self.archive is None:
            self.archive = self.root / "archive"
        if self.docs is None:
            self.docs = self.root / "docs"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify revision structure has required components.

        Returns:
            (is_valid, list_of_missing_items_with_paths)
        """
        missing = []

        # Check contents structure
        contents_valid, contents_issues = self.contents.verify_structure()
        if not contents_valid:
            # Contents already includes full paths, just pass them through
            missing.extend(contents_issues)

        # Check root level files
        if not self.base.path.exists():
            expected_path = (
                self.base.path.relative_to(self.git_root)
                if self.git_root
                else self.base.path
            )
            missing.append(f"base.tex (expected at: {expected_path})")
        if not self.revision.path.exists():
            expected_path = (
                self.revision.path.relative_to(self.git_root)
                if self.git_root
                else self.revision.path
            )
            missing.append(f"revision.tex (expected at: {expected_path})")

        return len(missing) == 0, missing


__all__ = ["RevisionTree"]

# EOF
