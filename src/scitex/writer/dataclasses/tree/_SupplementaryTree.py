#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:16:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/tree/_SupplementaryTree.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/tree/_SupplementaryTree.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SupplementaryTree - dataclass for supplementary directory structure.

Represents the 02_supplementary/ directory with all subdirectories.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..contents import SupplementaryContents
from ..core import DocumentSection


@dataclass
class SupplementaryTree:
    """Supplementary directory structure (02_supplementary/)."""

    root: Path
    git_root: Optional[Path] = None

    # Contents subdirectory
    contents: SupplementaryContents = None

    # Root level files
    base: DocumentSection = None
    supplementary: DocumentSection = None
    supplementary_diff: DocumentSection = None
    readme: DocumentSection = None

    # Directories
    archive: Path = None

    def __post_init__(self):
        """Initialize all instances."""
        if self.contents is None:
            self.contents = SupplementaryContents(self.root / "contents", self.git_root)
        if self.base is None:
            self.base = DocumentSection(self.root / "base.tex", self.git_root)
        if self.supplementary is None:
            self.supplementary = DocumentSection(
                self.root / "supplementary.tex", self.git_root
            )
        if self.supplementary_diff is None:
            self.supplementary_diff = DocumentSection(
                self.root / "supplementary_diff.tex", self.git_root
            )
        if self.readme is None:
            self.readme = DocumentSection(self.root / "README.md", self.git_root)
        if self.archive is None:
            self.archive = self.root / "archive"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify supplementary structure has required components.

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
        if not self.supplementary.path.exists():
            expected_path = (
                self.supplementary.path.relative_to(self.git_root)
                if self.git_root
                else self.supplementary.path
            )
            missing.append(f"supplementary.tex (expected at: {expected_path})")

        return len(missing) == 0, missing


__all__ = ["SupplementaryTree"]

# EOF
