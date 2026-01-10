#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:44 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_RevisionContents.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_RevisionContents.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
RevisionContents - dataclass for revision contents structure.

Represents the 03_revision/contents/ directory structure.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


@dataclass
class RevisionContents:
    """Contents subdirectory of revision (03_revision/contents/)."""

    root: Path
    git_root: Optional[Path] = None

    # Core sections
    introduction: DocumentSection = None
    conclusion: DocumentSection = None
    references: DocumentSection = None

    # Metadata
    title: DocumentSection = None
    authors: DocumentSection = None
    keywords: DocumentSection = None
    journal_name: DocumentSection = None

    # Reviewer responses (subdirectories)
    editor: Path = None
    reviewer1: Path = None
    reviewer2: Path = None

    # Files/directories
    figures: Path = None
    tables: Path = None
    bibliography: DocumentSection = None
    latex_styles: Path = None

    def __post_init__(self):
        """Initialize all DocumentSection instances."""
        if self.introduction is None:
            self.introduction = DocumentSection(
                self.root / "introduction.tex", self.git_root
            )
        if self.conclusion is None:
            self.conclusion = DocumentSection(
                self.root / "conclusion.tex", self.git_root
            )
        if self.references is None:
            self.references = DocumentSection(
                self.root / "references.tex", self.git_root
            )
        if self.title is None:
            self.title = DocumentSection(self.root / "title.tex", self.git_root)
        if self.authors is None:
            self.authors = DocumentSection(self.root / "authors.tex", self.git_root)
        if self.keywords is None:
            self.keywords = DocumentSection(self.root / "keywords.tex", self.git_root)
        if self.journal_name is None:
            self.journal_name = DocumentSection(
                self.root / "journal_name.tex", self.git_root
            )
        if self.editor is None:
            self.editor = self.root / "editor"
        if self.reviewer1 is None:
            self.reviewer1 = self.root / "reviewer1"
        if self.reviewer2 is None:
            self.reviewer2 = self.root / "reviewer2"
        if self.figures is None:
            self.figures = self.root / "figures"
        if self.tables is None:
            self.tables = self.root / "tables"
        if self.bibliography is None:
            self.bibliography = DocumentSection(
                self.root / "bibliography.bib", self.git_root
            )
        if self.latex_styles is None:
            self.latex_styles = self.root / "latex_styles"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify revision contents structure.

        Returns:
            (is_valid, list_of_issues_with_paths)
        """
        issues = []

        # Check required directories
        if not self.figures.exists():
            expected_path = (
                self.figures.relative_to(self.git_root)
                if self.git_root
                else self.figures
            )
            issues.append(f"Missing figures/ (expected at: {expected_path})")
        if not self.tables.exists():
            expected_path = (
                self.tables.relative_to(self.git_root) if self.git_root else self.tables
            )
            issues.append(f"Missing tables/ (expected at: {expected_path})")
        if not self.latex_styles.exists():
            expected_path = (
                self.latex_styles.relative_to(self.git_root)
                if self.git_root
                else self.latex_styles
            )
            issues.append(f"Missing latex_styles/ (expected at: {expected_path})")
        if not self.editor.exists():
            expected_path = (
                self.editor.relative_to(self.git_root) if self.git_root else self.editor
            )
            issues.append(f"Missing editor/ (expected at: {expected_path})")

        return len(issues) == 0, issues


__all__ = ["RevisionContents"]

# EOF
