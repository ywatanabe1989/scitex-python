#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 16:40:33 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/revision_document.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/types/revision_document.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
RevisionDocument - dataclass for revision response structure.

Represents the 03_revision/ directory structure.
Provides typed access and verification of revision sections.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .document_section import DocumentSection


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
            self.title = DocumentSection(
                self.root / "title.tex", self.git_root
            )
        if self.authors is None:
            self.authors = DocumentSection(
                self.root / "authors.tex", self.git_root
            )
        if self.keywords is None:
            self.keywords = DocumentSection(
                self.root / "keywords.tex", self.git_root
            )
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
            (is_valid, list_of_issues)
        """
        issues = []

        # Check required directories
        if not self.figures.exists():
            issues.append("Missing figures/")
        if not self.tables.exists():
            issues.append("Missing tables/")
        if not self.latex_styles.exists():
            issues.append("Missing latex_styles/")
        if not self.editor.exists():
            issues.append("Missing editor/")

        return len(issues) == 0, issues


@dataclass
class RevisionDocument:
    """
    Revision response document with validation.

    Represents 03_revision/ directory structure.
    """

    dir: Path
    git_root: Optional[Path] = None

    # Subdirectories
    contents: RevisionContents = None
    archive: Path = None
    docs: Path = None

    # Files
    base: DocumentSection = None
    revision: DocumentSection = None
    readme: DocumentSection = None

    def __post_init__(self):
        """Initialize all components."""
        if self.contents is None:
            self.contents = RevisionContents(
                self.dir / "contents", self.git_root
            )
        if self.archive is None:
            self.archive = self.dir / "archive"
        if self.docs is None:
            self.docs = self.dir / "docs"
        if self.base is None:
            self.base = DocumentSection(self.dir / "base.tex", self.git_root)
        if self.revision is None:
            self.revision = DocumentSection(
                self.dir / "revision.tex", self.git_root
            )
        if self.readme is None:
            self.readme = DocumentSection(
                self.dir / "README.md", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify revision directory structure.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check required directories
        if not (self.dir / "contents").exists():
            issues.append("Missing 03_revision/contents/")
        if not self.archive.exists():
            issues.append("Missing 03_revision/archive/")
        if not self.docs.exists():
            issues.append("Missing 03_revision/docs/")

        # Check contents structure
        contents_valid, content_issues = self.contents.verify_structure()
        if not contents_valid:
            for issue in content_issues:
                issues.append(f"03_revision/contents/{issue}")

        return len(issues) == 0, issues

    def __repr__(self) -> str:
        """String representation."""
        return f"RevisionDocument({self.dir.name})"


__all__ = ["RevisionDocument", "RevisionContents"]

# EOF
