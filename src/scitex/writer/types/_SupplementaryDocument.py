#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 16:40:44 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/supplementary_document.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/types/supplementary_document.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SupplementaryDocument - dataclass for supplementary materials structure.

Represents the 02_supplementary/ directory structure.
Provides typed access and verification of supplementary sections.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ._DocumentSection import DocumentSection


@dataclass
class SupplementaryContents:
    """Contents subdirectory of supplementary (02_supplementary/contents/)."""

    root: Path
    git_root: Optional[Path] = None

    # Core sections
    methods: DocumentSection = None
    results: DocumentSection = None

    # Metadata
    title: DocumentSection = None
    authors: DocumentSection = None
    keywords: DocumentSection = None
    journal_name: DocumentSection = None

    # Files/directories
    figures: Path = None
    tables: Path = None
    bibliography: DocumentSection = None
    latex_styles: Path = None
    wordcount: DocumentSection = None

    def __post_init__(self):
        """Initialize all DocumentSection instances."""
        if self.methods is None:
            self.methods = DocumentSection(
                self.root / "methods.tex", self.git_root
            )
        if self.results is None:
            self.results = DocumentSection(
                self.root / "results.tex", self.git_root
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
        if self.wordcount is None:
            self.wordcount = DocumentSection(
                self.root / "wordcount.tex", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify supplementary contents structure.

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

        return len(issues) == 0, issues


@dataclass
class SupplementaryDocument:
    """
    Supplementary materials document with validation.

    Represents 02_supplementary/ directory structure.
    """

    dir: Path
    git_root: Optional[Path] = None

    # Subdirectories
    contents: SupplementaryContents = None
    archive: Path = None

    # Files
    base: DocumentSection = None
    supplementary: DocumentSection = None
    supplementary_diff: DocumentSection = None
    readme: DocumentSection = None

    def __post_init__(self):
        """Initialize all components."""
        if self.contents is None:
            self.contents = SupplementaryContents(
                self.dir / "contents", self.git_root
            )
        if self.archive is None:
            self.archive = self.dir / "archive"
        if self.base is None:
            self.base = DocumentSection(self.dir / "base.tex", self.git_root)
        if self.supplementary is None:
            self.supplementary = DocumentSection(
                self.dir / "supplementary.tex", self.git_root
            )
        if self.supplementary_diff is None:
            self.supplementary_diff = DocumentSection(
                self.dir / "supplementary_diff.tex", self.git_root
            )
        if self.readme is None:
            self.readme = DocumentSection(
                self.dir / "README.md", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify supplementary directory structure.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check required directories
        if not (self.dir / "contents").exists():
            issues.append("Missing 02_supplementary/contents/")
        if not self.archive.exists():
            issues.append("Missing 02_supplementary/archive/")

        # Check contents structure
        contents_valid, content_issues = self.contents.verify_structure()
        if not contents_valid:
            for issue in content_issues:
                issues.append(f"02_supplementary/contents/{issue}")

        return len(issues) == 0, issues

    def __repr__(self) -> str:
        """String representation."""
        return f"SupplementaryDocument({self.dir.name})"


__all__ = ["SupplementaryDocument", "SupplementaryContents"]

# EOF
