#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:46 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_SupplementaryContents.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_SupplementaryContents.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SupplementaryContents - dataclass for supplementary contents structure.

Represents the 02_supplementary/contents/ directory structure.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


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
            self.methods = DocumentSection(self.root / "methods.tex", self.git_root)
        if self.results is None:
            self.results = DocumentSection(self.root / "results.tex", self.git_root)
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
            self.wordcount = DocumentSection(self.root / "wordcount.tex", self.git_root)

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify supplementary contents structure.

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

        return len(issues) == 0, issues


__all__ = ["SupplementaryContents"]

# EOF
