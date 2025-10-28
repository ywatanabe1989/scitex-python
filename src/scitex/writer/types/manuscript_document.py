#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 16:40:17 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/manuscript_document.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/types/manuscript_document.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
ManuscriptDocument - dataclass for manuscript structure with validation.

Represents the 01_manuscript/ directory structure with all expected files.
Provides typed access and verification of manuscript sections.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .document_section import DocumentSection


@dataclass
class ManuscriptContents:
    """Contents subdirectory of manuscript (01_manuscript/contents/)."""

    root: Path

    # Core sections
    abstract: DocumentSection = None
    introduction: DocumentSection = None
    methods: DocumentSection = None
    results: DocumentSection = None
    discussion: DocumentSection = None

    # Metadata
    title: DocumentSection = None
    authors: DocumentSection = None
    keywords: DocumentSection = None
    journal_name: DocumentSection = None

    # Optional sections
    graphical_abstract: DocumentSection = None
    highlights: DocumentSection = None
    data_availability: DocumentSection = None
    additional_info: DocumentSection = None
    wordcount: DocumentSection = None

    # Files/directories
    figures: Path = None
    tables: Path = None
    bibliography: DocumentSection = None
    latex_styles: Path = None

    def __post_init__(self, git_root: Optional[Path] = None):
        """Initialize all DocumentSection instances."""
        if self.abstract is None:
            self.abstract = DocumentSection(
                self.root / "abstract.tex", git_root
            )
        if self.introduction is None:
            self.introduction = DocumentSection(
                self.root / "introduction.tex", git_root
            )
        if self.methods is None:
            self.methods = DocumentSection(self.root / "methods.tex", git_root)
        if self.results is None:
            self.results = DocumentSection(self.root / "results.tex", git_root)
        if self.discussion is None:
            self.discussion = DocumentSection(
                self.root / "discussion.tex", git_root
            )
        if self.title is None:
            self.title = DocumentSection(self.root / "title.tex", git_root)
        if self.authors is None:
            self.authors = DocumentSection(self.root / "authors.tex", git_root)
        if self.keywords is None:
            self.keywords = DocumentSection(
                self.root / "keywords.tex", git_root
            )
        if self.journal_name is None:
            self.journal_name = DocumentSection(
                self.root / "journal_name.tex", git_root
            )
        if self.graphical_abstract is None:
            self.graphical_abstract = DocumentSection(
                self.root / "graphical_abstract.tex", git_root
            )
        if self.highlights is None:
            self.highlights = DocumentSection(
                self.root / "highlights.tex", git_root
            )
        if self.data_availability is None:
            self.data_availability = DocumentSection(
                self.root / "data_availability.tex", git_root
            )
        if self.additional_info is None:
            self.additional_info = DocumentSection(
                self.root / "additional_info.tex", git_root
            )
        if self.wordcount is None:
            self.wordcount = DocumentSection(
                self.root / "wordcount.tex", git_root
            )
        if self.figures is None:
            self.figures = self.root / "figures"
        if self.tables is None:
            self.tables = self.root / "tables"
        if self.bibliography is None:
            self.bibliography = DocumentSection(
                self.root / "bibliography.bib", git_root
            )
        if self.latex_styles is None:
            self.latex_styles = self.root / "latex_styles"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify manuscript structure has required files.

        Returns:
            (is_valid, list_of_missing_files)
        """
        required = [
            ("abstract.tex", self.abstract),
            ("introduction.tex", self.introduction),
            ("methods.tex", self.methods),
            ("results.tex", self.results),
            ("discussion.tex", self.discussion),
        ]

        missing = []
        for name, section in required:
            if not section.path.exists():
                missing.append(name)

        return len(missing) == 0, missing


@dataclass
class ManuscriptDocument:
    """
    Manuscript document accessor with validation.

    Represents 01_manuscript/ directory with all subdirectories and files.
    """

    dir: Path
    git_root: Optional[Path] = None

    # Subdirectories
    contents: ManuscriptContents = None
    archive: Path = None

    # Files
    base: DocumentSection = None
    readme: DocumentSection = None

    def __post_init__(self):
        """Initialize all components."""
        if self.contents is None:
            self.contents = ManuscriptContents(
                self.dir / "contents", self.git_root
            )
        if self.archive is None:
            self.archive = self.dir / "archive"
        if self.base is None:
            self.base = DocumentSection(self.dir / "base.tex", self.git_root)
        if self.readme is None:
            self.readme = DocumentSection(
                self.dir / "README.md", self.git_root
            )

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify manuscript directory structure.

        Returns:
            (is_valid, list_of_issues)
        """
        issues = []

        # Check required directories
        if not (self.dir / "contents").exists():
            issues.append("Missing 01_manuscript/contents/")
        if not self.archive.exists():
            issues.append("Missing 01_manuscript/archive/")

        # Check contents structure
        contents_valid, missing_files = self.contents.verify_structure()
        if not contents_valid:
            for file in missing_files:
                issues.append(f"Missing 01_manuscript/contents/{file}")

        return len(issues) == 0, issues

    def __repr__(self) -> str:
        """String representation."""
        return f"ManuscriptDocument({self.dir.name})"


__all__ = ["ManuscriptDocument", "ManuscriptContents"]

# EOF
