#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:43 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_ManuscriptContents.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/dataclasses/_ManuscriptContents.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
ManuscriptContents - dataclass for manuscript contents structure.

Represents the 01_manuscript/contents/ directory structure with all files.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ._DocumentSection import DocumentSection


@dataclass
class ManuscriptContents:
    """Contents subdirectory of manuscript (01_manuscript/contents/)."""

    root: Path
    git_root: Optional[Path] = None

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

    def __post_init__(self):
        """Initialize all DocumentSection instances."""
        if self.abstract is None:
            self.abstract = DocumentSection(
                self.root / "abstract.tex", self.git_root
            )
        if self.introduction is None:
            self.introduction = DocumentSection(
                self.root / "introduction.tex", self.git_root
            )
        if self.methods is None:
            self.methods = DocumentSection(
                self.root / "methods.tex", self.git_root
            )
        if self.results is None:
            self.results = DocumentSection(
                self.root / "results.tex", self.git_root
            )
        if self.discussion is None:
            self.discussion = DocumentSection(
                self.root / "discussion.tex", self.git_root
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
        if self.graphical_abstract is None:
            self.graphical_abstract = DocumentSection(
                self.root / "graphical_abstract.tex", self.git_root
            )
        if self.highlights is None:
            self.highlights = DocumentSection(
                self.root / "highlights.tex", self.git_root
            )
        if self.data_availability is None:
            self.data_availability = DocumentSection(
                self.root / "data_availability.tex", self.git_root
            )
        if self.additional_info is None:
            self.additional_info = DocumentSection(
                self.root / "additional_info.tex", self.git_root
            )
        if self.wordcount is None:
            self.wordcount = DocumentSection(
                self.root / "wordcount.tex", self.git_root
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


__all__ = ["ManuscriptContents"]

# EOF
