#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:13:09 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/tree/_SharedTree.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/types/tree/_SharedTree.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
SharedTree - dataclass for shared directory structure.

Represents the 00_shared/ directory with files used across documents.
"""

from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ..core import DocumentSection


@dataclass
class SharedTree:
    """Shared directory structure (00_shared/)."""

    root: Path
    git_root: Optional[Path] = None

    # Metadata files
    authors: DocumentSection = None
    title: DocumentSection = None
    keywords: DocumentSection = None
    journal_name: DocumentSection = None

    # Directories
    bib_files: Path = None
    latex_styles: Path = None
    templates: Path = None

    # Bibliography
    bibliography: DocumentSection = None

    def __post_init__(self):
        """Initialize all DocumentSection and Path instances."""
        if self.authors is None:
            self.authors = DocumentSection(self.root / "authors.tex", self.git_root)
        if self.title is None:
            self.title = DocumentSection(self.root / "title.tex", self.git_root)
        if self.keywords is None:
            self.keywords = DocumentSection(self.root / "keywords.tex", self.git_root)
        if self.journal_name is None:
            self.journal_name = DocumentSection(
                self.root / "journal_name.tex", self.git_root
            )
        if self.bibliography is None:
            self.bibliography = DocumentSection(
                self.root / "bib_files" / "bibliography.bib", self.git_root
            )
        if self.bib_files is None:
            self.bib_files = self.root / "bib_files"
        if self.latex_styles is None:
            self.latex_styles = self.root / "latex_styles"
        if self.templates is None:
            self.templates = self.root / "templates"

    def verify_structure(self) -> tuple[bool, list[str]]:
        """
        Verify shared structure has required files.

        Returns:
            (is_valid, list_of_missing_files_with_paths)
        """
        required = [
            ("authors.tex", self.authors),
            ("title.tex", self.title),
            ("keywords.tex", self.keywords),
            ("journal_name.tex", self.journal_name),
            ("bibliography.bib", self.bibliography),
        ]

        missing = []
        for name, section in required:
            if not section.path.exists():
                expected_path = (
                    section.path.relative_to(self.git_root)
                    if self.git_root
                    else section.path
                )
                missing.append(f"{name} (expected at: {expected_path})")

        return len(missing) == 0, missing


__all__ = ["SharedTree"]

# EOF
