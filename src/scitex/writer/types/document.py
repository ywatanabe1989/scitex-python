#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/document.py

"""
Base Document class for document type accessors.

Provides dynamic file access via attribute lookups.
"""

from pathlib import Path
from typing import Optional

from .document_section import DocumentSection


class Document:
    """
    Base document accessor.

    Provides dynamic file access by mapping attribute names to .tex files:
    - document.introduction -> introduction.tex
    - document.methods -> methods.tex
    - Custom: document.custom_section -> custom_section.tex
    """

    def __init__(self, doc_dir: Path, git_root: Optional[Path] = None):
        """
        Initialize document accessor.

        Args:
            doc_dir: Path to document directory (contains 'contents/' subdirectory)
            git_root: Path to git repository root (optional, for efficiency)
        """
        self.dir = doc_dir
        self.git_root = git_root

    def __getattr__(self, name: str) -> DocumentSection:
        """
        Get file path by name (e.g., introduction -> introduction.tex).

        Args:
            name: Section name without .tex extension

        Returns:
            DocumentSection for the requested file

        Example:
            >>> manuscript = ManuscriptDocument(Path("01_manuscript"))
            >>> manuscript.introduction.read()  # Reads contents/introduction.tex
        """
        if name.startswith('_'):
            # Avoid infinite recursion for private attributes
            raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

        file_path = self.dir / "contents" / f"{name}.tex"
        return DocumentSection(file_path, git_root=self.git_root)

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}({self.dir.name})"


__all__ = ['Document']

# EOF
