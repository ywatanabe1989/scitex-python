#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/manuscript_document.py

"""
ManuscriptDocument - document accessor for manuscript structure.

Provides typed access to standard manuscript sections.
"""

from pathlib import Path
from typing import Optional

from .document import Document
from .document_section import DocumentSection


class ManuscriptDocument(Document):
    """
    Manuscript document accessor.

    Provides property access to standard manuscript sections:
    - abstract
    - introduction
    - methods
    - results
    - discussion

    Validates against expected manuscript structure.
    """

    @property
    def abstract(self) -> DocumentSection:
        """Get abstract.tex."""
        return DocumentSection(
            self.dir / "contents" / "abstract.tex", git_root=self.git_root
        )

    @property
    def introduction(self) -> DocumentSection:
        """Get introduction.tex."""
        return DocumentSection(
            self.dir / "contents" / "introduction.tex", git_root=self.git_root
        )

    @property
    def methods(self) -> DocumentSection:
        """Get methods.tex."""
        return DocumentSection(
            self.dir / "contents" / "methods.tex", git_root=self.git_root
        )

    @property
    def results(self) -> DocumentSection:
        """Get results.tex."""
        return DocumentSection(
            self.dir / "contents" / "results.tex", git_root=self.git_root
        )

    @property
    def discussion(self) -> DocumentSection:
        """Get discussion.tex."""
        return DocumentSection(
            self.dir / "contents" / "discussion.tex", git_root=self.git_root
        )


__all__ = ['ManuscriptDocument']

# EOF
