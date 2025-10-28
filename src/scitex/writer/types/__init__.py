#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/__init__.py

"""
Type definitions for Writer module.

Provides dataclasses for document structures with clear separation of concerns.
Each document type includes:
- Typed access to sections and files
- Nested Contents dataclass for content/ subdirectory
- verify_structure() method for validation
"""

from ._DocumentSection import DocumentSection
from ._Document import Document
from ._ManuscriptDocument import ManuscriptDocument, ManuscriptContents
from ._SupplementaryDocument import SupplementaryDocument, SupplementaryContents
from ._RevisionDocument import RevisionDocument, RevisionContents

__all__ = [
    # Core types
    'DocumentSection',
    'Document',

    # Manuscript
    'ManuscriptDocument',
    'ManuscriptContents',

    # Supplementary
    'SupplementaryDocument',
    'SupplementaryContents',

    # Revision
    'RevisionDocument',
    'RevisionContents',
]

# EOF
