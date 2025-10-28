#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/__init__.py

"""
Type definitions for Writer module.

Provides dataclasses for document structures with clear separation of concerns.
"""

from .document_section import DocumentSection
from .document import Document
from .manuscript_document import ManuscriptDocument
from .supplementary_document import SupplementaryDocument
from .revision_document import RevisionDocument

__all__ = [
    'DocumentSection',
    'Document',
    'ManuscriptDocument',
    'SupplementaryDocument',
    'RevisionDocument',
]

# EOF
