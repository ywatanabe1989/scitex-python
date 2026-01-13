#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/__init__.py

"""
Type definitions for Writer module.

Provides dataclasses for document structures with clear separation of concerns.
Each document type includes:
- Typed access to sections and files
- Nested Contents dataclass for content/ subdirectory
- verify_structure() method for validation
"""

from .core import Document, DocumentSection
from .contents import ManuscriptContents, SupplementaryContents, RevisionContents
from .config import WriterConfig
from .results import CompilationResult, LaTeXIssue

# Tree structures (internal use)
from .tree import (
    ConfigTree,
    SharedTree,
    ScriptsTree,
    ManuscriptTree,
    SupplementaryTree,
    RevisionTree,
)

__all__ = [
    # Core document dataclasses
    "DocumentSection",
    "Document",
    # Document contents
    "ManuscriptContents",
    "SupplementaryContents",
    "RevisionContents",
    # Configuration and results
    "CompilationResult",
    "WriterConfig",
    "LaTeXIssue",
]

# EOF
