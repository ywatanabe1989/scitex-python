#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/supplementary_document.py

"""
SupplementaryDocument - document accessor for supplementary materials.

Provides flexible access to supplementary material sections.
"""

from .document import Document


class SupplementaryDocument(Document):
    """
    Supplementary materials document accessor.

    Provides flexible file access for supplementary materials.
    Uses dynamic attribute lookup via parent Document class:
    - document.figures -> figures.tex
    - document.tables -> tables.tex
    - document.custom -> custom.tex
    """

    pass


__all__ = ['SupplementaryDocument']

# EOF
