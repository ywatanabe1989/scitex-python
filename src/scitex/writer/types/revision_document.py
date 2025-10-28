#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/types/revision_document.py

"""
RevisionDocument - document accessor for revision responses.

Provides flexible access to revision response sections.
"""

from .document import Document


class RevisionDocument(Document):
    """
    Revision response document accessor.

    Provides flexible file access for revision responses.
    Uses dynamic attribute lookup via parent Document class:
    - document.response -> response.tex
    - document.detailed_response -> detailed_response.tex
    - document.point_by_point -> point_by_point.tex
    """

    pass


__all__ = ['RevisionDocument']

# EOF
