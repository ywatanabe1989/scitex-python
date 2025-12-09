#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:36 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_CONSTANTS.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_CONSTANTS.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Constants for writer module.

Centralized definitions for document dataclasses and directory mappings.
"""

# Document type to directory mapping
DOC_TYPE_DIRS = {
    "manuscript": "01_manuscript",
    "supplementary": "02_supplementary",
    "revision": "03_revision",
}

# Document type to command-line flag mapping
DOC_TYPE_FLAGS = {
    "manuscript": "-m",
    "supplementary": "-s",
    "revision": "-r",
}

# Document type to PDF filename mapping
DOC_TYPE_PDFS = {
    "manuscript": "manuscript.pdf",
    "supplementary": "supplementary.pdf",
    "revision": "revision.pdf",
}

__all__ = [
    "DOC_TYPE_DIRS",
    "DOC_TYPE_FLAGS",
    "DOC_TYPE_PDFS",
]

# EOF
