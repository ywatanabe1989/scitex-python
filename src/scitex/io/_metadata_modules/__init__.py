#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/io/_metadata_modules/__init__.py

"""
Image and PDF metadata embedding and extraction for research reproducibility.

This module provides functions to embed and extract metadata from image and PDF files.
Metadata is stored using standard formats:
- PNG: tEXt chunks
- JPEG: EXIF ImageDescription field
- SVG: <metadata> element with scitex namespace
- PDF: XMP metadata (industry standard)

The metadata is stored as JSON strings, allowing flexible dictionary structures.
"""

from ._embed import embed_metadata
from ._read import read_metadata
from ._utils import has_metadata

# Format-specific modules (for direct access if needed)
from .embed_metadata_png import embed_metadata_png
from .embed_metadata_jpeg import embed_metadata_jpeg
from .embed_metadata_svg import embed_metadata_svg
from .embed_metadata_pdf import embed_metadata_pdf
from .read_metadata_png import read_metadata_png
from .read_metadata_jpeg import read_metadata_jpeg
from .read_metadata_svg import read_metadata_svg
from .read_metadata_pdf import read_metadata_pdf

__all__ = [
    "embed_metadata",
    "read_metadata",
    "has_metadata",
    "embed_metadata_png",
    "embed_metadata_jpeg",
    "embed_metadata_svg",
    "embed_metadata_pdf",
    "read_metadata_png",
    "read_metadata_jpeg",
    "read_metadata_svg",
    "read_metadata_pdf",
]

# EOF
