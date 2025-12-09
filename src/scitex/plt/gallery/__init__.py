#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-08 23:30:00 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/plt/gallery/__init__.py

"""
SciTeX Plot Gallery

Generate example plots with CSVs organized by visualization purpose.

Usage:
    import scitex as stx

    # Generate full gallery
    stx.plt.gallery.generate("./gallery")

    # Generate specific category
    stx.plt.gallery.generate("./gallery", category="line")

    # List available plots
    stx.plt.gallery.list()
"""

from ._generate import generate
from ._registry import CATEGORIES, list_plots

__all__ = ["generate", "list_plots", "CATEGORIES"]

# EOF
