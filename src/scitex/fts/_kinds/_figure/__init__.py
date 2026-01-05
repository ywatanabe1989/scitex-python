#!/usr/bin/env python3
# Timestamp: 2025-12-21
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fts/_kinds/_figure/__init__.py

"""Figure kind - Composite container for multiple elements.

A figure is a container that holds other bundles (plots, tables, text, etc.)
arranged in a layout. It has no payload data of its own.

Structure:
- children/: Contains embedded child bundles
- layout: Defines arrangement (rows, cols, panels)
"""

from ._composite import render_composite

__all__ = ["render_composite"]

# EOF
