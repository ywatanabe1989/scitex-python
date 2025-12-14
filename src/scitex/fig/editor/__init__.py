#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-12-14 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/fig/editor/__init__.py

"""
SciTeX Visual Editor Module

Provides interactive GUI for editing figure styles and annotations.
Supports multiple backends with graceful degradation:
  - flask: Browser-based (Flask) - modern UI
  - dearpygui: GPU-accelerated desktop (fast)
  - qt: Rich desktop (PyQt/PySide)
  - tkinter: Built-in Python (works everywhere)
  - mpl: Minimal matplotlib (always works)
"""

from .edit import edit, save_manual_overrides

__all__ = [
    "edit",
    "save_manual_overrides",
]


# EOF
