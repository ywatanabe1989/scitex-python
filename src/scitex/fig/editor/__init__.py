#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: ./src/scitex/vis/editor/__init__.py
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

from ._edit import edit, save_manual_overrides

__all__ = [
    "edit",
    "save_manual_overrides",
]

# EOF
