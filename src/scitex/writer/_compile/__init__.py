#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-11-08 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/__init__.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/__init__.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeX compilation module for writer.

Provides organized compilation functionality:
- manuscript: Manuscript compilation with figure/conversion options
- supplementary: Supplementary materials compilation
- revision: Revision response compilation with change tracking
- _runner: Script execution engine
- _parser: Output parsing utilities
- _validator: Pre-compile validation
"""

from ._runner import run_compile
from ..dataclasses import CompilationResult
from .manuscript import compile_manuscript
from .supplementary import compile_supplementary
from .revision import compile_revision


__all__ = [
    "run_compile",
    "compile_manuscript",
    "compile_supplementary",
    "compile_revision",
    "CompilationResult",
]

# EOF
