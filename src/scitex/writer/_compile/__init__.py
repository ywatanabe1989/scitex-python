#!/usr/bin/env python3
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

from .._dataclasses import CompilationResult
from ._compile_unified import compile
from ._runner import run_compile
from .manuscript import compile_manuscript
from .revision import compile_revision
from .supplementary import compile_supplementary

__all__ = [
    "compile",
    "run_compile",
    "compile_manuscript",
    "compile_supplementary",
    "compile_revision",
    "CompilationResult",
]

# EOF
