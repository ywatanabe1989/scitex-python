#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:41 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_LaTeXIssue.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/dataclasses/_LaTeXIssue.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeXIssue - dataclass for LaTeX compilation issues.
"""

from dataclasses import dataclass


@dataclass
class LaTeXIssue:
    """Single LaTeX error or warning."""

    type: str  # 'error' or 'warning'
    message: str

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"{self.type.upper()}: {self.message}"


__all__ = ["LaTeXIssue"]

# EOF
