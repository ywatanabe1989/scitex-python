#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:35 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_CompilationResult.py
# ----------------------------------------
from __future__ import annotations
import os
__FILE__ = (
    "./src/scitex/writer/dataclasses/_CompilationResult.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
CompilationResult - dataclass for LaTeX compilation results.
"""

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Optional
from typing import List


@dataclass
class CompilationResult:
    """Result of LaTeX compilation."""

    success: bool
    """Whether compilation succeeded (exit code 0)"""

    exit_code: int
    """Process exit code"""

    stdout: str
    """Standard output from compilation"""

    stderr: str
    """Standard error from compilation"""

    output_pdf: Optional[Path] = None
    """Path to generated PDF (if successful)"""

    diff_pdf: Optional[Path] = None
    """Path to diff PDF with tracked changes (if generated)"""

    log_file: Optional[Path] = None
    """Path to compilation log file"""

    duration: float = 0.0
    """Compilation duration in seconds"""

    errors: List[str] = field(default_factory=list)
    """Parsed LaTeX errors (if any)"""

    warnings: List[str] = field(default_factory=list)
    """Parsed LaTeX warnings (if any)"""

    def __str__(self):
        """Human-readable summary."""
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"Compilation {status} (exit code: {self.exit_code})",
            f"Duration: {self.duration:.2f}s",
        ]
        if self.output_pdf:
            lines.append(f"Output: {self.output_pdf}")
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            lines.append(f"Warnings: {len(self.warnings)}")
        return "\n".join(lines)


__all__ = ["CompilationResult"]

# EOF
