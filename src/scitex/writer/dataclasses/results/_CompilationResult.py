#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:35 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_CompilationResult.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_CompilationResult.py"
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


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng = stx.session.start(
        sys,
        plt,
        args=args,
        file=__FILE__,
        sdir_suffix=None,
        verbose=False,
        agg=True,
    )

    exit_status = main(args)

    stx.session.close(
        CONFIG,
        verbose=False,
        notify=False,
        message="",
        exit_status=exit_status,
    )


def main(args):
    result = CompilationResult(
        success=args.success,
        exit_code=args.exit_code,
        stdout="Sample stdout",
        stderr="Sample stderr" if not args.success else "",
        output_pdf=Path(args.pdf) if args.pdf else None,
        duration=args.duration,
        errors=["Error 1", "Error 2"] if not args.success else [],
        warnings=["Warning 1"] if args.warnings else [],
    )

    print(result)
    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Demonstrate CompilationResult dataclass"
    )
    parser.add_argument(
        "--success",
        action="store_true",
        help="Simulate successful compilation",
    )
    parser.add_argument(
        "--exit-code",
        type=int,
        default=1,
        help="Exit code (default: 1)",
    )
    parser.add_argument(
        "--pdf",
        type=str,
        help="Output PDF path",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Compilation duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--warnings",
        action="store_true",
        help="Include warnings",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["CompilationResult"]

# python -m scitex.writer.dataclasses.results._CompilationResult --success --pdf ./manuscript.pdf --duration 10.5

# EOF
