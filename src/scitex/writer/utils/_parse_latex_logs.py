#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-28 17:11:22 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/parse_latex.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/parse_latex.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
LaTeX error and warning parsing from compilation output.

Simple parsing of LaTeX errors and warnings from stdout/stderr.
"""

from pathlib import Path
from typing import List
from typing import Tuple

from ..dataclasses import LaTeXIssue


def parse_compilation_output(
    output: str, log_file: Path = None
) -> Tuple[List[LaTeXIssue], List[LaTeXIssue]]:
    """
    Parse errors and warnings from compilation output.

    Args:
        output: Compilation output (stdout + stderr)
        log_file: Optional path to .log file (unused, for compatibility)

    Returns:
        Tuple of (error_issues, warning_issues)
    """
    errors = []
    warnings = []

    for line in output.split("\n"):
        # LaTeX error pattern: "! Error message"
        if line.startswith("!"):
            error_text = line[1:].strip()
            if error_text:
                errors.append(LaTeXIssue(type="error", message=error_text))

        # LaTeX warning pattern
        elif "warning" in line.lower():
            warnings.append(LaTeXIssue(type="warning", message=line.strip()))

    return errors, warnings


def run_session() -> None:
    """Initialize scitex framework, run main function, and cleanup."""
    global CONFIG, CC, sys, plt, rng
    import sys
    import matplotlib.pyplot as plt
    import scitex as stx

    args = parse_args()

    CONFIG, sys.stdout, sys.stderr, plt, CC, rng_manager = stx.session.start(
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
    if args.file:
        with open(args.file) as f_f:
            output = f_f.read()
    else:
        output = args.text

    errors, warnings = parse_compilation_output(output)

    print(f"Errors: {len(errors)}")
    for err in errors:
        print(f"  - {err}")

    print(f"Warnings: {len(warnings)}")
    for warn in warnings:
        print(f"  - {warn}")

    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse LaTeX compilation output for errors and warnings"
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        help="File containing compilation output",
    )
    parser.add_argument(
        "--text",
        "-t",
        type=str,
        help="Compilation output text",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = [
    "parse_compilation_output",
]

# python -m scitex.writer.utils._parse_latex_logs --file compilation.log

# EOF
