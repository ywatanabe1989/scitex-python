#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 06:08:41 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/dataclasses/_LaTeXIssue.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/dataclasses/_LaTeXIssue.py"
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
    issue = LaTeXIssue(
        type=args.type,
        message=args.message,
    )

    print(issue)
    print(f"\nFormatted: {issue}")
    return 0


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Demonstrate LaTeXIssue dataclass")
    parser.add_argument(
        "--type",
        type=str,
        default="error",
        choices=["error", "warning"],
        help="Issue type (default: error)",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="Undefined control sequence",
        help="Issue message (default: 'Undefined control sequence')",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_session()


__all__ = ["LaTeXIssue"]

# python -m scitex.writer.dataclasses.results._LaTeXIssue --type warning --message "Citation not found"

# EOF
