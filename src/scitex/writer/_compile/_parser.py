#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-10-29 (ywatanabe)"
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/_compile/_parser.py
# ----------------------------------------
from __future__ import annotations
import os

__FILE__ = "./src/scitex/writer/_compile/_parser.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
Compilation output parsing.

Parses LaTeX compilation output and log files for errors and warnings.
"""

from pathlib import Path
from typing import Tuple, List, Optional

from scitex.logging import getLogger
from scitex.writer.utils._parse_latex_logs import parse_compilation_output

logger = getLogger(__name__)


def parse_output(
    stdout: str,
    stderr: str,
    log_file: Optional[Path] = None,
) -> Tuple[List[str], List[str]]:
    """
    Parse compilation output for errors and warnings.

    Parameters
    ----------
    stdout : str
        Standard output from compilation
    stderr : str
        Standard error from compilation
    log_file : Path, optional
        Path to LaTeX log file

    Returns
    -------
    tuple
        (errors, warnings) as lists of strings
    """
    error_issues, warning_issues = parse_compilation_output(
        stdout + stderr, log_file=log_file
    )

    # Convert LaTeXIssue objects to strings for backward compatibility
    errors = [str(issue) for issue in error_issues]
    warnings = [str(issue) for issue in warning_issues]

    return errors, warnings


__all__ = ["parse_output"]

# EOF
