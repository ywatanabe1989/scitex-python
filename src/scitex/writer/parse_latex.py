#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/ywatanabe/proj/scitex-code/src/scitex/writer/parse_latex.py

"""
LaTeX error and warning parsing from compilation output and .log files.

Provides detailed error/warning extraction with file, line, and context information.
"""

import re
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from scitex.logging import getLogger

logger = getLogger(__name__)


@dataclass
class LaTeXIssue:
    """Single LaTeX error or warning."""

    type: str  # 'error' or 'warning'
    message: str
    file: str = None
    line: int = None
    context: str = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.file and self.line:
            return f"{self.type.upper()}: {self.file}:{self.line} - {self.message}"
        elif self.file:
            return f"{self.type.upper()}: {self.file} - {self.message}"
        else:
            return f"{self.type.upper()}: {self.message}"


def parse_latex_log(log_file: Path) -> Tuple[List[LaTeXIssue], List[LaTeXIssue]]:
    """
    Parse comprehensive error/warning information from .log file.

    Extracts:
    - Error messages with file and line numbers
    - Warning messages
    - Context from log file

    Args:
        log_file: Path to LaTeX .log file

    Returns:
        Tuple of (errors, warnings)
    """
    errors = []
    warnings = []

    if not log_file.exists():
        logger.debug(f"Log file not found: {log_file}")
        return errors, warnings

    try:
        content = log_file.read_text(encoding='utf-8', errors='replace')
    except Exception as e:
        logger.error(f"Failed to read log file {log_file}: {e}")
        return errors, warnings

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # LaTeX error pattern: "! Error message"
        if line.startswith('!'):
            error_text = line[1:].strip()

            # Extract file and line info from following lines
            file_info = None
            line_num = None

            # Look ahead for file information
            j = i + 1
            while j < len(lines) and j < i + 5:
                current = lines[j]

                # File reference pattern: "l.123" or "<*> filename"
                if 'l.' in current:
                    match = re.search(r'l\.(\d+)', current)
                    if match:
                        line_num = int(match.group(1))

                # Try to extract filename from context
                if '<' in current and '>' in current:
                    match = re.search(r'<([^>]+)>', current)
                    if match:
                        file_info = match.group(1)

                j += 1

            issue = LaTeXIssue(
                type='error',
                message=error_text,
                file=file_info,
                line=line_num,
            )
            errors.append(issue)
            i = j
            continue

        # LaTeX warning pattern: "Warning: ..." or "warning: ..."
        if 'Warning' in line or 'warning' in line.lower():
            match = re.search(r'(?:Warning|warning)[:\s]*(.+)', line)
            if match:
                warning_text = match.group(1).strip()
                issue = LaTeXIssue(
                    type='warning',
                    message=warning_text
                )
                warnings.append(issue)

        i += 1

    return errors, warnings


def parse_compilation_output(
    output: str,
    log_file: Path = None
) -> Tuple[List[LaTeXIssue], List[LaTeXIssue]]:
    """
    Parse errors and warnings from compilation output.

    Combines output parsing with log file parsing for completeness.

    Args:
        output: Compilation output (stdout + stderr)
        log_file: Optional path to .log file for detailed parsing

    Returns:
        Tuple of (error_issues, warning_issues)
    """
    errors = []
    warnings = []

    # Parse log file if available
    if log_file:
        log_errors, log_warnings = parse_latex_log(log_file)
        errors.extend(log_errors)
        warnings.extend(log_warnings)

    # Supplement with output parsing
    for line in output.split('\n'):
        if line.startswith('!'):
            error_text = line[1:].strip()
            issue = LaTeXIssue(type='error', message=error_text)
            # Avoid duplicates
            if issue not in errors:
                errors.append(issue)
        elif 'Warning:' in line or 'warning:' in line.lower():
            match = re.search(r'(?:Warning|warning)[:\s]*(.+)', line)
            if match:
                warning_text = match.group(1).strip()
                issue = LaTeXIssue(type='warning', message=warning_text)
                if issue not in warnings:
                    warnings.append(issue)

    return errors, warnings


def format_issues(issues: List[LaTeXIssue], indent: str = "  ") -> str:
    """
    Format list of issues for display.

    Args:
        issues: List of LaTeX issues
        indent: Indentation string

    Returns:
        Formatted string for display
    """
    if not issues:
        return ""

    lines = []
    for issue in issues:
        lines.append(f"{indent}{issue}")

    return "\n".join(lines)


def summarize_issues(
    errors: List[LaTeXIssue],
    warnings: List[LaTeXIssue]
) -> str:
    """
    Create summary of errors and warnings.

    Args:
        errors: List of errors
        warnings: List of warnings

    Returns:
        Summary string
    """
    summary = []

    if errors:
        summary.append(f"Errors ({len(errors)}):")
        summary.append(format_issues(errors))

    if warnings:
        summary.append(f"Warnings ({len(warnings)}):")
        summary.append(format_issues(warnings))

    if not errors and not warnings:
        summary.append("No errors or warnings found")

    return "\n".join(summary)


__all__ = [
    'LaTeXIssue',
    'parse_latex_log',
    'parse_compilation_output',
    'format_issues',
    'summarize_issues',
]

# EOF
