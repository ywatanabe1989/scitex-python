#!/usr/bin/env python3
"""Tests for scitex.writer.utils._parse_latex_logs."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses import LaTeXIssue
from scitex.writer.utils._parse_latex_logs import parse_compilation_output


class TestParseCompilationOutputErrors:
    """Tests for parse_compilation_output error detection."""

    def test_parses_error_with_exclamation(self):
        """Verify errors starting with '!' are detected."""
        output = "! Undefined control sequence."
        errors, warnings = parse_compilation_output(output)

        assert len(errors) == 1
        assert errors[0].type == "error"
        assert "Undefined control sequence" in errors[0].message

    def test_parses_multiple_errors(self):
        """Verify multiple errors are detected."""
        output = """! First error
Some other text
! Second error
! Third error"""
        errors, warnings = parse_compilation_output(output)

        assert len(errors) == 3

    def test_skips_empty_error_lines(self):
        """Verify empty error lines are skipped."""
        output = "!\n! Actual error"
        errors, warnings = parse_compilation_output(output)

        assert len(errors) == 1
        assert errors[0].message == "Actual error"

    def test_error_type_is_error(self):
        """Verify error issues have type='error'."""
        output = "! LaTeX Error: Something went wrong."
        errors, warnings = parse_compilation_output(output)

        assert errors[0].type == "error"


class TestParseCompilationOutputWarnings:
    """Tests for parse_compilation_output warning detection."""

    def test_parses_lowercase_warning(self):
        """Verify lowercase 'warning' is detected."""
        output = "Package natbib warning: Citation undefined."
        errors, warnings = parse_compilation_output(output)

        assert len(warnings) == 1
        assert warnings[0].type == "warning"

    def test_parses_uppercase_warning(self):
        """Verify uppercase 'Warning' is detected."""
        output = "LaTeX Warning: Reference undefined."
        errors, warnings = parse_compilation_output(output)

        assert len(warnings) == 1

    def test_parses_mixed_case_warning(self):
        """Verify mixed case 'WARNING' is detected."""
        output = "PACKAGE WARNING: Something not quite right."
        errors, warnings = parse_compilation_output(output)

        assert len(warnings) == 1

    def test_parses_multiple_warnings(self):
        """Verify multiple warnings are detected."""
        output = """LaTeX Warning: First warning
Some text
Package warning: Second warning"""
        errors, warnings = parse_compilation_output(output)

        assert len(warnings) == 2


class TestParseCompilationOutputMixed:
    """Tests for parse_compilation_output with mixed errors and warnings."""

    def test_separates_errors_and_warnings(self):
        """Verify errors and warnings are separated correctly."""
        output = """! Error happened
LaTeX Warning: Warning happened
! Another error
Package warning: Another warning"""
        errors, warnings = parse_compilation_output(output)

        assert len(errors) == 2
        assert len(warnings) == 2

    def test_empty_output_returns_empty_lists(self):
        """Verify empty output returns empty lists."""
        output = ""
        errors, warnings = parse_compilation_output(output)

        assert errors == []
        assert warnings == []

    def test_no_issues_returns_empty_lists(self):
        """Verify output without issues returns empty lists."""
        output = """Processing document.tex
Running pdflatex...
Output written to document.pdf"""
        errors, warnings = parse_compilation_output(output)

        assert errors == []
        assert warnings == []


class TestParseCompilationOutputLogFile:
    """Tests for parse_compilation_output log_file parameter."""

    def test_log_file_parameter_is_optional(self):
        """Verify log_file parameter is optional."""
        output = "! Error"
        errors, warnings = parse_compilation_output(output)

        assert len(errors) == 1

    def test_log_file_parameter_is_ignored(self):
        """Verify log_file parameter is ignored (for compatibility)."""
        output = "! Error"
        log_file = Path("/some/path/document.log")
        errors, warnings = parse_compilation_output(output, log_file)

        assert len(errors) == 1


class TestParseCompilationOutputIssueObjects:
    """Tests for LaTeXIssue objects returned by parse_compilation_output."""

    def test_returns_latex_issue_for_errors(self):
        """Verify errors are LaTeXIssue objects."""
        output = "! Test error"
        errors, warnings = parse_compilation_output(output)

        assert isinstance(errors[0], LaTeXIssue)

    def test_returns_latex_issue_for_warnings(self):
        """Verify warnings are LaTeXIssue objects."""
        output = "Package warning: Test warning"
        errors, warnings = parse_compilation_output(output)

        assert isinstance(warnings[0], LaTeXIssue)

    def test_error_message_strips_exclamation(self):
        """Verify error message has exclamation stripped."""
        output = "! Test error message"
        errors, warnings = parse_compilation_output(output)

        assert errors[0].message == "Test error message"
        assert not errors[0].message.startswith("!")

    def test_warning_message_includes_full_line(self):
        """Verify warning message includes full line."""
        output = "LaTeX Warning: Reference 'fig:test' on page 1 undefined."
        errors, warnings = parse_compilation_output(output)

        assert "LaTeX Warning" in warnings[0].message
        assert "fig:test" in warnings[0].message


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
