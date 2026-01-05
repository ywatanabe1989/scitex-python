#!/usr/bin/env python3
"""Tests for scitex.writer.dataclasses.results._CompilationResult."""

from pathlib import Path

import pytest

from scitex.writer.dataclasses.results._CompilationResult import CompilationResult


class TestCompilationResultCreation:
    """Tests for CompilationResult instantiation."""

    def test_required_fields(self):
        """Verify required fields are set correctly."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="Output",
            stderr="",
        )
        assert result.success is True
        assert result.exit_code == 0
        assert result.stdout == "Output"
        assert result.stderr == ""

    def test_optional_fields_default_to_none_or_empty(self):
        """Verify optional fields have proper defaults."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
        )
        assert result.output_pdf is None
        assert result.diff_pdf is None
        assert result.log_file is None
        assert result.duration == 0.0
        assert result.errors == []
        assert result.warnings == []

    def test_optional_fields_can_be_set(self):
        """Verify optional fields can be explicitly set."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            output_pdf=Path("/tmp/output.pdf"),
            diff_pdf=Path("/tmp/diff.pdf"),
            log_file=Path("/tmp/compile.log"),
            duration=10.5,
            errors=["Error 1"],
            warnings=["Warning 1", "Warning 2"],
        )
        assert result.output_pdf == Path("/tmp/output.pdf")
        assert result.diff_pdf == Path("/tmp/diff.pdf")
        assert result.log_file == Path("/tmp/compile.log")
        assert result.duration == 10.5
        assert result.errors == ["Error 1"]
        assert result.warnings == ["Warning 1", "Warning 2"]


class TestCompilationResultStr:
    """Tests for CompilationResult __str__ method."""

    def test_str_success(self):
        """Verify string representation for successful compilation."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            duration=5.25,
        )
        str_result = str(result)
        assert "SUCCESS" in str_result
        assert "exit code: 0" in str_result
        assert "5.25s" in str_result

    def test_str_failure(self):
        """Verify string representation for failed compilation."""
        result = CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="Error occurred",
            duration=2.0,
        )
        str_result = str(result)
        assert "FAILED" in str_result
        assert "exit code: 1" in str_result

    def test_str_with_output_pdf(self):
        """Verify string representation includes output PDF."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
            output_pdf=Path("/tmp/manuscript.pdf"),
        )
        str_result = str(result)
        assert "Output:" in str_result
        assert "manuscript.pdf" in str_result

    def test_str_with_errors_and_warnings(self):
        """Verify string representation includes error/warning counts."""
        result = CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="",
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
        )
        str_result = str(result)
        assert "Errors: 2" in str_result
        assert "Warnings: 1" in str_result


class TestCompilationResultTypes:
    """Tests for CompilationResult type handling."""

    def test_errors_list_is_mutable(self):
        """Verify errors list can be appended to."""
        result = CompilationResult(
            success=False,
            exit_code=1,
            stdout="",
            stderr="",
        )
        result.errors.append("New error")
        assert "New error" in result.errors

    def test_warnings_list_is_mutable(self):
        """Verify warnings list can be appended to."""
        result = CompilationResult(
            success=True,
            exit_code=0,
            stdout="",
            stderr="",
        )
        result.warnings.append("New warning")
        assert "New warning" in result.warnings


if __name__ == "__main__":
    import os

    pytest.main([os.path.abspath(__file__), "-v"])
