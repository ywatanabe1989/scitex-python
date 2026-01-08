#!/usr/bin/env python3
"""Tests for scitex.cli.tex - LaTeX compilation CLI commands."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.tex import tex


class TestTexGroup:
    """Tests for the tex command group."""

    def test_tex_help(self):
        """Test that tex help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(tex, ["--help"])
        assert result.exit_code == 0
        assert "LaTeX compilation" in result.output

    def test_tex_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(tex, ["--help"])
        expected_commands = ["compile", "preview", "to-vec", "check"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in tex help"


class TestTexCompile:
    """Tests for the tex compile command."""

    def test_compile_success(self):
        """Test compile command with successful compilation."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("scitex.tex.compile_tex") as mock_compile:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = temp_path.replace(".tex", ".pdf")
                mock_compile.return_value = mock_result

                result = runner.invoke(tex, ["compile", temp_path])
                assert result.exit_code == 0
                assert "successful" in result.output.lower()
                mock_compile.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_compile_with_engine(self):
        """Test compile command with specific engine."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("scitex.tex.compile_tex") as mock_compile:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = temp_path.replace(".tex", ".pdf")
                mock_compile.return_value = mock_result

                result = runner.invoke(
                    tex, ["compile", temp_path, "--engine", "xelatex"]
                )
                assert result.exit_code == 0
                call_kwargs = mock_compile.call_args[1]
                assert call_kwargs["engine"] == "xelatex"
        finally:
            os.unlink(temp_path)

    def test_compile_with_output(self):
        """Test compile command with custom output path."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("scitex.tex.compile_tex") as mock_compile:
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = "/tmp/custom.pdf"
                mock_compile.return_value = mock_result

                result = runner.invoke(
                    tex, ["compile", temp_path, "--output", "/tmp/custom.pdf"]
                )
                assert result.exit_code == 0
        finally:
            os.unlink(temp_path)

    def test_compile_failure(self):
        """Test compile command with failed compilation."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("\\invalid latex")
            temp_path = f.name

        try:
            with patch("scitex.tex.compile_tex") as mock_compile:
                mock_result = MagicMock()
                mock_result.success = False
                mock_result.exit_code = 1
                mock_result.errors = ["Undefined control sequence"]
                mock_compile.return_value = mock_result

                result = runner.invoke(tex, ["compile", temp_path])
                assert result.exit_code == 1
                assert "failed" in result.output.lower()
        finally:
            os.unlink(temp_path)

    def test_compile_nonexistent_file(self):
        """Test compile command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(tex, ["compile", "/nonexistent/file.tex"])
        assert result.exit_code != 0


class TestTexPreview:
    """Tests for the tex preview command."""

    def test_preview_basic(self):
        """Test preview command."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("scitex.tex.preview") as mock_preview:
                result = runner.invoke(tex, ["preview", temp_path])
                assert result.exit_code == 0
                mock_preview.assert_called_once()
        finally:
            os.unlink(temp_path)

    def test_preview_with_viewer(self):
        """Test preview command with custom viewer."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("scitex.tex.preview") as mock_preview:
                result = runner.invoke(
                    tex, ["preview", temp_path, "--viewer", "evince"]
                )
                assert result.exit_code == 0
                mock_preview.assert_called_once()
                call_kwargs = mock_preview.call_args[1]
                assert call_kwargs["viewer"] == "evince"
        finally:
            os.unlink(temp_path)


class TestTexToVec:
    """Tests for the tex to-vec command."""

    def test_to_vec_svg(self):
        """Test to-vec command with SVG output."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("$E = mc^2$")
            temp_path = f.name

        try:
            with patch("scitex.tex.to_vec") as mock_convert:
                mock_convert.return_value = temp_path.replace(".tex", ".svg")
                result = runner.invoke(tex, ["to-vec", temp_path])
                assert result.exit_code == 0
                assert "successful" in result.output.lower()
        finally:
            os.unlink(temp_path)

    def test_to_vec_pdf(self):
        """Test to-vec command with PDF format."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("$E = mc^2$")
            temp_path = f.name

        try:
            with patch("scitex.tex.to_vec") as mock_convert:
                mock_convert.return_value = temp_path.replace(".tex", ".pdf")
                result = runner.invoke(tex, ["to-vec", temp_path, "--format", "pdf"])
                assert result.exit_code == 0
        finally:
            os.unlink(temp_path)

    def test_to_vec_with_output(self):
        """Test to-vec command with custom output path."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("$E = mc^2$")
            temp_path = f.name

        try:
            with patch("scitex.tex.to_vec") as mock_convert:
                mock_convert.return_value = "/tmp/custom.svg"
                result = runner.invoke(
                    tex, ["to-vec", temp_path, "--output", "/tmp/custom.svg"]
                )
                assert result.exit_code == 0
        finally:
            os.unlink(temp_path)

    def test_to_vec_failure(self):
        """Test to-vec command handles failure."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write("\\invalid")
            temp_path = f.name

        try:
            with patch("scitex.tex.to_vec") as mock_convert:
                mock_convert.return_value = None
                result = runner.invoke(tex, ["to-vec", temp_path])
                assert result.exit_code == 1
                assert "failed" in result.output.lower()
        finally:
            os.unlink(temp_path)


class TestTexCheck:
    """Tests for the tex check command."""

    def test_check_no_issues(self):
        """Test check command with no issues."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        # Also create an empty log file
        log_path = temp_path.replace(".tex", ".log")

        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Create empty log file
                with open(log_path, "w") as log_f:
                    log_f.write("")

                result = runner.invoke(tex, ["check", temp_path])
                assert result.exit_code == 0
                assert "No issues" in result.output
        finally:
            os.unlink(temp_path)
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_check_with_warnings(self):
        """Test check command with warnings."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        log_path = temp_path.replace(".tex", ".log")

        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)
                # Create log file with warnings
                with open(log_path, "w") as log_f:
                    log_f.write("Warning: Font shape not defined\n")
                    log_f.write("Overfull \\hbox (badness 10000)\n")

                result = runner.invoke(tex, ["check", temp_path])
                assert result.exit_code == 0
                assert "issue" in result.output.lower()
        finally:
            os.unlink(temp_path)
            if os.path.exists(log_path):
                os.unlink(log_path)

    def test_check_pdflatex_not_found(self):
        """Test check command when pdflatex not found."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".tex", delete=False) as f:
            f.write(
                "\\documentclass{article}\n\\begin{document}\nTest\n\\end{document}"
            )
            temp_path = f.name

        try:
            with patch("subprocess.run") as mock_run:
                mock_run.side_effect = FileNotFoundError("pdflatex not found")
                result = runner.invoke(tex, ["check", temp_path])
                assert result.exit_code == 1
                assert (
                    "pdflatex not found" in result.output or "TeX Live" in result.output
                )
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
