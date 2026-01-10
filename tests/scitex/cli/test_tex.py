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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/tex.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """
# SciTeX CLI - TeX Commands (LaTeX Operations)
# 
# Provides LaTeX compilation and preview utilities.
# """
# 
# import sys
# from pathlib import Path
# 
# import click
# 
# 
# @click.group(context_settings={"help_option_names": ["-h", "--help"]})
# def tex():
#     """
#     LaTeX compilation and utilities
# 
#     \b
#     Commands:
#       compile    Compile LaTeX document to PDF
#       preview    Preview LaTeX document
#       to-vec     Convert to vector format (SVG/PDF)
# 
#     \b
#     Examples:
#       scitex tex compile paper.tex
#       scitex tex preview paper.tex
#       scitex tex to-vec figure.tex --format svg
#     """
#     pass
# 
# 
# @tex.command()
# @click.argument("tex_file", type=click.Path(exists=True))
# @click.option("--output", "-o", type=click.Path(), help="Output PDF path")
# @click.option(
#     "--engine",
#     "-e",
#     type=click.Choice(["pdflatex", "xelatex", "lualatex"]),
#     default="pdflatex",
#     help="LaTeX engine (default: pdflatex)",
# )
# @click.option(
#     "--clean", "-c", is_flag=True, help="Clean auxiliary files after compilation"
# )
# @click.option(
#     "--timeout",
#     type=int,
#     default=300,
#     help="Compilation timeout in seconds (default: 300)",
# )
# @click.option("--verbose", "-v", is_flag=True, help="Show detailed compilation output")
# def compile(tex_file, output, engine, clean, timeout, verbose):
#     """
#     Compile LaTeX document to PDF
# 
#     \b
#     Examples:
#       scitex tex compile paper.tex
#       scitex tex compile paper.tex --output ./output/paper.pdf
#       scitex tex compile paper.tex --engine xelatex
#       scitex tex compile paper.tex --clean --verbose
#     """
#     try:
#         from scitex.tex import compile_tex
# 
#         path = Path(tex_file)
#         click.echo(f"Compiling: {path.name}")
# 
#         result = compile_tex(
#             tex_path=path,
#             output_path=output,
#             engine=engine,
#             timeout=timeout,
#         )
# 
#         if result.success:
#             click.secho("Compilation successful!", fg="green")
#             click.echo(f"PDF: {result.output_pdf}")
# 
#             if clean:
#                 # Clean auxiliary files
#                 aux_extensions = [
#                     ".aux",
#                     ".log",
#                     ".out",
#                     ".toc",
#                     ".bbl",
#                     ".blg",
#                     ".fls",
#                     ".fdb_latexmk",
#                 ]
#                 for ext in aux_extensions:
#                     aux_file = path.with_suffix(ext)
#                     if aux_file.exists():
#                         aux_file.unlink()
#                 click.echo("Auxiliary files cleaned")
#         else:
#             click.secho(
#                 f"Compilation failed (exit code {result.exit_code})", fg="red", err=True
#             )
#             if result.errors and verbose:
#                 click.echo("\nErrors:")
#                 for error in result.errors[:10]:
#                     click.echo(f"  {error}")
#             sys.exit(result.exit_code)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @tex.command()
# @click.argument("tex_file", type=click.Path(exists=True))
# @click.option("--viewer", "-v", help="PDF viewer to use (default: system default)")
# def preview(tex_file, viewer):
#     """
#     Preview LaTeX document (compile and open)
# 
#     \b
#     Examples:
#       scitex tex preview paper.tex
#       scitex tex preview paper.tex --viewer evince
#     """
#     try:
#         from scitex.tex import preview as tex_preview
# 
#         path = Path(tex_file)
#         click.echo(f"Previewing: {path.name}")
# 
#         tex_preview(path, viewer=viewer)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @tex.command("to-vec")
# @click.argument("tex_file", type=click.Path(exists=True))
# @click.option(
#     "--format",
#     "-f",
#     "fmt",
#     type=click.Choice(["svg", "pdf", "eps"]),
#     default="svg",
#     help="Output format (default: svg)",
# )
# @click.option("--output", "-o", type=click.Path(), help="Output file path")
# def to_vec(tex_file, fmt, output):
#     """
#     Convert LaTeX to vector format
# 
#     \b
#     Useful for embedding equations and figures in other documents.
# 
#     \b
#     Examples:
#       scitex tex to-vec equation.tex
#       scitex tex to-vec equation.tex --format pdf
#       scitex tex to-vec figure.tex --output ./vectors/fig.svg
#     """
#     try:
#         from scitex.tex import to_vec as convert_to_vec
# 
#         path = Path(tex_file)
#         click.echo(f"Converting: {path.name} -> {fmt.upper()}")
# 
#         if output:
#             output_path = Path(output)
#         else:
#             output_path = path.with_suffix(f".{fmt}")
# 
#         result = convert_to_vec(path, output_path, format=fmt)
# 
#         if result:
#             click.secho("Conversion successful!", fg="green")
#             click.echo(f"Output: {result}")
#         else:
#             click.secho("Conversion failed", fg="red", err=True)
#             sys.exit(1)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @tex.command()
# @click.argument("tex_file", type=click.Path(exists=True))
# def check(tex_file):
#     """
#     Check LaTeX document for common issues
# 
#     \b
#     Checks:
#       - Missing packages
#       - Undefined references
#       - Overfull/underfull boxes
#       - Citation warnings
# 
#     \b
#     Example:
#       scitex tex check paper.tex
#     """
#     try:
#         import subprocess
#         from pathlib import Path
# 
#         path = Path(tex_file)
#         click.echo(f"Checking: {path.name}")
# 
#         # Run LaTeX in draft mode to check
#         result = subprocess.run(
#             ["pdflatex", "-draftmode", "-interaction=nonstopmode", str(path)],
#             capture_output=True,
#             text=True,
#             cwd=path.parent,
#             timeout=60,
#         )
# 
#         # Parse log for issues
#         log_file = path.with_suffix(".log")
#         issues = {
#             "warnings": [],
#             "errors": [],
#             "overfull": [],
#             "undefined": [],
#         }
# 
#         if log_file.exists():
#             with open(log_file) as f:
#                 for line in f:
#                     if "Warning:" in line:
#                         issues["warnings"].append(line.strip())
#                     elif "Error:" in line or "!" in line[:2]:
#                         issues["errors"].append(line.strip())
#                     elif "Overfull" in line or "Underfull" in line:
#                         issues["overfull"].append(line.strip())
#                     elif "undefined" in line.lower():
#                         issues["undefined"].append(line.strip())
# 
#         # Report
#         total_issues = sum(len(v) for v in issues.values())
# 
#         if total_issues == 0:
#             click.secho("No issues found!", fg="green")
#         else:
#             click.secho(f"Found {total_issues} issue(s):", fg="yellow")
# 
#             if issues["errors"]:
#                 click.secho(f"\nErrors ({len(issues['errors'])}):", fg="red")
#                 for err in issues["errors"][:5]:
#                     click.echo(f"  {err}")
# 
#             if issues["undefined"]:
#                 click.secho(
#                     f"\nUndefined References ({len(issues['undefined'])}):", fg="yellow"
#                 )
#                 for ref in issues["undefined"][:5]:
#                     click.echo(f"  {ref}")
# 
#             if issues["overfull"]:
#                 click.secho(
#                     f"\nOverfull/Underfull Boxes ({len(issues['overfull'])}):",
#                     fg="yellow",
#                 )
#                 for box in issues["overfull"][:5]:
#                     click.echo(f"  {box}")
# 
#             if issues["warnings"]:
#                 click.secho(f"\nWarnings ({len(issues['warnings'])}):", fg="yellow")
#                 for warn in issues["warnings"][:5]:
#                     click.echo(f"  {warn}")
# 
#     except subprocess.TimeoutExpired:
#         click.secho("Check timed out", fg="yellow")
#     except FileNotFoundError:
#         click.secho("Error: pdflatex not found. Install TeX Live.", fg="red", err=True)
#         sys.exit(1)
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# if __name__ == "__main__":
#     tex()

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/tex.py
# --------------------------------------------------------------------------------
