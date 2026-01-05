#!/usr/bin/env python3
"""Tests for scitex.cli.writer - Writer CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.writer import writer


class TestWriterGroup:
    """Tests for the writer command group."""

    def test_writer_help(self):
        """Test that writer help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(writer, ["--help"])
        assert result.exit_code == 0
        assert "Manuscript writing" in result.output

    def test_writer_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["-h"])
        assert result.exit_code == 0
        assert "Manuscript writing" in result.output

    def test_writer_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(writer, ["--help"])
        expected_commands = ["clone", "compile", "info", "watch"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in writer help"


class TestWriterClone:
    """Tests for the writer clone command."""

    def test_clone_help(self):
        """Test clone command help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["clone", "--help"])
        assert result.exit_code == 0
        assert "Clone a new writer project" in result.output

    def test_clone_missing_argument(self):
        """Test clone command without project directory."""
        runner = CliRunner()
        result = runner.invoke(writer, ["clone"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_clone_basic(self):
        """Test clone command with basic project directory."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            # Mock at the source module where it's imported
            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.return_value = True

                result = runner.invoke(writer, ["clone", str(project_path)])
                assert result.exit_code == 0
                assert "Successfully cloned project" in result.output

    def test_clone_failure(self):
        """Test clone command when cloning fails."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.return_value = False

                result = runner.invoke(writer, ["clone", str(project_path)])
                assert result.exit_code == 1
                assert "Failed to clone" in result.output

    def test_clone_with_git_strategy(self):
        """Test clone command with --git-strategy option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.return_value = True

                for strategy in ["child", "parent", "origin", "none"]:
                    result = runner.invoke(
                        writer,
                        ["clone", str(project_path), "--git-strategy", strategy],
                    )
                    # none strategy becomes None
                    if strategy == "none":
                        expected_strategy = None
                    else:
                        expected_strategy = strategy
                    if mock_clone.call_args:
                        assert (
                            mock_clone.call_args[1]["git_strategy"] == expected_strategy
                        )

    def test_clone_with_branch(self):
        """Test clone command with --branch option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.return_value = True

                result = runner.invoke(
                    writer,
                    ["clone", str(project_path), "--branch", "develop"],
                )
                assert result.exit_code == 0
                if mock_clone.call_args:
                    assert mock_clone.call_args[1]["branch"] == "develop"

    def test_clone_with_tag(self):
        """Test clone command with --tag option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.return_value = True

                result = runner.invoke(
                    writer,
                    ["clone", str(project_path), "--tag", "v1.0.0"],
                )
                assert result.exit_code == 0
                if mock_clone.call_args:
                    assert mock_clone.call_args[1]["tag"] == "v1.0.0"

    def test_clone_branch_and_tag_exclusive(self):
        """Test that --branch and --tag are mutually exclusive."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            result = runner.invoke(
                writer,
                [
                    "clone",
                    str(project_path),
                    "--branch",
                    "develop",
                    "--tag",
                    "v1.0.0",
                ],
            )
            assert result.exit_code == 1
            assert "Cannot specify both" in result.output


class TestWriterCompile:
    """Tests for the writer compile command."""

    def test_compile_help(self):
        """Test compile command help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["compile", "--help"])
        assert result.exit_code == 0
        assert "Compile LaTeX document" in result.output

    def test_compile_document_choices(self):
        """Test compile command accepts valid document types."""
        runner = CliRunner()
        # Check help shows valid choices
        result = runner.invoke(writer, ["compile", "--help"])
        assert "manuscript" in result.output
        assert "supplementary" in result.output
        assert "revision" in result.output

    def test_compile_manuscript(self):
        """Test compile command for manuscript."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = Path(tmpdir) / "manuscript.pdf"
                mock_writer.compile_manuscript.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(
                    writer, ["compile", "manuscript", "--dir", tmpdir]
                )
                assert result.exit_code == 0
                assert "Compilation successful" in result.output

    def test_compile_manuscript_failure(self):
        """Test compile command when compilation fails."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = False
                mock_result.exit_code = 1
                mock_result.errors = ["Undefined control sequence"]
                mock_writer.compile_manuscript.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(
                    writer, ["compile", "manuscript", "--dir", tmpdir]
                )
                assert result.exit_code == 1
                assert "Compilation failed" in result.output

    def test_compile_supplementary(self):
        """Test compile command for supplementary."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = Path(tmpdir) / "supplementary.pdf"
                mock_writer.compile_supplementary.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(
                    writer, ["compile", "supplementary", "--dir", tmpdir]
                )
                assert result.exit_code == 0

    def test_compile_revision(self):
        """Test compile command for revision."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = Path(tmpdir) / "revision.pdf"
                mock_writer.compile_revision.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(writer, ["compile", "revision", "--dir", tmpdir])
                assert result.exit_code == 0

    def test_compile_revision_with_track_changes(self):
        """Test compile command for revision with --track-changes."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = Path(tmpdir) / "revision.pdf"
                mock_writer.compile_revision.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(
                    writer,
                    ["compile", "revision", "--dir", tmpdir, "--track-changes"],
                )
                assert result.exit_code == 0
                mock_writer.compile_revision.assert_called_once()
                call_kwargs = mock_writer.compile_revision.call_args[1]
                assert call_kwargs["track_changes"] is True

    def test_compile_with_timeout(self):
        """Test compile command with --timeout option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_result.output_pdf = Path(tmpdir) / "manuscript.pdf"
                mock_writer.compile_manuscript.return_value = mock_result
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(
                    writer,
                    ["compile", "manuscript", "--dir", tmpdir, "--timeout", "600"],
                )
                assert result.exit_code == 0
                call_kwargs = mock_writer.compile_manuscript.call_args[1]
                assert call_kwargs["timeout"] == 600


class TestWriterInfo:
    """Tests for the writer info command."""

    def test_info_help(self):
        """Test info command help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["info", "--help"])
        assert result.exit_code == 0
        assert "Show project information" in result.output

    def test_info_basic(self):
        """Test info command displays project information."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_writer.project_name = "my_paper"
                mock_writer.project_dir = Path(tmpdir)
                mock_writer.git_root = Path(tmpdir)

                mock_manuscript = MagicMock()
                mock_manuscript.root = Path(tmpdir) / "01_manuscript"
                mock_writer.manuscript = mock_manuscript

                mock_supplementary = MagicMock()
                mock_supplementary.root = Path(tmpdir) / "02_supplementary"
                mock_writer.supplementary = mock_supplementary

                mock_revision = MagicMock()
                mock_revision.root = Path(tmpdir) / "03_revision"
                mock_writer.revision = mock_revision

                mock_writer.get_pdf.return_value = None
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(writer, ["info", "--dir", tmpdir])
                assert result.exit_code == 0
                assert "Project: my_paper" in result.output
                assert "Location:" in result.output
                assert "Documents:" in result.output

    def test_info_with_compiled_pdfs(self):
        """Test info command shows compiled PDFs."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_writer.project_name = "my_paper"
                mock_writer.project_dir = Path(tmpdir)
                mock_writer.git_root = Path(tmpdir)

                mock_manuscript = MagicMock()
                mock_manuscript.root = Path(tmpdir) / "01_manuscript"
                mock_writer.manuscript = mock_manuscript

                mock_supplementary = MagicMock()
                mock_supplementary.root = Path(tmpdir) / "02_supplementary"
                mock_writer.supplementary = mock_supplementary

                mock_revision = MagicMock()
                mock_revision.root = Path(tmpdir) / "03_revision"
                mock_writer.revision = mock_revision

                # Return PDF paths for some documents
                def get_pdf_side_effect(doc_type):
                    if doc_type == "manuscript":
                        return Path(tmpdir) / "manuscript.pdf"
                    return None

                mock_writer.get_pdf.side_effect = get_pdf_side_effect
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(writer, ["info", "--dir", tmpdir])
                assert result.exit_code == 0
                assert "Compiled PDFs" in result.output


class TestWriterWatch:
    """Tests for the writer watch command."""

    def test_watch_help(self):
        """Test watch command help."""
        runner = CliRunner()
        result = runner.invoke(writer, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch for file changes" in result.output

    def test_watch_with_keyboard_interrupt(self):
        """Test watch command handles keyboard interrupt."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer = MagicMock()
                mock_writer.watch.side_effect = KeyboardInterrupt()
                mock_writer_cls.return_value = mock_writer

                result = runner.invoke(writer, ["watch", "--dir", tmpdir])
                assert result.exit_code == 0
                assert "Stopped watching" in result.output


class TestWriterErrorHandling:
    """Tests for writer command error handling."""

    def test_clone_exception(self):
        """Test clone command handles exceptions."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = Path(tmpdir) / "my_paper"

            with patch(
                "scitex.writer._clone_writer_project.clone_writer_project"
            ) as mock_clone:
                mock_clone.side_effect = Exception("Git error")

                result = runner.invoke(writer, ["clone", str(project_path)])
                assert result.exit_code == 1
                assert "Error" in result.output

    def test_compile_exception(self):
        """Test compile command handles exceptions."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer_cls.side_effect = Exception("Project not found")

                result = runner.invoke(
                    writer, ["compile", "manuscript", "--dir", tmpdir]
                )
                assert result.exit_code == 1
                assert "Error" in result.output

    def test_info_exception(self):
        """Test info command handles exceptions."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer_cls.side_effect = Exception("Invalid project")

                result = runner.invoke(writer, ["info", "--dir", tmpdir])
                assert result.exit_code == 1
                assert "Error" in result.output

    def test_watch_exception(self):
        """Test watch command handles exceptions."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("scitex.writer.Writer") as mock_writer_cls:
                mock_writer_cls.side_effect = Exception("Watch failed")

                result = runner.invoke(writer, ["watch", "--dir", tmpdir])
                assert result.exit_code == 1
                assert "Error" in result.output


class TestWriterIntegration:
    """Integration tests for writer commands."""

    def test_help_all_subcommands(self):
        """Test that all subcommands have help text."""
        runner = CliRunner()
        subcommands = ["clone", "compile", "info", "watch"]
        for cmd in subcommands:
            result = runner.invoke(writer, [cmd, "--help"])
            assert result.exit_code == 0, f"Failed for {cmd}"
            assert len(result.output) > 50, f"Help too short for {cmd}"


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__), "-v"])
