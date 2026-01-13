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
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/writer.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# """
# SciTeX Writer Commands - LaTeX Manuscript Management
# 
# Provides manuscript project initialization and compilation.
# """
# 
# import sys
# from pathlib import Path
# 
# import click
# 
# 
# @click.group(context_settings={"help_option_names": ["-h", "--help"]})
# def writer():
#     """
#     Manuscript writing and LaTeX compilation
# 
#     \b
#     Provides manuscript project management:
#     - Initialize new manuscript projects
#     - Compile manuscript, supplementary, and revision documents
#     - Watch mode for auto-recompilation
#     - Project structure management
#     """
#     pass
# 
# 
# @writer.command()
# @click.argument("project_dir", type=click.Path())
# @click.option(
#     "--git-strategy",
#     "-g",
#     type=click.Choice(["child", "parent", "origin", "none"], case_sensitive=False),
#     default="child",
#     help="Git initialization strategy (default: child)",
# )
# @click.option("--branch", "-b", help="Specific branch of template to clone")
# @click.option("--tag", "-t", help="Specific tag/release of template to clone")
# def clone(project_dir, git_strategy, branch, tag):
#     """
#     Clone a new writer project from template
# 
#     \b
#     Arguments:
#         PROJECT_DIR  Path to project directory (will be created)
# 
#     \b
#     Git Strategies:
#         child   - Create isolated git in project directory (default)
#         parent  - Use parent git repository
#         origin  - Preserve template's original git history
#         none    - Disable git initialization
# 
#     \b
#     Examples:
#         scitex writer clone my_paper
#         scitex writer clone ./papers/my_paper
#         scitex writer clone my_paper --git-strategy parent
#         scitex writer clone my_paper --branch develop
#         scitex writer clone my_paper --tag v1.0.0
#     """
#     try:
#         from scitex.writer._clone_writer_project import clone_writer_project
# 
#         # Validate mutual exclusivity of branch and tag
#         if branch and tag:
#             click.echo("Error: Cannot specify both --branch and --tag", err=True)
#             sys.exit(1)
# 
#         # Convert git_strategy 'none' to None
#         if git_strategy and git_strategy.lower() == "none":
#             git_strategy = None
# 
#         click.echo(f"Cloning writer project: {project_dir}")
# 
#         # Clone writer project
#         result = clone_writer_project(
#             project_dir=project_dir,
#             git_strategy=git_strategy,
#             branch=branch,
#             tag=tag,
#         )
# 
#         if result:
#             project_path = Path(project_dir)
#             click.echo()
#             click.secho(
#                 f"✓ Successfully cloned project at {project_path.absolute()}",
#                 fg="green",
#             )
#             click.echo()
#             click.echo("Project structure:")
#             click.echo(f"  {project_dir}/")
#             click.echo(
#                 "    ├── 00_shared/          # Shared resources (figures, bibliography)"
#             )
#             click.echo("    ├── 01_manuscript/      # Main manuscript")
#             click.echo("    ├── 02_supplementary/   # Supplementary materials")
#             click.echo("    ├── 03_revision/        # Revision documents")
#             click.echo("    └── scripts/            # Compilation scripts")
#             click.echo()
#             click.echo("Next steps:")
#             click.echo(f"  cd {project_dir}")
#             click.echo("  # Edit your manuscript in 01_manuscript/contents/")
#             click.echo("  scitex writer compile manuscript")
#         else:
#             click.secho("✗ Failed to clone project", fg="red", err=True)
#             sys.exit(1)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @writer.command()
# @click.argument(
#     "document",
#     type=click.Choice(
#         ["manuscript", "supplementary", "revision"], case_sensitive=False
#     ),
#     default="manuscript",
# )
# @click.option(
#     "--dir",
#     "-d",
#     type=click.Path(exists=True),
#     help="Project directory (defaults to current directory)",
# )
# @click.option(
#     "--track-changes", is_flag=True, help="Enable change tracking (revision only)"
# )
# @click.option(
#     "--timeout",
#     type=int,
#     default=300,
#     help="Compilation timeout in seconds (default: 300)",
# )
# def compile(document, dir, track_changes, timeout):
#     """
#     Compile LaTeX document to PDF
# 
#     \b
#     Arguments:
#         DOCUMENT  Document type to compile (manuscript|supplementary|revision)
# 
#     \b
#     Examples:
#         scitex writer compile manuscript
#         scitex writer compile manuscript --dir ./my_paper
#         scitex writer compile revision --track-changes
#         scitex writer compile supplementary --timeout 600
#     """
#     from scitex.writer import Writer
# 
#     project_dir = Path(dir) if dir else Path.cwd()
# 
#     # Check if this is a Writer project before trying to attach
#     required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
#     missing = [d for d in required_dirs if not (project_dir / d).exists()]
# 
#     if missing:
#         click.echo(f"Not a Writer project: {project_dir.absolute()}")
#         click.echo(f"Missing directories: {', '.join(missing)}")
#         click.echo()
#         click.echo("To create a new project, run:")
#         click.echo("  scitex writer clone <project-name>")
#         return  # Exit gracefully with code 0
# 
#     try:
#         writer = Writer(project_dir)
# 
#         click.echo(f"Compiling {document} in {project_dir}...")
#         click.echo()
# 
#         # Compile based on document type
#         if document == "manuscript":
#             result = writer.compile_manuscript(timeout=timeout)
#         elif document == "supplementary":
#             result = writer.compile_supplementary(timeout=timeout)
#         elif document == "revision":
#             result = writer.compile_revision(
#                 track_changes=track_changes, timeout=timeout
#             )
# 
#         if result.success:
#             click.secho("✓ Compilation successful!", fg="green")
#             click.echo(f"PDF: {result.output_pdf}")
#         else:
#             click.secho(
#                 f"✗ Compilation failed (exit code {result.exit_code})",
#                 fg="red",
#                 err=True,
#             )
#             if result.errors:
#                 click.echo()
#                 click.echo("Errors:")
#                 for error in result.errors[:10]:  # Show first 10 errors
#                     click.echo(f"  - {error}")
#             sys.exit(result.exit_code)
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @writer.command()
# @click.option(
#     "--dir",
#     "-d",
#     type=click.Path(exists=True),
#     help="Project directory (defaults to current directory)",
# )
# def info(dir):
#     """
#     Show project information
# 
#     \b
#     Examples:
#         scitex writer info
#         scitex writer info --dir ./my_paper
#     """
#     from scitex.writer import Writer
# 
#     project_dir = Path(dir) if dir else Path.cwd()
# 
#     # Check if this is a Writer project before trying to attach
#     required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
#     missing = [d for d in required_dirs if not (project_dir / d).exists()]
# 
#     if missing:
#         click.echo(f"Not a Writer project: {project_dir.absolute()}")
#         click.echo(f"Missing directories: {', '.join(missing)}")
#         return  # Exit gracefully with code 0
# 
#     try:
#         writer = Writer(project_dir)
# 
#         click.echo(f"Project: {writer.project_name}")
#         click.echo(f"Location: {writer.project_dir.absolute()}")
#         click.echo(f"Git root: {writer.git_root}")
#         click.echo()
#         click.echo("Documents:")
#         click.echo(f"  - Manuscript: {writer.manuscript.root}")
#         click.echo(f"  - Supplementary: {writer.supplementary.root}")
#         click.echo(f"  - Revision: {writer.revision.root}")
#         click.echo()
# 
#         # Check for compiled PDFs
#         click.echo("Compiled PDFs:")
#         for doc_type in ["manuscript", "supplementary", "revision"]:
#             pdf = writer.get_pdf(doc_type)
#             if pdf:
#                 click.secho(f"  ✓ {doc_type}: {pdf}", fg="green")
#             else:
#                 click.echo(f"  - {doc_type}: not compiled")
# 
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# @writer.command()
# @click.option(
#     "--dir",
#     "-d",
#     type=click.Path(exists=True),
#     help="Project directory (defaults to current directory)",
# )
# def watch(dir):
#     """
#     Watch for file changes and auto-recompile
# 
#     \b
#     Examples:
#         scitex writer watch
#         scitex writer watch --dir ./my_paper
#     """
#     from scitex.writer import Writer
# 
#     project_dir = Path(dir) if dir else Path.cwd()
# 
#     # Check if this is a Writer project before trying to attach
#     required_dirs = ["01_manuscript", "02_supplementary", "03_revision"]
#     missing = [d for d in required_dirs if not (project_dir / d).exists()]
# 
#     if missing:
#         click.echo(f"Not a Writer project: {project_dir.absolute()}")
#         click.echo(f"Missing directories: {', '.join(missing)}")
#         return  # Exit gracefully with code 0
# 
#     try:
#         writer = Writer(project_dir)
# 
#         click.echo(f"Watching {project_dir} for changes...")
#         click.echo("Press Ctrl+C to stop")
#         click.echo()
# 
#         writer.watch()
# 
#     except KeyboardInterrupt:
#         click.echo()
#         click.echo("Stopped watching")
#     except Exception as e:
#         click.secho(f"Error: {e}", fg="red", err=True)
#         sys.exit(1)
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/writer.py
# --------------------------------------------------------------------------------
