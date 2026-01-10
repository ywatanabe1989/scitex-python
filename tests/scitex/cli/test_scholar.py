#!/usr/bin/env python3
"""Tests for scitex.cli.scholar - Scholar CLI commands."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from scitex.cli.scholar import scholar


class TestScholarGroup:
    """Tests for the scholar command group."""

    def test_scholar_help(self):
        """Test that scholar help is displayed correctly."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["--help"])
        assert result.exit_code == 0
        assert "Literature management" in result.output

    def test_scholar_short_help(self):
        """Test that -h also shows help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["-h"])
        assert result.exit_code == 0
        assert "Literature management" in result.output

    def test_scholar_has_subcommands(self):
        """Test that all expected subcommands are registered."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["--help"])
        expected_commands = ["single", "parallel", "bibtex", "library", "config"]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in scholar help"


class TestScholarSingle:
    """Tests for the scholar single command."""

    def test_single_help(self):
        """Test single command help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["single", "--help"])
        assert result.exit_code == 0
        assert "Process a single paper" in result.output

    def test_single_requires_doi_or_title(self):
        """Test that single command requires either --doi or --title."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["single"])
        assert result.exit_code == 1
        assert "Either --doi or --title is required" in result.output

    def test_single_with_doi(self):
        """Test single command with --doi option."""
        runner = CliRunner()
        # Mock at the source module where it's imported
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineSingle.ScholarPipelineSingle"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline.process_single_paper = MagicMock(
                return_value=(MagicMock(), Path("/tmp/paper.pdf"))
            )
            mock_pipeline_cls.return_value = mock_pipeline

            # Need to mock asyncio.run since process_single_paper is async
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = (MagicMock(), Path("/tmp/paper.pdf"))
                result = runner.invoke(
                    scholar, ["single", "--doi", "10.1038/nature12373"]
                )
                # Command starts processing
                assert "Processing paper" in result.output

    def test_single_with_title(self):
        """Test single command with --title option."""
        runner = CliRunner()
        # Mock at the source module where it's imported
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineSingle.ScholarPipelineSingle"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline

            with patch("asyncio.run") as mock_run:
                mock_run.return_value = (MagicMock(), Path("/tmp/paper.pdf"))
                result = runner.invoke(scholar, ["single", "--title", "Spike sorting"])
                assert "Processing paper" in result.output

    def test_single_with_project(self):
        """Test single command with --project option."""
        runner = CliRunner()
        # Mock pipeline to avoid actual execution
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineSingle.ScholarPipelineSingle"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = (MagicMock(), Path("/tmp/paper.pdf"))
                result = runner.invoke(
                    scholar,
                    [
                        "single",
                        "--doi",
                        "10.1038/nature12373",
                        "--project",
                        "neuroscience",
                    ],
                )
                assert "Processing paper" in result.output

    def test_single_browser_modes(self):
        """Test single command with different browser modes."""
        runner = CliRunner()
        # Mock pipeline to avoid actual execution
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineSingle.ScholarPipelineSingle"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = (MagicMock(), Path("/tmp/paper.pdf"))
                for mode in ["stealth", "interactive"]:
                    result = runner.invoke(
                        scholar,
                        [
                            "single",
                            "--doi",
                            "10.1038/nature12373",
                            "--browser-mode",
                            mode,
                        ],
                    )
                    # Should accept the mode
                    assert "Processing paper" in result.output


class TestScholarParallel:
    """Tests for the scholar parallel command."""

    def test_parallel_help(self):
        """Test parallel command help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["parallel", "--help"])
        assert result.exit_code == 0
        assert "Process multiple papers" in result.output

    def test_parallel_requires_dois_or_titles(self):
        """Test that parallel command requires either --dois or --titles."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["parallel"])
        assert result.exit_code == 1
        assert "Either --dois or --titles is required" in result.output

    def test_parallel_with_dois(self):
        """Test parallel command with --dois option."""
        runner = CliRunner()
        # Mock pipeline to avoid actual execution
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineParallel.ScholarPipelineParallel"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = []
                result = runner.invoke(
                    scholar,
                    [
                        "parallel",
                        "--dois",
                        "10.1038/nature12373",
                        "--dois",
                        "10.1016/j.neuron.2018.01.023",
                    ],
                )
                # Should start processing
                assert "Processing" in result.output

    def test_parallel_with_num_workers(self):
        """Test parallel command with --num-workers option."""
        runner = CliRunner()
        # Mock pipeline to avoid actual execution
        with patch(
            "scitex.scholar.pipelines.ScholarPipelineParallel.ScholarPipelineParallel"
        ) as mock_pipeline_cls:
            mock_pipeline = MagicMock()
            mock_pipeline_cls.return_value = mock_pipeline
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = []
                result = runner.invoke(
                    scholar,
                    [
                        "parallel",
                        "--dois",
                        "10.1038/nature12373",
                        "--num-workers",
                        "8",
                    ],
                )
                assert "Processing" in result.output


class TestScholarBibtex:
    """Tests for the scholar bibtex command."""

    def test_bibtex_help(self):
        """Test bibtex command help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["bibtex", "--help"])
        assert result.exit_code == 0
        assert "Process papers from BibTeX" in result.output

    def test_bibtex_missing_file(self):
        """Test bibtex command without input file."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["bibtex"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_bibtex_nonexistent_file(self):
        """Test bibtex command with non-existent file."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["bibtex", "/nonexistent/file.bib"])
        assert result.exit_code != 0

    def test_bibtex_with_file(self):
        """Test bibtex command with a valid file."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            f.write(b"@article{test, title={Test}}")
            f.flush()

            # Mock pipeline to avoid actual execution
            with patch(
                "scitex.scholar.pipelines.ScholarPipelineBibTeX.ScholarPipelineBibTeX"
            ) as mock_pipeline_cls:
                mock_pipeline = MagicMock()
                mock_pipeline_cls.return_value = mock_pipeline
                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = None
                    result = runner.invoke(scholar, ["bibtex", f.name])
                    # Should start processing
                    assert "Processing BibTeX" in result.output

            os.unlink(f.name)

    def test_bibtex_with_output(self):
        """Test bibtex command with --output option."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix=".bib", delete=False) as f:
            f.write(b"@article{test, title={Test}}")
            f.flush()

            # Mock pipeline to avoid actual execution
            with patch(
                "scitex.scholar.pipelines.ScholarPipelineBibTeX.ScholarPipelineBibTeX"
            ) as mock_pipeline_cls:
                mock_pipeline = MagicMock()
                mock_pipeline_cls.return_value = mock_pipeline
                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = None
                    result = runner.invoke(
                        scholar, ["bibtex", f.name, "--output", "/tmp/enriched.bib"]
                    )
                    assert "Processing BibTeX" in result.output

            os.unlink(f.name)


class TestScholarLibrary:
    """Tests for the scholar library command."""

    def test_library_help(self):
        """Test library command help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["library", "--help"])
        assert result.exit_code == 0
        assert "Show your Scholar library" in result.output

    def test_library_no_library(self):
        """Test library command when no library exists."""
        runner = CliRunner()
        with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
            mock_paths = MagicMock()
            mock_paths.scholar_library = Path("/nonexistent/library")
            mock_get_paths.return_value = mock_paths

            result = runner.invoke(scholar, ["library"])
            assert result.exit_code == 0
            assert "No library found" in result.output

    def test_library_with_papers(self):
        """Test library command with existing library."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = Path(tmpdir)
            master_path = library_path / "MASTER"
            master_path.mkdir()
            # Create some fake paper directories
            (master_path / "paper1").mkdir()
            (master_path / "paper2").mkdir()

            with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
                mock_paths = MagicMock()
                mock_paths.scholar_library = library_path
                mock_get_paths.return_value = mock_paths

                result = runner.invoke(scholar, ["library"])
                assert result.exit_code == 0
                assert "Total papers in library" in result.output

    def test_library_with_project(self):
        """Test library command with --project option."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = Path(tmpdir)
            project_path = library_path / "neuroscience"
            project_path.mkdir()
            # Create a symlink to simulate a paper
            paper_link = project_path / "paper1"
            paper_link.symlink_to(tmpdir)

            with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
                mock_paths = MagicMock()
                mock_paths.scholar_library = library_path
                mock_get_paths.return_value = mock_paths

                result = runner.invoke(
                    scholar, ["library", "--project", "neuroscience"]
                )
                assert result.exit_code == 0
                assert "Project: neuroscience" in result.output

    def test_library_project_not_found(self):
        """Test library command with non-existent project."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = Path(tmpdir)

            with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
                mock_paths = MagicMock()
                mock_paths.scholar_library = library_path
                mock_get_paths.return_value = mock_paths

                result = runner.invoke(scholar, ["library", "--project", "nonexistent"])
                assert result.exit_code == 0
                assert "not found in library" in result.output


class TestScholarConfig:
    """Tests for the scholar config command."""

    def test_config_help(self):
        """Test config command help."""
        runner = CliRunner()
        result = runner.invoke(scholar, ["config", "--help"])
        assert result.exit_code == 0
        assert "Show Scholar configuration" in result.output

    def test_config_display(self):
        """Test config command displays configuration."""
        runner = CliRunner()
        with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
            mock_paths = MagicMock()
            mock_paths.scholar_library = Path("/home/user/.scitex/scholar/library")
            mock_get_paths.return_value = mock_paths

            result = runner.invoke(scholar, ["config"])
            assert result.exit_code == 0
            assert "SciTeX Scholar Configuration" in result.output
            assert "Library location" in result.output

    def test_config_with_existing_library(self):
        """Test config command with existing library."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            library_path = Path(tmpdir)
            master_path = library_path / "MASTER"
            master_path.mkdir()
            (master_path / "paper1").mkdir()

            with patch("scitex.cli.scholar.get_paths") as mock_get_paths:
                mock_paths = MagicMock()
                mock_paths.scholar_library = library_path
                mock_get_paths.return_value = mock_paths

                result = runner.invoke(scholar, ["config"])
                assert result.exit_code == 0
                assert "Library exists: Yes" in result.output
                assert "Papers in library" in result.output


class TestScholarIntegration:
    """Integration tests for scholar commands."""

    def test_help_all_subcommands(self):
        """Test that all subcommands have help text."""
        runner = CliRunner()
        subcommands = ["single", "parallel", "bibtex", "library", "config"]
        for cmd in subcommands:
            result = runner.invoke(scholar, [cmd, "--help"])
            assert result.exit_code == 0, f"Failed for {cmd}"
            assert len(result.output) > 50, f"Help too short for {cmd}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/scholar.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# SciTeX Scholar Commands - CLI wrapper for scitex.scholar module
# 
# Wraps existing Scholar implementation (browser automation, PDF download, metadata enrichment)
# into the unified scitex CLI interface.
# """
# 
# import click
# import sys
# import asyncio
# from pathlib import Path
# 
# from scitex.config import get_paths
# 
# 
# @click.group(context_settings={"help_option_names": ["-h", "--help"]})
# def scholar():
#     """
#     Literature management with browser automation
# 
#     \b
#     Provides scientific literature search, PDF download, and metadata enrichment:
#     - Single or batch paper processing
#     - BibTeX enrichment
#     - Browser automation for PDF downloads
#     - OpenAthens/institutional access support
# 
#     \b
#     Storage: $SCITEX_DIR/scholar/library/
#     Backend: Browser automation + metadata APIs
#     """
#     pass
# 
# 
# @scholar.command()
# @click.option("--doi", help='DOI of the paper (e.g., "10.1038/nature12373")')
# @click.option("--title", help="Paper title (will resolve DOI automatically)")
# @click.option("--project", help="Project name for organizing papers")
# @click.option(
#     "--browser-mode",
#     type=click.Choice(["stealth", "interactive"]),
#     default="stealth",
#     help="Browser mode for PDF download",
# )
# @click.option("--chrome-profile", default="system", help="Chrome profile name")
# @click.option(
#     "--force", "-f", is_flag=True, help="Force re-download even if files exist"
# )
# def single(doi, title, project, browser_mode, chrome_profile, force):
#     """
#     Process a single paper (DOI or title)
# 
#     Downloads PDF, enriches metadata, and stores in library.
# 
#     \b
#     Examples:
#         scitex scholar single --doi "10.1038/nature12373"
#         scitex scholar single --title "Spike sorting" --project neuroscience
#         scitex scholar single --doi "10.1016/j.neuron.2018.01.023" --force
#     """
#     if not doi and not title:
#         click.echo("Error: Either --doi or --title is required", err=True)
#         sys.exit(1)
# 
#     # Import here to avoid slow startup
#     from scitex.scholar.pipelines.ScholarPipelineSingle import ScholarPipelineSingle
# 
#     doi_or_title = doi if doi else title
# 
#     click.echo(f"Processing paper: {doi_or_title}")
# 
#     async def run():
#         pipeline = ScholarPipelineSingle(
#             browser_mode=browser_mode,
#             chrome_profile=chrome_profile,
#         )
# 
#         paper, symlink_path = await pipeline.process_single_paper(
#             doi_or_title=doi_or_title,
#             project=project,
#             force=force,
#         )
# 
#         click.echo(f"✓ Paper processed successfully")
#         if symlink_path:
#             click.echo(f"  Location: {symlink_path}")
# 
#     try:
#         asyncio.run(run())
#     except KeyboardInterrupt:
#         click.echo("\nInterrupted by user", err=True)
#         sys.exit(1)
#     except Exception as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)
# 
# 
# @scholar.command()
# @click.option(
#     "--dois", multiple=True, help="DOIs to process (can be specified multiple times)"
# )
# @click.option(
#     "--titles",
#     multiple=True,
#     help="Paper titles to process (can be specified multiple times)",
# )
# @click.option("--project", help="Project name for organizing papers")
# @click.option("--num-workers", type=int, default=4, help="Number of parallel workers")
# @click.option(
#     "--browser-mode",
#     type=click.Choice(["stealth", "interactive"]),
#     default="stealth",
#     help="Browser mode for all workers",
# )
# @click.option(
#     "--chrome-profile", default="system", help="Base Chrome profile to sync from"
# )
# def parallel(dois, titles, project, num_workers, browser_mode, chrome_profile):
#     """
#     Process multiple papers in parallel
# 
#     Uses multiple browser workers for efficient batch processing.
# 
#     \b
#     Examples:
#         scitex scholar parallel --dois 10.1038/nature12373 --dois 10.1016/j.neuron.2018.01.023
#         scitex scholar parallel --titles "Spike sorting" --titles "Neural networks" --num-workers 8
#         scitex scholar parallel --dois 10.1038/nature12373 --project neuroscience
#     """
#     if not dois and not titles:
#         click.echo("Error: Either --dois or --titles is required", err=True)
#         sys.exit(1)
# 
#     # Import here to avoid slow startup
#     from scitex.scholar.pipelines.ScholarPipelineParallel import ScholarPipelineParallel
# 
#     # Combine DOIs and titles
#     queries = list(dois) + list(titles)
# 
#     click.echo(f"Processing {len(queries)} papers with {num_workers} workers...")
# 
#     async def run():
#         pipeline = ScholarPipelineParallel(
#             num_workers=num_workers,
#             browser_mode=browser_mode,
#             base_chrome_profile=chrome_profile,
#         )
# 
#         papers = await pipeline.process_papers_from_list_async(
#             doi_or_title_list=queries,
#             project=project,
#         )
# 
#         click.echo(f"✓ {len(papers)} papers processed successfully")
# 
#     try:
#         asyncio.run(run())
#     except KeyboardInterrupt:
#         click.echo("\nInterrupted by user", err=True)
#         sys.exit(1)
#     except Exception as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)
# 
# 
# @scholar.command()
# @click.argument("bibtex_file", type=click.Path(exists=True))
# @click.option("--project", help="Project name for organizing papers")
# @click.option(
#     "--output", help="Output path for enriched BibTeX (default: {input}_processed.bib)"
# )
# @click.option("--num-workers", type=int, default=4, help="Number of parallel workers")
# @click.option(
#     "--browser-mode",
#     type=click.Choice(["stealth", "interactive"]),
#     default="stealth",
#     help="Browser mode for all workers",
# )
# @click.option(
#     "--chrome-profile", default="system", help="Base Chrome profile to sync from"
# )
# def bibtex(bibtex_file, project, output, num_workers, browser_mode, chrome_profile):
#     """
#     Process papers from BibTeX file
# 
#     Enriches BibTeX entries with metadata and downloads PDFs in parallel.
# 
#     \b
#     Examples:
#         scitex scholar bibtex papers.bib --project myresearch
#         scitex scholar bibtex refs.bib --output enriched.bib --num-workers 8
#         scitex scholar bibtex library.bib --browser-mode interactive
# 
#     \b
#     TIP: Get BibTeX files from Scholar QA (https://scholarqa.allen.ai/chat/)
#          Ask questions → Export All Citations → Save as .bib file
#     """
#     # Import here to avoid slow startup
#     from scitex.scholar.pipelines.ScholarPipelineBibTeX import ScholarPipelineBibTeX
# 
#     bibtex_path = Path(bibtex_file)
# 
#     click.echo(f"Processing BibTeX file: {bibtex_path}")
# 
#     async def run():
#         pipeline = ScholarPipelineBibTeX(
#             num_workers=num_workers,
#             browser_mode=browser_mode,
#             base_chrome_profile=chrome_profile,
#         )
# 
#         papers = await pipeline.process_bibtex_file_async(
#             bibtex_path=bibtex_path,
#             project=project,
#             output_bibtex_path=output,
#         )
# 
#         click.echo(f"✓ {len(papers)} papers processed from BibTeX")
#         if output:
#             click.echo(f"  Enriched BibTeX saved to: {output}")
# 
#     try:
#         asyncio.run(run())
#     except KeyboardInterrupt:
#         click.echo("\nInterrupted by user", err=True)
#         sys.exit(1)
#     except Exception as e:
#         click.echo(f"Error: {e}", err=True)
#         sys.exit(1)
# 
# 
# @scholar.command()
# @click.option("--project", help="Show library for specific project")
# def library(project):
#     """
#     Show your Scholar library
# 
#     \b
#     Examples:
#         scitex scholar library
#         scitex scholar library --project neuroscience
#     """
#     library_path = get_paths().scholar_library
# 
#     if not library_path.exists():
#         click.echo("No library found. Process some papers first!")
#         return
# 
#     if project:
#         project_path = library_path / project
#         if not project_path.exists():
#             click.echo(f"Project '{project}' not found in library")
#             return
# 
#         click.echo(f"\nProject: {project}")
#         click.echo(f"Location: {project_path}")
#         click.echo("\nPapers:")
# 
#         # List symlinks in project directory
#         for item in sorted(project_path.iterdir()):
#             if item.is_symlink():
#                 click.echo(f"  - {item.name}")
#     else:
#         # List all projects
#         master_path = library_path / "MASTER"
#         project_dirs = [
#             d for d in library_path.iterdir() if d.is_dir() and d.name != "MASTER"
#         ]
# 
#         if master_path.exists():
#             num_papers = len(list(master_path.iterdir()))
#             click.echo(f"\nTotal papers in library: {num_papers}")
# 
#         if project_dirs:
#             click.echo(f"\nProjects ({len(project_dirs)}):")
#             for proj_dir in sorted(project_dirs):
#                 num_papers = len([p for p in proj_dir.iterdir() if p.is_symlink()])
#                 click.echo(f"  - {proj_dir.name} ({num_papers} papers)")
#         else:
#             click.echo("\nNo projects yet. Use --project when processing papers.")
# 
# 
# @scholar.command()
# def config():
#     """
#     Show Scholar configuration
# 
#     Displays library location, browser settings, and authentication status.
#     """
#     library_path = get_paths().scholar_library
# 
#     click.echo("\n=== SciTeX Scholar Configuration ===\n")
#     click.echo(f"Library location: {library_path}")
#     click.echo(f"Library exists: {'Yes' if library_path.exists() else 'No'}")
# 
#     if library_path.exists():
#         master_path = library_path / "MASTER"
#         if master_path.exists():
#             num_papers = len(list(master_path.iterdir()))
#             click.echo(f"Papers in library: {num_papers}")
# 
#     # Check for Chrome profiles
#     chrome_config_path = Path.home() / ".config" / "google-chrome"
#     click.echo(f"\nChrome config: {chrome_config_path}")
#     click.echo(f"Chrome available: {'Yes' if chrome_config_path.exists() else 'No'}")
# 
#     click.echo("\nFor more info, see:")
#     click.echo("  https://github.com/ywatanabe1989/scitex-code")
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/scitex-code/src/scitex/cli/scholar.py
# --------------------------------------------------------------------------------
